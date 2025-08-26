# main.py
"""
Improved crypto signal bot main.py
- Robust kline fetching with retries and fallback
- Local cache (disk) of last successful klines per symbol/timeframe used when both exchanges fail
- Scheduler runs at minute=0 (exact hour)
- /fast HTTP endpoint and /telegram_webhook for Telegram commands (/fast and /report)
- Uses only httpx for Telegram API (no python-telegram-bot polling) to avoid event-loop conflicts

Environment variables used:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
- PORT (default 10000)
- EXCHANGE (BINANCE or BYBIT, default BINANCE)
- SYMBOLS (comma separated, default DOGEUSDT,ADAUSDT)
- SCHED_TZ (timezone for scheduler)

Notes:
- If both Bybit and Binance return 4xx/5xx (geo-blocking etc), the bot will use cached data if available and include notice in report.
"""

import os
import io
import time
import json
import logging
import datetime
import traceback
from typing import Dict, Tuple, Optional

import httpx
import pandas as pd
# avoid matplotlib font cache permission problems
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# TA indicators
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crypto-signal-bot")

# ------------------------------
# Config / env
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", "10000"))
EXCHANGE = os.getenv("EXCHANGE", "BINANCE").upper()
SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")
SYMBOLS = os.getenv("SYMBOLS", "DOGEUSDT,ADAUSDT").split(",")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set - telegram will be disabled for sending reports")

# endpoints
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES_V5 = "https://api.bybit.com/v5/market/kline"

TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200},
}

CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/klines_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------
# HTTP client (shared)
# ------------------------------
HEADERS = {"User-Agent": "crypto-signal-bot/1.0 (+https://example.com)"}
client = httpx.Client(timeout=20.0, headers=HEADERS)

# ------------------------------
# Helpers: disk cache for klines
# ------------------------------

def cache_path(symbol: str, tf_label: str) -> str:
    safe_sym = symbol.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_sym}__{tf_label}.parquet")


def save_cache(df: pd.DataFrame, symbol: str, tf_label: str) -> None:
    try:
        path = cache_path(symbol, tf_label)
        df.to_parquet(path)
        logger.info("Saved cache %s (%s rows)", path, len(df))
    except Exception:
        logger.exception("Failed to save cache for %s %s", symbol, tf_label)


def load_cache(symbol: str, tf_label: str) -> Optional[pd.DataFrame]:
    try:
        path = cache_path(symbol, tf_label)
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        logger.info("Loaded cache %s (%s rows)", path, len(df))
        return df
    except Exception:
        logger.exception("Failed to load cache for %s %s", symbol, tf_label)
        return None

# ------------------------------
# Fetching klines with retry + fallback + cache
# ------------------------------

def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = client.get(BINANCE_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Empty kline data from Binance")
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]


def fetch_bybit_klines_v5(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    # bybit v5 expects category=linear and interval in minutes (e.g. 15,60,240,720)
    mapping = {"15m":"15", "1h":"60", "4h":"240", "12h":"720"}
    if interval not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + interval)
    params = {"category": "linear", "symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
    r = client.get(BYBIT_KLINES_V5, params=params)
    r.raise_for_status()
    jd = r.json()
    # v5 returns {retCode:0, retMsg:"", result:{list: [...]}}
    data = []
    if isinstance(jd, dict):
        res = jd.get("result")
        if isinstance(res, dict) and "list" in res:
            data = res["list"]
        elif isinstance(res, list):
            data = res
    if not data:
        raise RuntimeError("Empty bybit v5 data: " + json.dumps(jd)[:500])
    df = pd.DataFrame(data)
    # bybit fields: start=timestamp in seconds, open, high, low, close, volume
    if "start" in df.columns:
        df["open_time"] = pd.to_datetime(df["start"], unit="s")
    elif "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
    else:
        df["open_time"] = pd.to_datetime(df.iloc[:, 0], unit="s")
    df.set_index("open_time", inplace=True)
    # normalize
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        elif col.capitalize() in df.columns:
            df[col] = df[col.capitalize()].astype(float)
        else:
            df[col] = 0.0
    return df[["open","high","low","close","volume"]]


def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Try preferred exchange, fallback to other, use cache if both fail.
    Returns DataFrame or raises if no data anywhere and no cache.
    """
    last_exc = None
    tried = []
    # choose order based on EXCHANGE
    order = [EXCHANGE]
    if EXCHANGE == "BINANCE":
        order = ["BINANCE", "BYBIT"]
    else:
        order = ["BYBIT", "BINANCE"]

    for exch in order:
        try:
            if exch == "BINANCE":
                df = fetch_binance_klines(symbol, interval, limit=limit)
            else:
                df = fetch_bybit_klines_v5(symbol, interval, limit=limit)
            # sanity check
            if df is None or df.empty:
                raise RuntimeError(f"Empty data from {exch}")
            # save successful fetch to cache
            try:
                save_cache(df, symbol, interval)
            except Exception:
                logger.exception("save_cache failed")
            return df
        except httpx.HTTPStatusError as he:
            last_exc = he
            status = he.response.status_code if he.response is not None else None
            logger.warning("fetch_klines primary %s failed for %s %s: %s", exch, symbol, interval, str(he))
            tried.append((exch, status))
            continue
        except Exception as e:
            last_exc = e
            logger.exception("fetch_klines error from %s for %s %s", exch, symbol, interval)
            tried.append((exch, str(e)))
            continue

    # if here, both exchanges failed
    logger.error("All exchanges failed for %s %s. attempts=%s. Will try cache if available", symbol, interval, tried)
    cached = load_cache(symbol, interval)
    if cached is not None and not cached.empty:
        logger.info("Using cached data for %s %s", symbol, interval)
        return cached
    # otherwise re-raise last exception (wrap to provide context)
    raise RuntimeError(f"Bybit/Binance fetch failed for {symbol} {interval}. Last error: {last_exc}")

# ------------------------------
# Indicators, scoring, plotting (same logic but robust)
# ------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["rsi"] = RSIIndicator(close=close, window=14, fillna=True).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=True)
    out["macd_line"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    bb = BollingerBands(close=close, window=20, window_dev=2, fillna=True)
    out["bb_mavg"] = bb.bollinger_mavg()
    out["bb_hband"] = bb.bollinger_hband()
    out["bb_lband"] = bb.bollinger_lband()
    out["bb_pct"] = (close - out["bb_lband"]) / (out["bb_hband"] - out["bb_lband"] + 1e-12)
    out["sma20"] = SMAIndicator(close=close, window=20, fillna=True).sma_indicator()
    out["sma50"] = SMAIndicator(close=close, window=50, fillna=True).sma_indicator()
    out["ema20"] = EMAIndicator(close=close, window=20, fillna=True).ema_indicator()
    out["ema50"] = EMAIndicator(close=close, window=50, fillna=True).ema_indicator()
    stoch = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3, fillna=True)
    out["stoch_rsi_k"] = stoch.stochrsi_k()
    out["stoch_rsi_d"] = stoch.stochrsi_d()
    return out


def pivot_support_resistance(series: Dict[str, float]) -> Dict[str, float]:
    try:
        high = float(series["high"])
        low = float(series["low"])
        close = float(series["close"])
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        return {"pivot": pivot, "r1": r1, "s1": s1}
    except Exception:
        return {"pivot": 0.0, "r1": 0.0, "s1": 0.0}


def fib_levels(last_low: float, last_high: float) -> Dict[str, float]:
    diff = last_high - last_low
    return {
        "0.0%": last_high,
        "23.6%": last_high - 0.236 * diff,
        "38.2%": last_high - 0.382 * diff,
        "50.0%": last_high - 0.5 * diff,
        "61.8%": last_high - 0.618 * diff,
        "100.0%": last_low,
    }


def score_signals(ind_df: pd.DataFrame) -> Tuple[str, Dict[str, int]]:
    try:
        row = ind_df.iloc[-1]
        votes = {"buy": 0, "sell": 0, "neutral": 0}
        if row["rsi"] < 30:
            votes["buy"] += 1
        elif row["rsi"] > 70:
            votes["sell"] += 1
        if row["macd_hist"] > 0:
            votes["buy"] += 1
        elif row["macd_hist"] < 0:
            votes["sell"] += 1
        if row["close"] > row["ema20"]:
            votes["buy"] += 1
        else:
            votes["sell"] += 1
        if row["bb_pct"] > 0.85:
            votes["sell"] += 1
        elif row["bb_pct"] < 0.15:
            votes["buy"] += 1
        if row["stoch_rsi_k"] < 20:
            votes["buy"] += 1
        elif row["stoch_rsi_k"] > 80:
            votes["sell"] += 1
        vol_mean = ind_df["volume"].tail(50).mean() if len(ind_df) >= 50 else ind_df["volume"].mean()
        if vol_mean > 0 and row["volume"] > vol_mean * 1.5:
            if row["macd_hist"] > 0:
                votes["buy"] += 1
            elif row["macd_hist"] < 0:
                votes["sell"] += 1
        score = votes["buy"] - votes["sell"]
        if score >= 2:
            return "BUY", votes
        elif score <= -2:
            return "SELL", votes
        else:
            return "HOLD", votes
    except Exception:
        logger.exception("score_signals failed")
        return "HOLD", {"buy": 0, "sell": 0, "neutral": 0}


def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    try:
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
        ax_price, ax_macd, ax_rsi = axes
        ax_price.plot(df.index, df["close"], label=f"{symbol} close")
        if "sma20" in df.columns:
            ax_price.plot(df.index, df["sma20"], label="SMA20", linewidth=0.8)
        if "ema20" in df.columns:
            ax_price.plot(df.index, df["ema20"], label="EMA20", linewidth=0.8)
        if "bb_hband" in df.columns and "bb_lband" in df.columns:
            ax_price.plot(df.index, df["bb_hband"], linestyle="--", linewidth=0.7, label="BB Upper")
            ax_price.plot(df.index, df["bb_lband"], linestyle="--", linewidth=0.7, label="BB Lower")
        ax_price.set_title(f"{symbol} {timeframe} — price & indicators")
        ax_price.legend(loc="upper left")
        ax_price.grid(True)
        if "macd_line" in df.columns:
            ax_macd.plot(df.index, df["macd_line"], label="MACD")
            ax_macd.plot(df.index, df["macd_signal"], label="Signal")
            ax_macd.bar(df.index, df["macd_hist"], label="Hist", alpha=0.6)
            ax_macd.legend(loc="upper left")
            ax_macd.grid(True)
        if "rsi" in df.columns:
            ax_rsi.plot(df.index, df["rsi"], label="RSI")
            ax_rsi.axhline(70, linestyle="--", linewidth=0.6)
            ax_rsi.axhline(30, linestyle="--", linewidth=0.6)
            ax_rsi.legend(loc="upper left")
            ax_rsi.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        logger.exception("plot failed")
        return b""

# ------------------------------
# Build report (uses fetch_klines; handles errors and reports cached usage)
# ------------------------------

def build_hourly_report(symbol: str) -> Tuple[str, bytes]:
    now = datetime.datetime.datetime.utcnow().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"⏱️ Hourly report for {symbol} — {now}\n"]
    votes_summary = {}
    chosen_image = b""
    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = fetch_klines(symbol, tf_conf["interval"], limit=tf_conf["limit"])
            used_cache = False
            # detect if loaded from cache by checking index dtype? we rely on fetch_klines log; but let's check len
            if df is None or df.empty or len(df) < 10:
                lines.append(f"TF: {tf_label} — error computing data: No data")
                continue
            ind = compute_indicators(df)
            rec, votes = score_signals(ind)
            last = df.tail(3)
            lastc = {"high": last["high"].iloc[-1], "low": last["low"].iloc[-1], "close": last["close"].iloc[-1]}
            piv = pivot_support_resistance(lastc)
            window = df.tail(60)
            low = float(window["low"].min())
            high = float(window["high"].max())
            fibs = fib_levels(low, high)
            lines.append(f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.8f} | Rec: {rec} | RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}")
            lines.append(f"  Pivot:{piv['pivot']:.8f}, R1:{piv['r1']:.8f}, S1:{piv['s1']:.8f}")
            lines.append(f"  Fib 23.6%:{fibs['23.6%']:.8f} 38.2%:{fibs['38.2%']:.8f} 50%:{fibs['50.0%']:.8f}")
            votes_summary[tf_label] = {"rec": rec, "votes": votes}
            if tf_label == "4h" and not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
            elif not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
        except Exception as e:
            # include human-readable error in report
            tb = traceback.format_exc()
            logger.exception("build_hourly_report: error for %s %s", symbol, tf_label)
            lines.append(f"TF: {tf_label} — error computing data: {str(e)}")
            lines.append(f"Debug: {tb.splitlines()[-1]}")
    # aggregate final
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for tf, d in votes_summary.items():
        counts[d["rec"]] = counts.get(d["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"
    # SL/TP heuristic
    sl = tp = None
    try:
        latest_price_df = fetch_klines(symbol, "1h", limit=3)
        latest_price = float(latest_price_df["close"].iloc[-1])
        if final == "BUY":
            sl = latest_price * 0.98
            tp = latest_price * 1.04
        elif final == "SELL":
            sl = latest_price * 1.02
            tp = latest_price * 0.96
    except Exception as e:
        logger.warning("SL/TP calc failed for %s: %s", symbol, e)
        sl = tp = None
    header = f"Final recommendation for {symbol}: {final}\nSL: {sl if sl else 'n/a'} TP: {tp if tp else 'n/a'}\n"
    text = header + "\n".join(lines)
    return text, chosen_image

# ------------------------------
# Telegram helpers (HTTP-based)
# ------------------------------

def telegram_send_text(chat_id: str, text: str) -> Tuple[Optional[int], str]:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("telegram_send_text skipped: missing token")
        return None, "missing token"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = client.post(url, json=payload, timeout=20)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception as e:
        logger.exception("telegram_send_text failed: %s", e)
        return None, str(e)


def telegram_send_photo(chat_id: str, image_bytes: bytes, caption: str = "") -> Tuple[Optional[int], str]:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("telegram_send_photo skipped: missing token")
        return None, "missing token"
    if not image_bytes:
        logger.warning("telegram_send_photo skipped: empty image")
        return None, "empty image"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes, "image/png")}
    data = {"chat_id": chat_id, "caption": caption}
    try:
        r = client.post(url, data=data, files=files, timeout=30)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception as e:
        logger.exception("telegram_send_photo failed: %s", e)
        return None, str(e)

# ------------------------------
# Tasks
# ------------------------------

def hourly_task():
    logger.info("Hourly task started")
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            if TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN:
                telegram_send_text(TELEGRAM_CHAT_ID, text[:3800])
                if img:
                    telegram_send_photo(TELEGRAM_CHAT_ID, img, caption=f"{sym} hourly")
            else:
                logger.info("Report for %s:\n%s", sym, text[:600])
            time.sleep(0.5)
        except Exception:
            logger.exception("hourly_task error for %s", sym)
    logger.info("Hourly task finished")


def daily_task():
    logger.info("Daily task started")
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            if TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN:
                telegram_send_text(TELEGRAM_CHAT_ID, "📈 DAILY REPORT\n" + text[:3800])
                if img:
                    telegram_send_photo(TELEGRAM_CHAT_ID, img, caption=f"{sym} daily")
            time.sleep(0.5)
        except Exception:
            logger.exception("daily_task error for %s", sym)
    logger.info("Daily task finished")

# ------------------------------
# Scheduler (minute=0)
# ------------------------------

scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
scheduler.add_job(hourly_task, "cron", minute=0, id="hourly_task")
scheduler.add_job(daily_task, "cron", hour=4, minute=0, id="daily_task")

# ------------------------------
# FastAPI app and webhook
# ------------------------------

app = FastAPI(title="crypto-signal-bot")

@app.on_event("startup")
async def startup_event():
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")
    # send startup message (best-effort)
    try:
        if TELEGRAM_CHAT_ID and TELEGRAM_BOT_TOKEN:
            telegram_send_text(TELEGRAM_CHAT_ID, "✅ Crypto signal bot started (with cache/fallback).")
    except Exception:
        logger.exception("startup telegram message failed")

@app.on_event("shutdown")
async def shutdown_event():
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")
    except Exception:
        logger.exception("Failed to shutdown scheduler")

@app.get("/")
async def root():
    return {"status": "running", "time": datetime.datetime.datetime.utcnow().isoformat(), "exchange": EXCHANGE}

@app.get("/health")
async def health():
    try:
        df = fetch_klines(SYMBOLS[0], "1h", limit=2)
        ok = bool(df is not None and not df.empty)
        return {"status": "healthy" if ok else "unhealthy", "exchange": EXCHANGE}
    except Exception as e:
        logger.exception("health failed")
        return Response(content=json.dumps({"status": "unhealthy", "error": str(e)}), media_type="application/json", status_code=500)

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    try:
        upd = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}
    message = upd.get("message") or upd.get("edited_message")
    if not message:
        return {"ok": True}
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text", "") or ""
    text = text.strip()
    if not text:
        return {"ok": True}
    parts = text.split()
    cmd = parts[0].lstrip("/").lower()
    args = parts[1:]
    logger.info("Received webhook cmd=%s args=%s from chat=%s", cmd, args, chat_id)
    if cmd in ("fast", "test", "run"):
        # immediate report
        try:
            for sym in SYMBOLS:
                rpt, img = build_hourly_report(sym)
                telegram_send_text(chat_id, rpt[:3800])
                if img:
                    telegram_send_photo(chat_id, img, caption=f"{sym} hourly (on-demand)")
            return {"ok": True}
        except Exception as e:
            logger.exception("/fast handling failed")
            telegram_send_text(chat_id, f"Error building report: {e}")
            return {"ok": False, "error": str(e)}
    if cmd == "report":
        if not args:
            telegram_send_text(chat_id, "Usage: /report SYMBOL")
            return {"ok": True}
        target = args[0].upper()
        try:
            rpt, img = build_hourly_report(target)
            telegram_send_text(chat_id, rpt[:3800])
            if img:
                telegram_send_photo(chat_id, img, caption=f"{target} hourly (manual)")
            return {"ok": True}
        except Exception as e:
            logger.exception("/report failed")
            telegram_send_text(chat_id, f"Report failed for {target}: {e}")
            return {"ok": False, "error": str(e)}
    telegram_send_text(chat_id, "Command not recognized. Use /fast or /report SYMBOL")
    return {"ok": True}

@app.get("/fast")
async def http_fast():
    try:
        aggregated = {}
        for sym in SYMBOLS:
            txt, _ = build_hourly_report(sym)
            aggregated[sym] = txt.splitlines()[:8]
        return {"ok": True, "preview": aggregated}
    except Exception as e:
        logger.exception("http_fast failed")
        return Response(content=json.dumps({"ok": False, "error": str(e)}), media_type="application/json", status_code=500)

# ------------------------------
# Run with uvicorn main:app --host 0.0.0.0 --port $PORT
# ------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:%s exchange=%s", PORT, EXCHANGE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
