# main.py
"""
Robust crypto signal bot:
- async httpx client
- Bybit v5 + Binance kline fetch with fallback
- local pickle cache to avoid "No data" reports
- indicators and plotting
- AsyncIOScheduler: hourly (minute=0), daily (04:00), weekly (Sun 06:00)
- Telegram webhook endpoint (/telegram_webhook) and HTTP /fast preview endpoint
"""
import os
import io
import asyncio
import time
import json
import logging
import datetime
import traceback
from typing import Dict, Tuple, Optional

import httpx
import pandas as pd
import numpy as np

# avoid matplotlib font cache permission issues
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# TA
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
EXCHANGE = os.getenv("EXCHANGE", "BYBIT").upper()  # BYBIT or BINANCE
SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")
SYMBOLS = os.getenv("SYMBOLS", "DOGEUSDT,ADAUSDT,XRPUSDT").split(",")
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200}
}

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set - telegram notifications will be skipped")

# endpoints
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES_V5 = "https://api.bybit.com/v5/market/kline"

# cache dir (pickle)
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/klines_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# httpx async client
HEADERS = {"User-Agent": "crypto-signal-bot/1.0"}
client = httpx.AsyncClient(timeout=20.0, headers=HEADERS)

# ------------------------------
# Cache helpers
# ------------------------------
def _cache_path(symbol: str, tf_label: str) -> str:
    safe = symbol.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}__{tf_label}.pkl")

def save_cache(df: pd.DataFrame, symbol: str, tf_label: str) -> None:
    try:
        path = _cache_path(symbol, tf_label)
        df.to_pickle(path)
        logger.info("Saved cache %s (%d rows)", path, len(df))
    except Exception:
        logger.exception("Failed to save cache for %s %s", symbol, tf_label)

def load_cache(symbol: str, tf_label: str) -> Optional[pd.DataFrame]:
    try:
        path = _cache_path(symbol, tf_label)
        if not os.path.exists(path):
            return None
        df = pd.read_pickle(path)
        logger.info("Loaded cache %s (%d rows)", path, len(df))
        return df
    except Exception:
        logger.exception("Failed to load cache for %s %s", symbol, tf_label)
        return None

# ------------------------------
# Fetching klines (async) - Bybit v5 + Binance
# ------------------------------
async def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = await client.get(BINANCE_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Empty kline data from Binance")
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open","high","low","close","volume"]]

async def fetch_bybit_v5_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    mapping = {"15m":"15", "1h":"60", "4h":"240", "12h":"720"}
    if interval not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + interval)
    params = {"category":"linear", "symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
    r = await client.get(BYBIT_KLINES_V5, params=params)
    r.raise_for_status()
    jd = r.json()
    data = []
    if isinstance(jd, dict):
        res = jd.get("result")
        # result could be dict with "list" or list directly
        if isinstance(res, dict) and "list" in res:
            data = res["list"]
        elif isinstance(res, list):
            data = res
    if not data:
        # sometimes older endpoints use "ret_code" naming - include whole response for debug
        raise RuntimeError("Empty bybit v5 data: " + json.dumps(jd)[:400])
    df = pd.DataFrame(data)
    # bybit fields: start or start_at (seconds)
    if "start" in df.columns:
        df["open_time"] = pd.to_datetime(df["start"], unit="s")
    elif "start_at" in df.columns:
        df["open_time"] = pd.to_datetime(df["start_at"], unit="s")
    elif "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
    else:
        df["open_time"] = pd.to_datetime(df.iloc[:,0], unit="s")
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

async def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Try preferred exchange then fallback, use cache if both fail."""
    last_exc = None
    order = ["BYBIT","BINANCE"] if EXCHANGE == "BYBIT" else ["BINANCE","BYBIT"]
    for exch in order:
        try:
            if exch == "BYBIT":
                df = await fetch_bybit_v5_klines(symbol, interval, limit=limit)
            else:
                df = await fetch_binance_klines(symbol, interval, limit=limit)
            if df is None or df.empty:
                raise RuntimeError(f"Empty data from {exch}")
            # save cache
            try:
                save_cache(df, symbol, interval)
            except Exception:
                logger.exception("save_cache failed")
            return df
        except httpx.HTTPStatusError as he:
            last_exc = he
            status = he.response.status_code if he.response is not None else None
            logger.warning("fetch_klines %s failed for %s %s: %s", exch, symbol, interval, str(he))
            continue
        except Exception as e:
            last_exc = e
            logger.warning("fetch_klines error from %s for %s %s : %s", exch, symbol, interval, str(e))
            continue

    # both failed -> try cache
    logger.error("All exchanges failed for %s %s. Trying cache. last_exc=%s", symbol, interval, last_exc)
    cached = load_cache(symbol, interval)
    if cached is not None and not cached.empty:
        logger.info("Using cached data for %s %s", symbol, interval)
        return cached
    raise RuntimeError(f"Bybit/Binance fetch failed for {symbol} {interval}. Last error: {last_exc}")

# ------------------------------
# Indicators / scoring / plotting
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
        return {"pivot": pivot, "r1": 2*pivot - low, "s1": 2*pivot - high}
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
        "100.0%": last_low
    }

def score_signals(ind_df: pd.DataFrame) -> Tuple[str, Dict[str,int]]:
    try:
        row = ind_df.iloc[-1]
        votes = {"buy":0,"sell":0,"neutral":0}
        if row["rsi"] < 30: votes["buy"] += 1
        elif row["rsi"] > 70: votes["sell"] += 1
        if row["macd_hist"] > 0: votes["buy"] += 1
        elif row["macd_hist"] < 0: votes["sell"] += 1
        if row["close"] > row["ema20"]: votes["buy"] += 1
        else: votes["sell"] += 1
        if row["bb_pct"] > 0.85: votes["sell"] += 1
        elif row["bb_pct"] < 0.15: votes["buy"] += 1
        if row["stoch_rsi_k"] < 20: votes["buy"] += 1
        elif row["stoch_rsi_k"] > 80: votes["sell"] += 1
        vol_mean = ind_df["volume"].tail(50).mean() if len(ind_df) >= 50 else ind_df["volume"].mean()
        if vol_mean > 0 and row["volume"] > vol_mean * 1.5:
            if row["macd_hist"] > 0: votes["buy"] += 1
            elif row["macd_hist"] < 0: votes["sell"] += 1
        score = votes["buy"] - votes["sell"]
        if score >= 2: return "BUY", votes
        elif score <= -2: return "SELL", votes
        else: return "HOLD", votes
    except Exception:
        logger.exception("score_signals failed")
        return "HOLD", {"buy":0,"sell":0,"neutral":0}

def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    try:
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(3,1, figsize=(11,10), sharex=True, gridspec_kw={"height_ratios":[3,1,1]})
        ax_price, ax_macd, ax_rsi = axes
        ax_price.plot(df.index, df["close"], label=f"{symbol} close")
        if "sma20" in df.columns: ax_price.plot(df.index, df["sma20"], label="SMA20", linewidth=0.8)
        if "ema20" in df.columns: ax_price.plot(df.index, df["ema20"], label="EMA20", linewidth=0.8)
        if "bb_hband" in df.columns and "bb_lband" in df.columns:
            ax_price.plot(df.index, df["bb_hband"], linestyle="--", linewidth=0.7, label="BB Upper")
            ax_price.plot(df.index, df["bb_lband"], linestyle="--", linewidth=0.7, label="BB Lower")
        ax_price.set_title(f"{symbol} {timeframe} — price & indicators")
        ax_price.legend(loc="upper left"); ax_price.grid(True)
        if "macd_line" in df.columns:
            ax_macd.plot(df.index, df["macd_line"], label="MACD")
            ax_macd.plot(df.index, df["macd_signal"], label="Signal")
            ax_macd.bar(df.index, df["macd_hist"], label="Hist", alpha=0.6)
            ax_macd.legend(loc="upper left"); ax_macd.grid(True)
        if "rsi" in df.columns:
            ax_rsi.plot(df.index, df["rsi"], label="RSI")
            ax_rsi.axhline(70, linestyle="--", linewidth=0.6)
            ax_rsi.axhline(30, linestyle="--", linewidth=0.6)
            ax_rsi.legend(loc="upper left"); ax_rsi.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        logger.exception("plot failed")
        return b""

# ------------------------------
# Build report (async)
# ------------------------------
async def build_hourly_report(symbol: str) -> Tuple[str, bytes]:
    now = datetime.datetime.datetime.utcnow().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"⏱️ Hourly report for {symbol} — {now}\n"]
    votes_summary = {}
    chosen_image = b""
    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = await fetch_klines(symbol, tf_conf["interval"], limit=tf_conf["limit"])
            if df is None or df.empty or len(df) < 10:
                lines.append(f"TF: {tf_label} — not enough data")
                continue
            ind = compute_indicators(df)
            rec, votes = score_signals(ind)
            last = df.tail(3)
            lastc = {"high": last["high"].iloc[-1], "low": last["low"].iloc[-1], "close": last["close"].iloc[-1]}
            piv = pivot_support_resistance(lastc)
            window = df.tail(60)
            low = float(window["low"].min()); high = float(window["high"].max())
            fibs = fib_levels(low, high)
            lines.append(f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.8f} | Rec: {rec} | RSI:{ind['rsi'].iloc[-1]:.1f}")
            lines.append(f"  Pivot:{piv['pivot']:.8f}, R1:{piv['r1']:.8f}, S1:{piv['s1']:.8f}")
            lines.append(f"  Fib 23.6%:{fibs['23.6%']:.8f} 38.2%:{fibs['38.2%']:.8f} 50%:{fibs['50.0%']:.8f}")
            votes_summary[tf_label] = {"rec": rec, "votes": votes}
            if tf_label == "4h" and not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
            elif not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("build_hourly_report error for %s %s", symbol, tf_label)
            lines.append(f"TF: {tf_label} — error computing data: {str(e)}")
            lines.append(f"Debug: {tb.splitlines()[-1]}")
    # aggregate
    counts = {"BUY":0,"SELL":0,"HOLD":0}
    for tf, d in votes_summary.items():
        counts[d["rec"]] = counts.get(d["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"
    # SL/TP
    sl = tp = None
    try:
        latest_df = await fetch_klines(symbol, "1h", limit=3)
        latest_price = float(latest_df["close"].iloc[-1])
        if final == "BUY":
            sl = latest_price * 0.98; tp = latest_price * 1.04
        elif final == "SELL":
            sl = latest_price * 1.02; tp = latest_price * 0.96
    except Exception as e:
        logger.warning("SL/TP calc failed for %s: %s", symbol, e)
        sl = tp = None
    header = f"Final recommendation for {symbol}: {final}\nSL: {('n/a' if sl is None else f'{sl:.8f}')} TP: {('n/a' if tp is None else f'{tp:.8f}')}\n"
    text = header + "\n".join(lines)
    return text, chosen_image

# ------------------------------
# Telegram helpers (async HTTP)
# ------------------------------
async def telegram_send_text(text: str, chat_id: Optional[str] = None) -> Tuple[Optional[int], str]:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("telegram_send_text skipped: missing token")
        return None, "missing token"
    if chat_id is None: chat_id = TELEGRAM_CHAT_ID
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = await client.post(url, json=payload, timeout=20.0)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_text failed")
        return None, "error"

async def telegram_send_photo(image_bytes: bytes, caption: str = "", chat_id: Optional[str] = None) -> Tuple[Optional[int], str]:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("telegram_send_photo skipped: missing token")
        return None, "missing token"
    if chat_id is None: chat_id = TELEGRAM_CHAT_ID
    if not image_bytes:
        logger.warning("telegram_send_photo skipped: empty image")
        return None, "empty image"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes, "image/png")}
    data = {"chat_id": str(chat_id), "caption": caption}
    try:
        r = await client.post(url, data=data, files=files, timeout=30.0)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_photo failed")
        return None, "error"

# ------------------------------
# Tasks (async)
# ------------------------------
async def hourly_task():
    logger.info("Hourly task started")
    for sym in SYMBOLS:
        try:
            text, img = await build_hourly_report(sym)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                await telegram_send_text(text[:3800])
                if img:
                    await telegram_send_photo(img, caption=f"{sym} hourly chart")
            else:
                logger.info("Report for %s:\n%s", sym, text[:800])
            await asyncio.sleep(0.5)
        except Exception:
            logger.exception("hourly_task error for %s", sym)
    logger.info("Hourly task finished")

async def daily_task():
    logger.info("Daily task started")
    for sym in SYMBOLS:
        try:
            text, img = await build_hourly_report(sym)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                await telegram_send_text("📈 DAILY REPORT\n" + text[:3800])
                if img:
                    await telegram_send_photo(img, caption=f"{sym} daily chart")
            await asyncio.sleep(0.5)
        except Exception:
            logger.exception("daily_task error for %s", sym)
    logger.info("Daily task finished")

async def weekly_task():
    logger.info("Weekly task started")
    for sym in SYMBOLS:
        try:
            # weekly report uses 1w timeframe if user wants; for now use aggregated daily/weekly logic
            text, img = await build_hourly_report(sym)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                await telegram_send_text("🗓 WEEKLY REPORT\n" + text[:3800])
                if img:
                    await telegram_send_photo(img, caption=f"{sym} weekly chart")
            await asyncio.sleep(0.5)
        except Exception:
            logger.exception("weekly_task error for %s", sym)
    logger.info("Weekly task finished")

# ------------------------------
# Scheduler (AsyncIOScheduler)
# ------------------------------
scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
# hourly at minute=0
scheduler.add_job(hourly_task, "cron", minute=0, id="hourly_task")
# daily at 04:00
scheduler.add_job(daily_task, "cron", hour=4, minute=0, id="daily_task")
# weekly: Sunday at 06:00
scheduler.add_job(weekly_task, "cron", day_of_week="sun", hour=6, minute=0, id="weekly_task")

# ------------------------------
# FastAPI app & webhook
# ------------------------------
app = FastAPI(title="crypto-signal-bot")

@app.on_event("startup")
async def on_startup():
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")
    # startup message best effort
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            await telegram_send_text("✅ Crypto signal bot started (hourly reports at :00).")
    except Exception:
        logger.exception("startup message failed")

@app.on_event("shutdown")
async def on_shutdown():
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")
    except Exception:
        logger.exception("scheduler shutdown failed")
    try:
        await client.aclose()
    except Exception:
        logger.exception("client close failed")

@app.get("/")
async def root():
    return {"status":"running","time": datetime.datetime.utcnow().isoformat(), "exchange": EXCHANGE}

@app.get("/fast")
async def http_fast_preview():
    """Return short preview for each symbol — useful for testing / uptime monitors."""
    try:
        out = {}
        for sym in SYMBOLS:
            txt, _ = await build_hourly_report(sym)
            out[sym] = txt.splitlines()[:8]
        return {"ok": True, "preview": out}
    except Exception as e:
        logger.exception("http_fast failed")
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)

@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    """Webhook handler for Telegram updates.
    Supported commands (from user chat): /fast, /temp, /report SYMBOL
    """
    try:
        upd = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}
    message = upd.get("message") or upd.get("edited_message")
    if not message:
        return {"ok": True}
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()
    if not text:
        return {"ok": True}
    parts = text.split()
    cmd = parts[0].lstrip("/").split("@")[0].lower()
    args = parts[1:]
    logger.info("Received webhook cmd=%s args=%s chat=%s", cmd, args, chat_id)
    try:
        if cmd in ("fast", "test", "run"):
            # send immediate reports for all symbols
            for sym in SYMBOLS:
                rpt, img = await build_hourly_report(sym)
                await telegram_send_text(rpt[:3800], chat_id=chat_id)
                if img:
                    await telegram_send_photo(img, caption=f"{sym} hourly (on-demand)", chat_id=chat_id)
            return {"ok": True}
        if cmd in ("temp",):
            await telegram_send_text("✅ temp OK — bot responds", chat_id=chat_id)
            return {"ok": True}
        if cmd == "report":
            if not args:
                await telegram_send_text("Usage: /report SYMBOL (e.g. /report DOGEUSDT)", chat_id=chat_id)
                return {"ok": True}
            target = args[0].upper()
            rpt, img = await build_hourly_report(target)
            await telegram_send_text(rpt[:3800], chat_id=chat_id)
            if img:
                await telegram_send_photo(img, caption=f"{target} hourly (manual)", chat_id=chat_id)
            return {"ok": True}
        # optional: set webhook via command (admin only) - not implemented for security
        await telegram_send_text("Command not recognized. Use /fast, /temp, or /report SYMBOL", chat_id=chat_id)
    except Exception:
        logger.exception("telegram_webhook handling error")
        try:
            await telegram_send_text("Internal error handling command.", chat_id=chat_id)
        except Exception:
            pass
    return {"ok": True}

# ------------------------------
# Run if invoked directly
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:%s (exchange=%s)", PORT, EXCHANGE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
