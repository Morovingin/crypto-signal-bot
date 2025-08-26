# main.py
"""
Async crypto-signal bot (main.py)
- Async httpx client
- AsyncIOScheduler (jobs run at minute=0)
- /telegram_webhook endpoint (accepts commands: /fast, /temp, /report)
- /fast HTTP endpoint for quick preview
- root accepts GET and HEAD (fixes UptimeRobot 405)
- Uses compute_indicators()/plot/... from TA libs
"""

import os
import io
import json
import logging
import datetime
import asyncio
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
# Logging (DEBUG for troubleshooting)
# ------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger("crypto-signal-bot")

# ------------------------------
# Config
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", "10000"))
EXCHANGE = os.getenv("EXCHANGE", "BYBIT").upper()
SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")
SYMBOLS = os.getenv("SYMBOLS", "DOGEUSDT,ADAUSDT").split(",")
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200}
}

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set - telegram will be disabled for sending reports")

# ------------------------------
# Async HTTP client (shared)
# ------------------------------
HEADERS = {"User-Agent": "crypto-signal-bot/1.0 (+https://example.com)"}
client = httpx.AsyncClient(timeout=30.0, headers=HEADERS)

# ------------------------------
# Endpoints (sync strings)
# ------------------------------
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES_URL = "https://api.bybit.com/v5/market/kline"

# ------------------------------
# Helpers: fetch klines (async)
# ------------------------------
async def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    logger.debug("fetch_binance_klines: %s %s", symbol, interval)
    r = await client.get(BINANCE_KLINES_URL, params=params)
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
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    return df[["open","high","low","close","volume"]]

async def fetch_bybit_klines_v5(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    mapping = {"15m":"15", "1h":"60", "4h":"240", "12h":"720"}
    if interval not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + interval)
    params = {"category": "linear", "symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
    logger.debug("fetch_bybit_klines_v5: %s %s", symbol, interval)
    r = await client.get(BYBIT_KLINES_URL, params=params)
    r.raise_for_status()
    jd = r.json()
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
    # bybit: 'start' is seconds
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
    """Try preferred exchange first, fallback to the other"""
    last_exc = None
    order = ["BYBIT", "BINANCE"] if EXCHANGE == "BYBIT" else ["BINANCE", "BYBIT"]
    for exch in order:
        try:
            if exch == "BYBIT":
                df = await fetch_bybit_klines_v5(symbol, interval, limit=limit)
            else:
                df = await fetch_binance_klines(symbol, interval, limit=limit)
            if df is None or df.empty:
                raise RuntimeError(f"Empty data from {exch}")
            return df
        except httpx.HTTPStatusError as he:
            logger.warning("fetch_klines %s failed on %s: %s", symbol, exch, he)
            last_exc = he
            continue
        except Exception as e:
            logger.exception("fetch_klines error from %s for %s %s", exch, symbol, interval)
            last_exc = e
            continue
    # both failed
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
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios":[3,1,1]})
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
        fig.savefig(buf, format="png", dpi=130)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        logger.exception("plot failed")
        return b""

# ------------------------------
# Build hourly report (async)
# ------------------------------
async def build_hourly_report(symbol: str) -> Tuple[str, bytes]:
    now = datetime.datetime.utcnow().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
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
            logger.exception("build_hourly_report: error for %s %s", symbol, tf_label)
            lines.append(f"TF: {tf_label} — error computing data: {e}")
    # aggregate final
    counts = {"BUY":0,"SELL":0,"HOLD":0}
    for tf, d in votes_summary.items():
        counts[d["rec"]] = counts.get(d["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"
    # SL/TP heuristic
    sl = tp = None
    try:
        latest_df = await fetch_klines(symbol, "1h", limit=3)
        latest_price = float(latest_df["close"].iloc[-1])
        if final == "BUY":
            sl = latest_price * 0.98
            tp = latest_price * 1.04
        elif final == "SELL":
            sl = latest_price * 1.02
            tp = latest_price * 0.96
    except Exception:
        logger.warning("SL/TP calc failed for %s", symbol)
        sl = tp = None
    header = f"Final recommendation for {symbol}: {final}\nSL: {sl if sl else 'n/a'} TP: {tp if tp else 'n/a'}\n"
    text = header + "\n".join(lines)
    return text, chosen_image

# ------------------------------
# Telegram helpers (async HTTP)
# ------------------------------
async def telegram_send_text(chat_id: str, text: str) -> Tuple[Optional[int], str]:
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("telegram_send_text skipped: missing token")
        return None, "missing token"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        logger.debug("telegram_send_text OK chat=%s len=%s", chat_id, len(text))
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_text failed")
        return None, "error"

async def telegram_send_photo(chat_id: str, image_bytes: bytes, caption: str = "") -> Tuple[Optional[int], str]:
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
        r = await client.post(url, data=data, files=files)
        r.raise_for_status()
        logger.debug("telegram_send_photo OK chat=%s size=%s", chat_id, len(image_bytes))
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
                await telegram_send_text(TELEGRAM_CHAT_ID, text[:3800])
                if img:
                    await telegram_send_photo(TELEGRAM_CHAT_ID, img, caption=f"{sym} hourly")
            else:
                logger.info("Report for %s:\n%s", sym, text[:600])
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
                await telegram_send_text(TELEGRAM_CHAT_ID, "📈 DAILY REPORT\n" + text[:3800])
                if img:
                    await telegram_send_photo(TELEGRAM_CHAT_ID, img, caption=f"{sym} daily")
            await asyncio.sleep(0.5)
        except Exception:
            logger.exception("daily_task error for %s", sym)
    logger.info("Daily task finished")

# ------------------------------
# Scheduler (use AsyncIOScheduler, exact hour minute=0)
# ------------------------------
scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
# schedule coroutine functions directly; AsyncIOScheduler will handle them
scheduler.add_job(lambda: asyncio.create_task(hourly_task()), "cron", minute=0, id="hourly_task", coalesce=True)
scheduler.add_job(lambda: asyncio.create_task(daily_task()), "cron", hour=4, minute=0, id="daily_task", coalesce=True)

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Crypto Signal Bot")

# root supports GET and HEAD (fixes 405 from uptime monitors)
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "crypto-signal-bot running",
        "time": datetime.datetime.utcnow().isoformat(),
        "exchange": EXCHANGE
    }

@app.get("/health")
async def health():
    try:
        df = await fetch_klines(SYMBOLS[0], "1h", limit=2)
        ok = bool(df is not None and not df.empty)
        return {"status": "healthy" if ok else "unhealthy", "exchange": EXCHANGE}
    except Exception as e:
        logger.exception("health failed")
        return Response(content=json.dumps({"status": "unhealthy", "error": str(e)}), media_type="application/json", status_code=500)

@app.get("/status")
async def status():
    return {"status": "ok", "message": "Service is running"}

@app.get("/ping")
async def ping():
    return Response(content="pong", media_type="text/plain")

# ------------------------------
# Telegram webhook endpoint (single path, easier for webhook setup)
# ------------------------------
@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    """
    Incoming Telegram webhook. Accepts commands:
      /fast  - generate immediate reports for configured SYMBOLS
      /temp  - quick preview (alias)
      /report <SYMBOL> - single symbol report
    """
    try:
        upd = await request.json()
    except Exception:
        logger.exception("telegram_webhook: invalid json")
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
    try:
        if cmd in ("fast", "temp"):
            # immediate report for all configured symbols
            for sym in SYMBOLS:
                rpt, img = await build_hourly_report(sym)
                await telegram_send_text(chat_id, rpt[:3800])
                if img:
                    await telegram_send_photo(chat_id, img, caption=f"{sym} hourly (on-demand)")
            return {"ok": True}
        if cmd == "report":
            if not args:
                await telegram_send_text(chat_id, "Usage: /report SYMBOL")
                return {"ok": True}
            target = args[0].upper()
            rpt, img = await build_hourly_report(target)
            await telegram_send_text(chat_id, rpt[:3800])
            if img:
                await telegram_send_photo(chat_id, img, caption=f"{target} hourly (manual)")
            return {"ok": True}
        if cmd == "start":
            await telegram_send_text(chat_id, "Привет! Отправь /fast для немедленного отчёта или /report SYMBOL")
            return {"ok": True}
        # unknown
        await telegram_send_text(chat_id, "Команда не распознана. Используйте /fast или /report SYMBOL")
        return {"ok": True}
    except Exception as e:
        logger.exception("telegram_webhook handling failed")
        # report error back to user (short message)
        try:
            await telegram_send_text(chat_id, f"Internal error handling command: {e}")
        except Exception:
            logger.exception("failed to notify user about internal error")
        return {"ok": False, "error": str(e)}

# ------------------------------
# HTTP /fast endpoint (preview)
# ------------------------------
@app.get("/fast")
async def http_fast():
    try:
        aggregated = {}
        for sym in SYMBOLS:
            txt, _ = await build_hourly_report(sym)
            aggregated[sym] = txt.splitlines()[:8]
        return {"ok": True, "preview": aggregated}
    except Exception as e:
        logger.exception("http_fast failed")
        return Response(content=json.dumps({"ok": False, "error": str(e)}), media_type="application/json", status_code=500)

# ------------------------------
# Lifespan events: start/stop scheduler and client
# ------------------------------
@app.on_event("startup")
async def startup_event():
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")
    # send startup telegram message (best-effort)
    try:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            await telegram_send_text(TELEGRAM_CHAT_ID, "✅ Crypto signal bot started and running 24/7.")
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
    try:
        await client.aclose()
    except Exception:
        logger.exception("Failed to close httpx client")

# ------------------------------
# Run app with uvicorn main:app --host 0.0.0.0 --port $PORT
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:%s (exchange=%s)", PORT, EXCHANGE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
