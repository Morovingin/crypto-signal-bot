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
    mapping = {"15m":"15", "1h":"60", "4h":"240", "12h":"720"}
    if interval not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + interval)
    params = {"category": "linear", "symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
    r = client.get(BYBIT_KLINES_V5, params=params)
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
    if "start" in df.columns:
        df["open_time"] = pd.to_datetime(df["start"], unit="s")
    elif "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
    else:
        df["open_time"] = pd.to_datetime(df.iloc[:, 0], unit="s")
    df.set_index("open_time", inplace=True)
    for col in ["open","high","low","close","volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        elif col.capitalize() in df.columns:
            df[col] = df[col.capitalize()].astype(float)
        else:
            df[col] = 0.0
    return df[["open","high","low","close","volume"]]


def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    last_exc = None
    tried = []
    order = ["BINANCE", "BYBIT"] if EXCHANGE=="BINANCE" else ["BYBIT", "BINANCE"]
    for exch in order:
        try:
            df = fetch_binance_klines(symbol, interval, limit) if exch=="BINANCE" else fetch_bybit_klines_v5(symbol, interval, limit)
            if df is None or df.empty:
                raise RuntimeError(f"Empty data from {exch}")
            try: save_cache(df, symbol, interval)
            except Exception: logger.exception("save_cache failed")
            return df
        except httpx.HTTPStatusError as he:
            last_exc = he
            status = he.response.status_code if he.response else None
            logger.warning("fetch_klines %s failed for %s %s: %s", exch, symbol, interval, str(he))
            tried.append((exch, status))
            continue
        except Exception as e:
            last_exc = e
            logger.exception("fetch_klines error from %s for %s %s", exch, symbol, interval)
            tried.append((exch, str(e)))
            continue
    logger.error("All exchanges failed for %s %s. attempts=%s. Will try cache if available", symbol, interval, tried)
    cached = load_cache(symbol, interval)
    if cached is not None and not cached.empty:
        logger.info("Using cached data for %s %s", symbol, interval)
        return cached
    raise RuntimeError(f"Bybit/Binance fetch failed for {symbol} {interval}. Last error: {last_exc}")

# ------------------------------
# Indicators, scoring, plotting
# ------------------------------

# ... (compute_indicators, pivot_support_resistance, fib_levels, score_signals, plot_price_and_indicators)
# оставлены без изменений

# ------------------------------
# Build report
# ------------------------------

# ... (build_hourly_report) оставлена без изменений

# ------------------------------
# Telegram helpers
# ------------------------------

# ... (telegram_send_text, telegram_send_photo) оставлены без изменений

# ------------------------------
# Tasks
# ------------------------------

# ... (hourly_task, daily_task) оставлены без изменений

# ------------------------------
# Scheduler
# ------------------------------

scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
scheduler.add_job(hourly_task, "cron", minute=0, id="hourly_task")
scheduler.add_job(daily_task, "cron", hour=4, minute=0, id="daily_task")

# ------------------------------
# FastAPI app
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
    # send startup message
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
    return {"status": "running", "time": datetime.datetime.utcnow().isoformat(), "exchange": EXCHANGE}

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
    chat_id = message.get("chat", {}).get("id")
    text = (message.get("text") or "").strip()
    if not text:
        return {"ok": True}
    cmd = text.split()[0].lstrip("/").lower()
    args = text.split()[1:]
    logger.info("Received webhook cmd=%s args=%s from chat=%s", cmd, args, chat_id)
    if cmd in ("fast", "test", "run"):
        try:
            for sym in SYMBOLS:
                rpt, img = build_hourly_report(sym)
                telegram_send_text(chat_id, rpt[:3800])
                if img: telegram_send_photo(chat_id, img, caption=f"{sym} hourly (on-demand)")
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
            if img: telegram_send_photo(chat_id, img, caption=f"{target} hourly (manual)")
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
