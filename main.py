# main.py
import os
import io
import json
import logging
import datetime
import asyncio
from typing import Dict, Tuple

import httpx
import numpy as np
import pandas as pd
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crypto-signal-bot")

# ------------------------------
# Config / environment
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", "10000"))
EXCHANGE = os.getenv("EXCHANGE", "BYBIT").upper()
SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("Telegram token/chat_id not set - notifications will fail")

BYBIT_KLINES_ENDPOINT = "https://api.bybit.com/public/linear/kline"
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"

SYMBOLS = ["DOGEUSDT", "ADAUSDT"]
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200},
}

# ------------------------------
# HTTP client
# ------------------------------
client = httpx.AsyncClient(timeout=30.0)

# ------------------------------
# Kline fetchers
# ------------------------------
async def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = await client.get(BINANCE_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()
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

async def fetch_bybit_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    mapping = {"15m":"15", "1h":"60", "4h":"240", "12h":"720"}
    if interval not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + interval)
    params = {"symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
    r = await client.get(BYBIT_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    jd = r.json()
    data = []
    if isinstance(jd, dict):
        if "result" in jd:
            res = jd["result"]
            if isinstance(res, list):
                data = res
            elif isinstance(res, dict) and "list" in res:
                data = res["list"]
    if not data:
        raise RuntimeError("Empty kline data from Bybit: " + json.dumps(jd)[:500])
    df = pd.DataFrame(data)
    if "start_at" in df.columns:
        df["open_time"] = pd.to_datetime(df["start_at"], unit="s")
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

async def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    try:
        if EXCHANGE == "BYBIT":
            return await fetch_bybit_klines(symbol, interval, limit)
        else:
            return await fetch_binance_klines(symbol, interval, limit)
    except Exception:
        logger.exception("fetch_klines failed for %s %s", symbol, interval)
        raise

# ------------------------------
# Indicators
# ------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    out["rsi"] = RSIIndicator(close, window=14, fillna=True).rsi()
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9, fillna=True)
    out["macd_line"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    bb = BollingerBands(close, window=20, window_dev=2, fillna=True)
    out["bb_mavg"] = bb.bollinger_mavg()
    out["bb_hband"] = bb.bollinger_hband()
    out["bb_lband"] = bb.bollinger_lband()
    out["bb_pct"] = (close - out["bb_lband"]) / (out["bb_hband"] - out["bb_lband"] + 1e-12)
    out["sma20"] = SMAIndicator(close, 20, fillna=True).sma_indicator()
    out["sma50"] = SMAIndicator(close, 50, fillna=True).sma_indicator()
    out["ema20"] = EMAIndicator(close, 20, fillna=True).ema_indicator()
    out["ema50"] = EMAIndicator(close, 50, fillna=True).ema_indicator()
    stoch = StochRSIIndicator(close, 14, 3, 3, fillna=True)
    out["stoch_rsi_k"] = stoch.stochrsi_k()
    out["stoch_rsi_d"] = stoch.stochrsi_d()
    return out

# ------------------------------
# Support/Fib
# ------------------------------
def pivot_support_resistance(series: pd.Series) -> Dict[str, float]:
    try:
        high, low, close = series["high"], series["low"], series["close"]
        pivot = (high + low + close)/3
        return {"pivot": pivot, "r1": 2*pivot - low, "s1": 2*pivot - high}
    except Exception:
        return {"pivot":0.0,"r1":0.0,"s1":0.0}

def fib_levels(low: float, high: float) -> Dict[str,float]:
    diff = high - low
    return {
        "0.0%": high,
        "23.6%": high - 0.236*diff,
        "38.2%": high - 0.382*diff,
        "50.0%": high - 0.5*diff,
        "61.8%": high - 0.618*diff,
        "100.0%": low
    }

# ------------------------------
# Signal scoring
# ------------------------------
def score_signals(ind_df: pd.DataFrame) -> Tuple[str, Dict[str,int]]:
    try:
        row = ind_df.iloc[-1]
        votes = {"buy":0,"sell":0,"neutral":0}
        if row["rsi"] < 30: votes["buy"]+=1
        elif row["rsi"]>70: votes["sell"]+=1
        if row["macd_hist"]>0: votes["buy"]+=1
        elif row["macd_hist"]<0: votes["sell"]+=1
        if row["close"]>row["ema20"]: votes["buy"]+=1
        else: votes["sell"]+=1
        if row["bb_pct"]>0.85: votes["sell"]+=1
        elif row["bb_pct"]<0.15: votes["buy"]+=1
        if row["stoch_rsi_k"]<20: votes["buy"]+=1
        elif row["stoch_rsi_k"]>80: votes["sell"]+=1
        vol_mean = ind_df["volume"].tail(50).mean() if len(ind_df)>=50 else ind_df["volume"].mean()
        if vol_mean>0 and row["volume"]>vol_mean*1.5:
            if row["macd_hist"]>0: votes["buy"]+=1
            elif row["macd_hist"]<0: votes["sell"]+=1
        score = votes["buy"] - votes["sell"]
        if score>=2: return "BUY", votes
        elif score<=-2: return "SELL", votes
        else: return "HOLD", votes
    except Exception:
        return "HOLD", {"buy":0,"sell":0,"neutral":0}

# ------------------------------
# Charting
# ------------------------------
def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    try:
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(3,1,figsize=(11,10),sharex=True,gridspec_kw={"height_ratios":[3,1,1]})
        ax_price, ax_macd, ax_rsi = axes
        ax_price.plot(df.index, df["close"], label="Close")
        if "sma20" in df: ax_price.plot(df.index, df["sma20"], label="SMA20", lw=0.8)
        if "ema20" in df: ax_price.plot(df.index, df["ema20"], label="EMA20", lw=0.8)
        if "bb_hband" in df: ax_price.plot(df.index, df["bb_hband"], "--", lw=0.7,label="BB Upper")
        if "bb_lband" in df: ax_price.plot(df.index, df["bb_lband"], "--", lw=0.7,label="BB Lower")
        ax_price.legend(); ax_price.grid(True)
        if "macd_line" in df:
            ax_macd.plot(df.index, df["macd_line"], label="MACD")
            ax_macd.plot(df.index, df["macd_signal"], label="Signal")
            ax_macd.bar(df.index, df["macd_hist"], label="Hist", alpha=0.6)
            ax_macd.legend(); ax_macd.grid(True)
        if "rsi" in df:
            ax_rsi.plot(df.index, df["rsi"], label="RSI")
            ax_rsi.axhline(70,"--",0.6); ax_rsi.axhline(30,"--",0.6)
            ax_rsi.legend(); ax_rsi.grid(True)
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
# Report
# ------------------------------
async def build_hourly_report(symbol: str) -> Tuple[str, bytes]:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines, votes_summary = [f"⏱ Hourly report for {symbol} — {now}\n"], {}
    chosen_image = b""
    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = await fetch_klines(symbol, tf_conf["interval"], tf_conf["limit"])
            if df.empty or len(df)<10: 
                lines.append(f"TF:{tf_label} — not enough data"); continue
            ind = compute_indicators(df)
            rec, votes = score_signals(ind)
            last = df.tail(3)
            piv = pivot_support_resistance({"high":last["high"].iloc[-1],"low":last["low"].iloc[-1],"close":last["close"].iloc[-1]})
            window = df.tail(60); low = float(window["low"].min()); high = float(window["high"].max())
            fibs = fib_levels(low, high)
            lines.append(f"TF:{tf_label} | Price:{df['close'].iloc[-1]:.6f} | Rec:{rec} | RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}")
            lines.append(f"  Pivot:{piv['pivot']:.6f}, R1:{piv['r1']:.6f}, S1:{piv['s1']:.6f}")
            lines.append(f"  Fib 23.6%:{fibs['23.6%']:.6f} 38.2%:{fibs['38.2%']:.6f} 50%:{fibs['50.0%']:.6f}")
            votes_summary[tf_label] = {"rec": rec,"votes":votes}
            if tf_label=="4h" and not chosen_image: chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
            elif not chosen_image: chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
        except Exception:
            logger.exception("report error %s %s", symbol, tf_label)
            lines.append(f"TF:{tf_label} — error")
    counts={"BUY":0,"SELL":0,"HOLD":0}
    for d in votes_summary.values(): counts[d["rec"]] = counts.get(d["rec"],0)+1
    final = max(counts.items(), key=lambda x:x[1])[0] if counts else "HOLD"
    try:
        latest_price = float((await fetch_klines(symbol,"1h",3))["close"].iloc[-1])
        sl = latest_price*(0.98 if final=="BUY" else 1.02)
        tp = latest_price*(1.04 if final=="BUY" else 0.96)
    except Exception: sl=tp=0.0
    header=f"Final recommendation for {symbol}: {final}\nSL:{sl:.6f} TP:{tp:.6f}\n"
    return header + "\n".join(lines), chosen_image

# ------------------------------
# Telegram
# ------------------------------
async def telegram_send_text(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return None, "missing"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_text failed")
        return None, "error"

async def telegram_send_photo(image_bytes: bytes, caption: str = ""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not image_bytes: return None, "skipped"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        r = await client.post(url, data=data, files=files)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_photo failed")
        return None, "error"

# ------------------------------
# Tasks
# ------------------------------
async def hourly_task():
    logger.info("Hourly task started")
    for sym in SYMBOLS:
        try:
            text, img = await build_hourly_report(sym)
            await telegram_send_text(text[:3800])
            if img: await telegram_send_photo(img, caption=f"{sym} hourly chart")
            await asyncio.sleep(0.5)
        except Exception:
            logger.exception("hourly_task error %s", sym)
    logger.info("Hourly task finished")

async def daily_task():
    logger.info("Daily task started")
    for sym in SYMBOLS:
        try:
            text, img = await build_hourly_report(sym)
            await telegram_send_text("📈 DAILY REPORT\n"+text[:3800])
            if img: await telegram_send_photo(img, caption=f"{sym} daily chart")
            await asyncio.sleep(0.5)
        except Exception:
            logger.exception("daily_task error %s", sym)
    logger.info("Daily task finished")

# ------------------------------
# Scheduler
# ------------------------------
scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
scheduler.add_job(hourly_task, "cron", minute=1, id="hourly_task")
scheduler.add_job(daily_task, "cron", hour=4, minute=2, id="daily_task")

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Crypto Signal Bot")

@app.api_route("/", methods=["GET","HEAD"])
async def root():
    return {"status":"crypto-signal-bot running","time":datetime.datetime.utcnow().isoformat(),"exchange":EXCHANGE}

@app.get("/health")
async def health_check():
    try:
        df = await fetch_klines("BTCUSDT","1m",1)
        return {"status":"healthy","exchange":EXCHANGE,"timestamp":datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        logger.exception("health_check failure")
        return Response(content=json.dumps({"status":"unhealthy","error":str(e),"timestamp":datetime.datetime.utcnow().isoformat()}),status_code=500,media_type="application/json")

@app.get("/ping")
async def ping():
    return Response(content="pong", media_type="text/plain")

@app.get("/fast")
async def fast_report():
    reports = []
    for sym in SYMBOLS:
        text, _ = await build_hourly_report(sym)
        reports.append(text[:3800])
    return {"reports": reports}

# ------------------------------
# Startup / shutdown
# ------------------------------
@app.on_event("startup")
async def on_startup():
    try:
        scheduler.start()
        logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")
    try:
        await telegram_send_text("✅ Crypto signal bot deployed and running 24/7.")
    except Exception:
        logger.exception("Startup telegram message failed")

@app.on_event("shutdown")
async def on_shutdown():
    try:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shutdown")
    except Exception:
        logger.exception("Failed to shutdown scheduler")

# ------------------------------
# Run server directly
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:%s (exchange=%s)", PORT, EXCHANGE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
