# main.py
import os
import io
import time
import json
import logging
import datetime
from typing import Dict, Tuple, Optional

import httpx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response
from apscheduler.schedulers.background import BackgroundScheduler

# Technical indicators from `ta` package
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crypto-signal-bot")

# ------------------------------
# Configuration (ENV)
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 10000))

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables")

# Exchanges endpoints
BYBIT_V5_KLINE = "https://api.bybit.com/v5/market/kline"
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"

# Symbols and timeframes
SYMBOLS = ["DOGEUSDT", "ADAUSDT"]
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200}
}

# Scheduler timezone (tz name)
SCHED_TZ = "Europe/Moscow"

# HTTP client with simple User-Agent to reduce chance of 403
client = httpx.Client(timeout=20.0, headers={"User-Agent": "crypto-signal-bot/1.0"})

# ------------------------------
# Fetching OHLCV: Bybit v5 (primary) + Binance (fallback)
# ------------------------------
def fetch_bybit_klines(symbol: str, tf_label: str, limit: int = 200, category: str = "linear") -> pd.DataFrame:
    """
    Fetch OHLCV from Bybit v5 public endpoint.
    tf_label in {"15m","1h","4h","12h"} mapped to v5 interval values.
    Returns DataFrame with columns ['open','high','low','close','volume'] and datetime index.
    """
    mapping = {"15m": "15", "1h": "60", "4h": "240", "12h": "720"}
    if tf_label not in mapping:
        raise ValueError("Unsupported timeframe for Bybit: " + tf_label)
    params = {"category": category, "symbol": symbol.upper(), "interval": mapping[tf_label], "limit": limit}
    try:
        r = client.get(BYBIT_V5_KLINE, params=params)
        r.raise_for_status()
        jd = r.json()
        # Bybit v5 standard: {"retCode":0,"retMsg":"OK","result":{"list":[ [ts_ms, open, high, low, close, volume, ...], ... ] }}
        if jd.get("retCode", 0) != 0:
            raise RuntimeError(f"Bybit error retCode={jd.get('retCode')} msg={jd.get('retMsg')}")
        result = jd.get("result", {})
        klines = result.get("list") or result.get("data") or []
        if not klines:
            raise RuntimeError("Empty kline list from Bybit")

        # If list of lists -> build DataFrame from first 6 fields
        first = klines[0]
        if isinstance(first, (list, tuple)):
            # ensure shape >=6
            arr = [item[:6] for item in klines]
            df = pd.DataFrame(arr, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            return df[["open", "high", "low", "close", "volume"]]
        elif isinstance(first, dict):
            df = pd.DataFrame(klines)
            # possible keys: 't' or 'start' or 'open_time' etc.
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            elif "start_at" in df.columns:
                df["ts"] = pd.to_datetime(df["start_at"], unit="s")
            elif "open_time" in df.columns:
                df["ts"] = pd.to_datetime(df["open_time"], unit="s")
            elif "timestamp" in df.columns:
                df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
            else:
                df["ts"] = pd.to_datetime(df.iloc[:, 0], unit="ms")
            df.set_index("ts", inplace=True)
            # normalize column names
            for target in ["open", "high", "low", "close", "volume"]:
                if target in df.columns:
                    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0.0)
                elif target.capitalize() in df.columns:
                    df[target] = pd.to_numeric(df[target.capitalize()], errors="coerce").fillna(0.0)
                else:
                    df[target] = 0.0
            return df[["open", "high", "low", "close", "volume"]]
        else:
            raise RuntimeError("Unknown kline item format from Bybit")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error fetching Bybit klines {symbol} {tf_label}: {e}")
        raise
    except Exception as e:
        logger.exception(f"Error fetching Bybit klines {symbol} {tf_label}: {e}")
        raise

def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch Binance klines fallback.
    interval examples: "15m","1h","4h","12h"
    """
    try:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        r = client.get(BINANCE_KLINES_ENDPOINT, params=params)
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
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        logger.exception(f"Error fetching Binance klines: {e}")
        raise

def fetch_klines(symbol: str, tf_label: str, limit: int = 200) -> pd.DataFrame:
    """
    Primary: Bybit v5. If Bybit fails, fallback to Binance.
    """
    try:
        df = fetch_bybit_klines(symbol, tf_label, limit=limit, category="linear")
        if df is None or df.empty:
            raise RuntimeError("Empty DataFrame from Bybit")
        return df
    except Exception as e:
        logger.warning(f"Bybit fetch failed for {symbol} {tf_label}, trying Binance: {e}")
        try:
            df = fetch_binance_klines(symbol, TIMEFRAMES[tf_label]["interval"], limit=limit)
            return df
        except Exception as e2:
            logger.exception(f"Binance fallback failed for {symbol} {tf_label}: {e2}")
            return pd.DataFrame()

# ------------------------------
# Indicators calculation
# ------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute indicators (uses `ta` where convenient).
    Returns df augmented with indicators.
    """
    if df.empty or len(df) < 3:
        raise RuntimeError("Not enough data to compute indicators")
    out = df.copy()
    close = out["close"]

    # RSI
    try:
        out["rsi"] = RSIIndicator(close=close, window=14, fillna=True).rsi()
    except Exception:
        out["rsi"] = compute_simple_rsi(close, 14)

    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=True)
    out["macd_line"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=close, window=20, window_dev=2, fillna=True)
    out["bb_mavg"] = bb.bollinger_mavg()
    out["bb_hband"] = bb.bollinger_hband()
    out["bb_lband"] = bb.bollinger_lband()
    out["bb_pct"] = (close - out["bb_lband"]) / (out["bb_hband"] - out["bb_lband"] + 1e-12)

    # SMA / EMA
    out["sma20"] = SMAIndicator(close=close, window=20, fillna=True).sma_indicator()
    out["sma50"] = SMAIndicator(close=close, window=50, fillna=True).sma_indicator()
    out["ema20"] = EMAIndicator(close=close, window=20, fillna=True).ema_indicator()
    out["ema50"] = EMAIndicator(close=close, window=50, fillna=True).ema_indicator()

    # StochRSI
    try:
        stoch_rsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3, fillna=True)
        out["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        out["stoch_rsi_d"] = stoch_rsi.stochrsi_d()
    except Exception:
        out["stoch_rsi_k"] = 50.0
        out["stoch_rsi_d"] = 50.0

    return out

def compute_simple_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / (down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# ------------------------------
# Support / Pivot / Fibonacci
# ------------------------------
def pivot_support_resistance(last_candle: Dict[str, float]) -> Dict[str, float]:
    try:
        high = float(last_candle["high"])
        low = float(last_candle["low"])
        close = float(last_candle["close"])
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        return {"pivot": pivot, "r1": r1, "s1": s1}
    except Exception as e:
        logger.exception("Pivot calc error: %s", e)
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

# ------------------------------
# Signal scoring (voting)
# ------------------------------
def score_signals(ind_df: pd.DataFrame) -> Tuple[str, Dict[str, int]]:
    try:
        row = ind_df.iloc[-1]
        votes = {"buy": 0, "sell": 0, "neutral": 0}

        # RSI
        if row.get("rsi", 50) < 30:
            votes["buy"] += 1
        elif row.get("rsi", 50) > 70:
            votes["sell"] += 1

        # MACD hist
        if row.get("macd_hist", 0) > 0:
            votes["buy"] += 1
        elif row.get("macd_hist", 0) < 0:
            votes["sell"] += 1

        # EMA20 trend
        if row.get("close", 0) > row.get("ema20", 0):
            votes["buy"] += 1
        else:
            votes["sell"] += 1

        # Bollinger proximity
        if row.get("bb_pct", 0.5) > 0.85:
            votes["sell"] += 1
        elif row.get("bb_pct", 0.5) < 0.15:
            votes["buy"] += 1

        # Stoch RSI (k)
        if row.get("stoch_rsi_k", 50) < 20:
            votes["buy"] += 1
        elif row.get("stoch_rsi_k", 50) > 80:
            votes["sell"] += 1

        # Volume spike support
        vol_mean = ind_df["volume"].tail(50).mean() if len(ind_df) >= 50 else ind_df["volume"].mean()
        if row.get("volume", 0) > (vol_mean or 1) * 1.5:
            if row.get("macd_hist", 0) > 0:
                votes["buy"] += 1
            elif row.get("macd_hist", 0) < 0:
                votes["sell"] += 1

        score = votes["buy"] - votes["sell"]
        if score >= 2:
            rec = "BUY"
        elif score <= -2:
            rec = "SELL"
        else:
            rec = "HOLD"
        return rec, votes
    except Exception as e:
        logger.exception("Error scoring signals: %s", e)
        return "HOLD", {"buy": 0, "sell": 0, "neutral": 0}

# ------------------------------
# Charting
# ------------------------------
def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    try:
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={'height_ratios':[3,1,1]})
        ax_price, ax_macd, ax_rsi = axes

        ax_price.plot(df.index, df["close"], label=f"{symbol} close", linewidth=1)
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
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        logger.exception("Error creating chart: %s", e)
        return b""

# ------------------------------
# Report building
# ------------------------------
def build_hourly_report(symbol: str) -> Tuple[str, Optional[bytes]]:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    header_lines = [f"⏱ Hourly report for {symbol} — {now}\n"]
    votes_summary = {}
    last_image = None

    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = fetch_klines(symbol, tf_label, limit=tf_conf.get("limit", 200))
            if df.empty:
                raise RuntimeError("No data")
            ind = compute_indicators(df)
            rec, votes = score_signals(ind)
            # pivot / fib
            last_row = df.tail(1).iloc[0]
            last_candle = {"high": last_row["high"], "low": last_row["low"], "close": last_row["close"]}
            piv = pivot_support_resistance(last_candle)
            window = df.tail(60)
            low = float(window["low"].min())
            high = float(window["high"].max())
            fibs = fib_levels(low, high)

            header_lines.append(
                f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.8f} | Rec: {rec} | RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}"
            )
            header_lines.append(f"  Pivot:{piv['pivot']:.8f}, R1:{piv['r1']:.8f}, S1:{piv['s1']:.8f}")
            header_lines.append(f"  Fib 23.6%:{fibs['23.6%']:.8f} 38.2%:{fibs['38.2%']:.8f} 50%:{fibs['50.0%']:.8f}")
            votes_summary[tf_label] = {"rec": rec, "votes": votes}

            # pick 4h chart or last available
            if tf_label == "4h":
                last_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
            elif last_image is None:
                last_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
        except Exception as e:
            logger.exception("TF %s error for %s: %s", tf_label, symbol, e)
            header_lines.append(f"TF: {tf_label} — error computing data: {e}")

    # aggregate final recommendation
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for tf, data in votes_summary.items():
        counts[data["rec"]] = counts.get(data["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"

    # SL/TP heuristic
    try:
        latest_price = float(fetch_klines(symbol, "1h", limit=3)["close"].iloc[-1])
        if final == "BUY":
            sl = latest_price * 0.98
            tp = latest_price * 1.04
        elif final == "SELL":
            sl = latest_price * 1.02
            tp = latest_price * 0.96
        else:
            sl = latest_price * 0.995
            tp = latest_price * 1.005
    except Exception:
        sl = tp = 0.0

    header = f"Final recommendation for {symbol}: {final}\nSL: {sl:.8f} TP: {tp:.8f}\n"
    text = header + "\n".join(header_lines)
    return text, last_image

# ------------------------------
# Telegram helpers
# ------------------------------
def telegram_send_text(text: str) -> Tuple[Optional[int], str]:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = client.post(url, json=payload, timeout=15.0)
        return r.status_code, r.text
    except Exception as e:
        logger.exception("Telegram send message error: %s", e)
        return None, str(e)

def telegram_send_photo(image_bytes: bytes, caption: str = "") -> Tuple[Optional[int], str]:
    try:
        if not image_bytes:
            return None, "Empty image"
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", image_bytes, "image/png")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        r = client.post(url, data=data, files=files, timeout=30.0)
        return r.status_code, r.text
    except Exception as e:
        logger.exception("Telegram send photo error: %s", e)
        return None, str(e)

# ------------------------------
# Scheduler tasks
# ------------------------------
def hourly_task():
    logger.info("Running hourly_task")
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            # Telegram message safe cut to 4000 chars
            if text:
                telegram_send_text(text[:3900])
            if img:
                telegram_send_photo(img, caption=f"{sym} hourly chart")
            time.sleep(0.8)
        except Exception as e:
            logger.exception("hourly_task error for %s: %s", sym, e)
    logger.info("Hourly task finished")

def daily_task():
    logger.info("Running daily_task")
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            if text:
                telegram_send_text("📈 DAILY REPORT\n" + text[:3900])
            if img:
                telegram_send_photo(img, caption=f"{sym} daily simulation chart")
            time.sleep(0.8)
        except Exception as e:
            logger.exception("daily_task error for %s: %s", sym, e)
    logger.info("Daily task finished")

# Scheduler init
scheduler = BackgroundScheduler(timezone=SCHED_TZ)
scheduler.add_job(hourly_task, "cron", minute=1)     # hourly at HH:01 MSK
scheduler.add_job(daily_task, "cron", hour=4, minute=2)  # daily 04:02 MSK
scheduler.start()

# ------------------------------
# FastAPI app (keep-alive for Render & UptimeRobot)
# ------------------------------
app = FastAPI(title="crypto-signal-bot")

@app.get("/")
async def root():
    return {
        "status": "crypto-signal-bot running",
        "time": datetime.datetime.utcnow().isoformat(),
        "service": "crypto-signal-bot"
    }

# Explicit HEAD root (UptimeRobot may use HEAD)
@app.head("/")
async def head_root():
    return Response(status_code=200)

@app.get("/health")
async def health_check():
    """Health check - tries to fetch a small sample from Bybit"""
    try:
        df = fetch_klines("BTCUSDT", "1h", limit=2)
        ok = not df.empty
        return {"status": "healthy" if ok else "unhealthy", "sample_rows": len(df)}
    except Exception as e:
        return Response(
            content=json.dumps({"status": "unhealthy", "error": str(e)}),
            status_code=500,
            media_type="application/json"
        )

@app.get("/ping")
async def ping():
    return Response(content="pong", media_type="text/plain")

@app.head("/ping")
async def head_ping():
    return Response(status_code=200)

# ------------------------------
# Graceful shutdown
# ------------------------------
import atexit
@atexit.register
def shutdown():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass

# ------------------------------
# Run (main)
# ------------------------------
if __name__ == "__main__":
    try:
        # Startup Telegram message (best effort)
        telegram_send_text("✅ Crypto signal bot deployed and running 24/7 (Bybit data).")
    except Exception as e:
        logger.warning("Startup Telegram message failed: %s", e)

    import uvicorn
    logger.info(f"Starting server on 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
