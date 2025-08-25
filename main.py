# main.py
import os
import io
import math
import time
import json
import httpx
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from fastapi import FastAPI, Response
from apscheduler.schedulers.background import BackgroundScheduler

# TA indicators from `ta` package
from ta.momentum import RSIIndicator, StochasticOscillator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ------------------------------
# Config / environment
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
PORT = int(os.getenv("PORT", 10000))

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables")

# Exchanges endpoints (public)
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"
BYBIT_KLINES_ENDPOINT = "https://api.bybit.com/public/linear/kline"

SYMBOLS = ["DOGEUSDT", "ADAUSDT"]
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200}
}

# Scheduler timezone
SCHED_TZ = "Europe/Moscow"

# ------------------------------
# Utilities: HTTP fetch
# ------------------------------
client = httpx.Client(timeout=30.0)

def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch klines from Binance public API and return DataFrame with columns:
    ['open_time','open','high','low','close','volume','close_time',...]
    All price columns are float and index is datetime of open_time.
    """
    try:
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        r = client.get(BINANCE_KLINES_ENDPOINT, params=params)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise RuntimeError("Empty kline data")
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
    except Exception as e:
        print(f"Error fetching Binance data for {symbol}: {e}")
        raise

def fetch_bybit_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch klines from Bybit public API.
    interval here: "15", "60", "240", "720" (minutes) for Bybit.
    We'll map common intervals.
    """
    try:
        mapping = {"15m":"15", "1h":"60", "4h":"240", "12h":"720"}
        if interval not in mapping:
            raise ValueError("Unsupported interval for Bybit")
        params = {"symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
        r = client.get(BYBIT_KLINES_ENDPOINT, params=params)
        r.raise_for_status()
        jd = r.json()
        if "result" in jd and isinstance(jd["result"], list):
            data = jd["result"]
        elif "result" in jd and "list" in jd["result"]:
            data = jd["result"]["list"]
        else:
            data = jd.get("result", [])
        if not data:
            raise RuntimeError("Empty bybit kline data")
        df = pd.DataFrame(data)
        # Bybit format may contain 'open_time' or 'start_at'(unix)
        if "start_at" in df.columns:
            df["open_time"] = pd.to_datetime(df["start_at"], unit="s")
        elif "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
        else:
            # fallback: use 't' or first column
            df["open_time"] = pd.to_datetime(df.iloc[:,0], unit="s")
        df.set_index("open_time", inplace=True)
        # Expected columns: open, high, low, close, volume
        for col in ["open","high","low","close","volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
            elif col.capitalize() in df.columns:
                df[col] = df[col.capitalize()].astype(float)
            else:
                df[col] = 0.0
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        print(f"Error fetching Bybit data for {symbol}: {e}")
        raise

# ------------------------------
# Indicators calculation
# ------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with ['open','high','low','close','volume'] compute indicators and return new df copy:
      - rsi (14)
      - macd (fast=12, slow=26, signal=9): macd_line, macd_signal, macd_hist
      - bb: bb_mavg, bb_hband, bb_lband, bb_percent
      - sma20, sma50, ema20, ema50
      - stoch_rsi_k, stoch_rsi_d
    """
    try:
        out = df.copy()
        close = out["close"]

        # RSI 14
        rsi = RSIIndicator(close=close, window=14, fillna=True).rsi()
        out["rsi"] = rsi

        # MACD
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=True)
        out["macd_line"] = macd.macd()
        out["macd_signal"] = macd.macd_signal()
        out["macd_hist"] = macd.macd_diff()

        # Bollinger Bands (20, 2)
        bb = BollingerBands(close=close, window=20, window_dev=2, fillna=True)
        out["bb_mavg"] = bb.bollinger_mavg()
        out["bb_hband"] = bb.bollinger_hband()
        out["bb_lband"] = bb.bollinger_lband()
        # position relative to bands
        out["bb_pct"] = (close - out["bb_lband"]) / (out["bb_hband"] - out["bb_lband"] + 1e-12)

        # SMA / EMA
        out["sma20"] = SMAIndicator(close=close, window=20, fillna=True).sma_indicator()
        out["sma50"] = SMAIndicator(close=close, window=50, fillna=True).sma_indicator()
        out["ema20"] = EMAIndicator(close=close, window=20, fillna=True).ema_indicator()
        out["ema50"] = EMAIndicator(close=close, window=50, fillna=True).ema_indicator()

        # Stochastic RSI
        stoch_rsi = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3, fillna=True)
        out["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        out["stoch_rsi_d"] = stoch_rsi.stochrsi_d()

        return out
    except Exception as e:
        print(f"Error computing indicators: {e}")
        raise

# ------------------------------
# Support / Resistance and Fib
# ------------------------------
def pivot_support_resistance(series: pd.Series) -> Dict[str, float]:
    """
    Simple pivot calculation based on last candle:
      Pivot = (High + Low + Close) / 3
      R1 = 2*Pivot - Low
      S1 = 2*Pivot - High
    Return dict with pivot, r1, s1
    """
    try:
        high = float(series["high"])
        low = float(series["low"])
        close = float(series["close"])
        pivot = (high + low + close) / 3.0
        r1 = 2*pivot - low
        s1 = 2*pivot - high
        return {"pivot": pivot, "r1": r1, "s1": s1}
    except Exception as e:
        print(f"Error calculating pivot: {e}")
        return {"pivot": 0, "r1": 0, "s1": 0}

def fib_levels(last_low: float, last_high: float) -> Dict[str, float]:
    """
    Return Fibonacci retracement levels between last_low and last_high
    """
    try:
        diff = last_high - last_low
        levels = {
            "0.0%": last_high,
            "23.6%": last_high - 0.236*diff,
            "38.2%": last_high - 0.382*diff,
            "50.0%": last_high - 0.5*diff,
            "61.8%": last_high - 0.618*diff,
            "100.0%": last_low
        }
        return levels
    except Exception as e:
        print(f"Error calculating Fibonacci levels: {e}")
        return {}

# ------------------------------
# Signal logic (simple voting)
# ------------------------------
def score_signals(ind_df: pd.DataFrame) -> Tuple[str, Dict[str,int]]:
    """
    Given indicators DataFrame for a timeframe, compute a simple vote:
     - RSI < 30 => +1 buy, >70 => +1 sell
     - MACD histogram positive => buy, negative => sell
     - Price above ema20 => buy, below => sell
     - Bollinger: price near upper band => sell, near lower band => buy
     - StochRSI: <0.2 buy, >0.8 sell
    Sum votes and return final recommendation and votes dict.
    """
    try:
        row = ind_df.iloc[-1]
        votes = {"buy": 0, "sell": 0, "neutral": 0}
        price = row["close"]

        # RSI
        if row["rsi"] < 30:
            votes["buy"] += 1
        elif row["rsi"] > 70:
            votes["sell"] += 1
        else:
            votes["neutral"] += 0

        # MACD hist
        if row["macd_hist"] > 0:
            votes["buy"] += 1
        elif row["macd_hist"] < 0:
            votes["sell"] += 1

        # EMA20 trend
        if row["close"] > row["ema20"]:
            votes["buy"] += 1
        else:
            votes["sell"] += 1

        # Bollinger
        if row["bb_pct"] > 0.85:
            votes["sell"] += 1
        elif row["bb_pct"] < 0.15:
            votes["buy"] += 1

        # Stoch RSI
        if row["stoch_rsi_k"] < 20:
            votes["buy"] += 1
        elif row["stoch_rsi_k"] > 80:
            votes["sell"] += 1

        # Volume momentum: check last vs mean
        vol_mean = ind_df["volume"].tail(50).mean()
        if row["volume"] > vol_mean * 1.5:
            # high volume supports the direction of price move (use macd hist sign)
            if row["macd_hist"] > 0:
                votes["buy"] += 1
            elif row["macd_hist"] < 0:
                votes["sell"] += 1

        # Final
        score = votes["buy"] - votes["sell"]
        if score >= 2:
            recommendation = "BUY"
        elif score <= -2:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        return recommendation, votes
    except Exception as e:
        print(f"Error scoring signals: {e}")
        return "HOLD", {"buy": 0, "sell": 0, "neutral": 0}

# ------------------------------
# Charting / simulation
# ------------------------------
def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    """
    Create a multilayer chart with price and indicators and return PNG bytes.
    """
    try:
        plt.switch_backend('Agg')
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios':[3,1,1]})
        ax_price, ax_macd, ax_rsi = axes

        # Price plot with SMA/EMA and Bollinger bands
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

        # MACD
        if "macd_line" in df.columns:
            ax_macd.plot(df.index, df["macd_line"], label="MACD")
            ax_macd.plot(df.index, df["macd_signal"], label="Signal")
            ax_macd.bar(df.index, df["macd_hist"], label="Hist", alpha=0.6)
            ax_macd.legend(loc="upper left")
            ax_macd.grid(True)

        # RSI
        if "rsi" in df.columns:
            ax_rsi.plot(df.index, df["rsi"], label="RSI")
            ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.6)
            ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.6)
            ax_rsi.legend(loc="upper left")
            ax_rsi.grid(True)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"Error creating chart: {e}")
        # Return empty bytes if chart creation fails
        return b""

# ------------------------------
# Report creation
# ------------------------------
def build_hourly_report(symbol: str) -> Tuple[str, bytes]:
    """
    For a symbol, compute indicators on multiple timeframes and create
    a text report + PNG image (for the most relevant timeframe).
    Return (text, image_bytes).
    """
    now = datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    report_lines = [f"⏱ Hourly report for {symbol} — {now}\n"]

    # accumulate votes per timeframe
    votes_summary = {}
    last_imgs = None

    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = fetch_binance_klines(symbol, tf_conf["interval"], limit=tf_conf["limit"])
            ind = compute_indicators(df)
            rec, votes = score_signals(ind)
            # support/resistance using last candle
            last = df.tail(3)
            last_candle = {"high": last["high"].iloc[-1], "low": last["low"].iloc[-1], "close": last["close"].iloc[-1]}
            piv = pivot_support_resistance(last_candle)
            # fib using last swing low/high (simple min/max of tail window)
            window = df.tail(60)
            low = float(window["low"].min())
            high = float(window["high"].max())
            fibs = fib_levels(low, high)

            report_lines.append(f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.6f} | Rec: {rec} | RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}")
            report_lines.append(f"  Pivot:{piv['pivot']:.6f}, R1:{piv['r1']:.6f}, S1:{piv['s1']:.6f}")
            report_lines.append(f"  Fib 23.6%:{fibs['23.6%']:.6f} 38.2%:{fibs['38.2%']:.6f} 50%:{fibs['50.0%']:.6f}")
            votes_summary[tf_label] = {"rec": rec, "votes": votes}

            # keep image for highest timeframe (4h preferred) or last loop
            if tf_label == "4h":
                last_imgs = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
            elif last_imgs is None:
                last_imgs = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
        except Exception as e:
            report_lines.append(f"TF: {tf_label} — error fetching or computing: {e}")

    # combine final recommendation by simple majority across timeframes
    counts = {"BUY":0,"SELL":0,"HOLD":0}
    for tf, data in votes_summary.items():
        counts[data["rec"]] = counts.get(data["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"

    # Suggested SL/TP: simple ATR-like placeholder (use small percent)
    try:
        latest_price = float(fetch_binance_klines(symbol, "1h", limit=3)["close"].iloc[-1])
        sl = latest_price * (0.98 if final=="BUY" else 1.02)  # 2% stop loss heuristic
        tp = latest_price * (1.04 if final=="BUY" else 0.96)  # 4% take profit heuristic
    except Exception:
        sl = tp = 0.0

    header = f"Final recommendation for {symbol}: {final}\nSL: {sl:.6f} TP: {tp:.6f}\n"
    text = header + "\n".join(report_lines)
    return text, last_imgs

# ------------------------------
# Telegram helpers (send text + image)
# ------------------------------
def telegram_send_text(text: str):
    """Send text message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = client.post(url, json=payload, timeout=15.0)
        return r.status_code, r.text
    except Exception as e:
        print("Telegram send message error:", e)
        return None, str(e)

def telegram_send_photo(image_bytes: bytes, caption: str = ""):
    """Send photo to Telegram"""
    try:
        if not image_bytes:
            return None, "Empty image"
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", image_bytes, "image/png")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        r = client.post(url, data=data, files=files, timeout=30.0)
        return r.status_code, r.text
    except Exception as e:
        print("Telegram send photo error:", e)
        return None, str(e)

# ------------------------------
# Scheduler tasks
# ------------------------------
def hourly_task():
    """
    Runs each hour: fetch reports for configured symbols, send aggregated messages.
    """
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            # Send text first (Telegram has limits), then image with caption summary
            telegram_send_text(text[:4000])  # Telegram message size safe cut
            if img:
                caption = f"{sym} hourly chart (simulation)"
                telegram_send_photo(img, caption=caption)
            time.sleep(1.0)
        except Exception as e:
            print("Error in hourly_task for", sym, e)

def daily_task():
    """
    Runs daily at 04:00 Moscow time. Sends a full daily report + simulation image for each symbol.
    """
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)  # reuse build function (multi-tf inside)
            header = f"📈 DAILY REPORT (simulation) for {sym}\n"
            telegram_send_text(header + text[:4000])
            if img:
                caption = f"{sym} daily simulation chart"
                telegram_send_photo(img, caption=caption)
            time.sleep(1.0)
        except Exception as e:
            print("Error in daily_task for", sym, e)

# ------------------------------
# Scheduler initialization
# ------------------------------
scheduler = BackgroundScheduler(timezone=SCHED_TZ)
# At minute 1 each hour to avoid exact 00 collisions, or you can use minute=0.
scheduler.add_job(hourly_task, "cron", minute=1)   # runs hourly at HH:01 Moscow time
scheduler.add_job(daily_task, "cron", hour=4, minute=2)   # daily at 04:02 Moscow
scheduler.start()

# ------------------------------
# FastAPI application (keep-alive)
# ------------------------------
app = FastAPI(title="Crypto Signal Bot")

@app.get("/")
async def root():
    return {
        "status": "crypto-signal-bot running", 
        "time": datetime.datetime.utcnow().isoformat(),
        "service": "crypto-signal-bot"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Binance API connectivity
        test_df = fetch_binance_klines("BTCUSDT", "1m", 1)
        return {
            "status": "healthy",
            "binance_api": "available",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        return Response(
            content=json.dumps({
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat()
            }),
            status_code=500,
            media_type="application/json"
        )

@app.get("/status")
async def status():
    """Status endpoint for uptimerobot"""
    return {"status": "ok", "message": "Service is running"}

# ------------------------------
# Graceful shutdown
# ------------------------------
import atexit
@atexit.register
def shutdown_scheduler():
    scheduler.shutdown()

# ------------------------------
# Boot message + start server when main
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    # send startup message
    try:
        startup_text = "✅ Crypto signal bot deployed and running 24/7 (simulation charts)."
        telegram_send_text(startup_text)
    except Exception as e:
        print("Startup telegram message failed:", e)
    print(f"Starting uvicorn on 0.0.0.0:{PORT}")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=PORT,
        timeout_keep_alive=300
    )
