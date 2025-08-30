# main.py
import os
import io
import time
import json
import logging
import datetime
from typing import Dict, Tuple

import httpx
import numpy as np
import pandas as pd
import matplotlib

# avoid permission issues for font cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Request, BackgroundTasks, Header
from apscheduler.schedulers.background import BackgroundScheduler

# TA indicators
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crypto-signal-bot")

# ------------------------------
# Config / environment
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET")
PORT = int(os.getenv("PORT", "10000"))
EXCHANGE = os.getenv("EXCHANGE", "BYBIT").upper()  # BYBIT or BINANCE
SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")

if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN not set")
if not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_CHAT_ID not set - reports to fixed chat will fail")

# API endpoints
BYBIT_KLINES_ENDPOINT = "https://api.bybit.com/v5/market/kline"
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
client = httpx.Client(timeout=30.0)

# ------------------------------
# Helpers: fetch klines
# ------------------------------
def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = client.get(BINANCE_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Empty kline data from Binance")
    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_asset_volume", "num_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_bybit_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    mapping = {"15m": "15", "1h": "60", "4h": "240", "12h": "720"}
    if interval not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + interval)
    params = {"symbol": symbol.upper(), "interval": mapping[interval], "limit": limit}
    r = client.get(BYBIT_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    jd = r.json()

    if jd.get("retCode", 0) != 0:
        raise RuntimeError("Bybit API error: " + str(jd))

    res = jd.get("result", {})
    data = []
    if isinstance(res, dict) and "list" in res:
        data = res["list"]
    elif isinstance(res, list):
        data = res
    if not data:
        raise RuntimeError("Empty kline data from Bybit: " + json.dumps(jd)[:500])

    df = pd.DataFrame(data)
    if "start" in df.columns:
        df["open_time"] = pd.to_datetime(df["start"], unit="s")
    elif "start_at" in df.columns:
        df["open_time"] = pd.to_datetime(df["start_at"], unit="s")
    elif "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="s")
    else:
        df["open_time"] = pd.to_datetime(df.iloc[:, 0], unit="s")

    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
        elif col.capitalize() in df.columns:
            df[col] = df[col.capitalize()].astype(float)
        else:
            df[col] = 0.0
    return df[["open", "high", "low", "close", "volume"]]


def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    try:
        if EXCHANGE == "BYBIT":
            return fetch_bybit_klines(symbol, interval, limit=limit)
        else:
            return fetch_binance_klines(symbol, interval, limit=limit)
    except Exception:
        logger.exception("fetch_klines failed for %s %s", symbol, interval)
        raise

# ------------------------------
# Indicators
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

# ------------------------------
# Support / Fib helpers
# ------------------------------
def pivot_support_resistance(series: pd.Series) -> Dict[str, float]:
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

# ------------------------------
# Signal scoring
# ------------------------------
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
        return "HOLD", {"buy": 0, "sell": 0, "neutral": 0}

# ------------------------------
# Charting
# ------------------------------
def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    try:
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1, 1]})
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
        logger.exception("plot_price_and_indicators failed")
        return b""

# ------------------------------
# Report building
# ------------------------------
def build_hourly_report(symbol: str) -> Tuple[str, bytes]:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"⏱ Hourly report for {symbol} — {now}\n"]
    votes_summary = {}
    chosen_image = b""

    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = fetch_klines(symbol, tf_conf["interval"], limit=tf_conf["limit"])
            if df.empty or len(df) < 10:
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

            lines.append(f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.6f} | Rec: {rec} | "
                         f"RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}")
            lines.append(f" Pivot:{piv['pivot']:.6f}, R1:{piv['r1']:.6f}, S1:{piv['s1']:.6f}")
            lines.append(f" Fib 23.6%:{fibs['23.6%']:.6f} 38.2%:{fibs['38.2%']:.6f} 50%:{fibs['50.0%']:.6f}")

            votes_summary[tf_label] = {"rec": rec, "votes": votes}

            if tf_label == "4h" and not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
            elif not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)
        except Exception:
            logger.exception("build_hourly_report: error for %s %s", symbol, tf_label)
            lines.append(f"TF: {tf_label} — error computing data")

    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for tf, d in votes_summary.items():
        counts[d["rec"]] = counts.get(d["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"

    try:
        latest_price = float(fetch_klines(symbol, "1h", limit=3)["close"].iloc[-1])
        sl = latest_price * (0.98 if final == "BUY" else 1.02)
        tp = latest_price * (1.04 if final == "BUY" else 0.96)
    except Exception:
        sl = tp = 0.0

    header = f"Final recommendation for {symbol}: {final}\nSL: {sl:.6f} TP: {tp:.6f}\n"
    text = header + "\n".join(lines)
    return text, chosen_image

# ------------------------------
# Telegram helpers
# ------------------------------
def telegram_send_text(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("telegram_send_text skipped: missing token/chat_id")
        return None, "missing token/chat"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = client.post(url, json=payload, timeout=15.0)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_text failed")
        return None, "error"


def telegram_send_photo(image_bytes: bytes, caption: str = ""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("telegram_send_photo skipped: missing token/chat_id")
        return None, "
