# main.py
import os
import io
import time
import json
import logging
import datetime
from typing import Dict, Tuple, List, Any, Optional

import httpx
import numpy as np
import pandas as pd

# Matplotlib: каталог кэша в /tmp (на Render иначе бывают права)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response
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
PORT = int(os.getenv("PORT", "10000"))

# Какую биржу пробуем первой: BYBIT или BINANCE
EXCHANGE = os.getenv("EXCHANGE", "BYBIT").upper()
# Для Bybit v5 укажи категорию: spot | linear | inverse
BYBIT_CATEGORY = os.getenv("BYBIT_CATEGORY", "linear").lower()

SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")

# Включить пуллинг команд из Telegram (/test)? 1 = да
TELEGRAM_POLL_UPDATES = os.getenv("TELEGRAM_POLL_UPDATES", "0") == "1"

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning("TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы — отправка в Telegram не сработает")

# Endpoints
BYBIT_KLINES_V5 = "https://api.bybit.com/v5/market/kline"
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"

# Символы и таймфреймы
SYMBOLS = ["DOGEUSDT", "ADAUSDT"]
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h": {"interval": "1h", "limit": 200},
    "4h": {"interval": "4h", "limit": 200},
    "12h": {"interval": "12h", "limit": 200},
}

# ------------------------------
# HTTP client (общий, с заголовками)
# ------------------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoSignalBot/1.0; +https://render.com)",
    "Accept": "application/json",
}
client = httpx.Client(timeout=30.0, headers=DEFAULT_HEADERS, follow_redirects=True)

# ------------------------------
# Helpers: fetch klines
# ------------------------------
def _bybit_interval_str(tf: str) -> str:
    # Bybit v5 ждёт минуты строкой: 1, 3, 5, 15, 60, 240, 720, 1440
    mapping = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "12h": "720", "1d": "1440"}
    if tf not in mapping:
        raise ValueError("Unsupported interval for Bybit: " + tf)
    return mapping[tf]

def fetch_bybit_klines(symbol: str, interval: str, limit: int = 200, category: str = "linear") -> pd.DataFrame:
    """
    Bybit v5: https://api.bybit.com/v5/market/kline
    Пары возвращаются как result.list = [[start, open, high, low, close, volume, turnover], ...] (строки)
    start/ts в миллисекундах.
    """
    params = {
        "category": category,             # "linear" для USDT perp, "spot" для спота
        "symbol": symbol.upper(),
        "interval": _bybit_interval_str(interval),
        "limit": str(min(limit, 1000)),
    }

    # Пара попыток с небольшим бэкофом
    last_exc: Optional[Exception] = None
    for attempt in range(2):
        try:
            r = client.get(BYBIT_KLINES_V5, params=params)
            r.raise_for_status()
            jd = r.json()
            if not isinstance(jd, dict) or jd.get("retCode") != 0:
                raise RuntimeError(f"Bybit retCode={jd.get('retCode')} retMsg={jd.get('retMsg')}")

            res = jd.get("result", {})
            data = None
            # Формат v5: {"result": {"category":"linear","symbol":"ADAUSDT","list":[[ts,open,high,low,close,volume,turnover], ...]}}
            if isinstance(res, dict) and "list" in res:
                data = res["list"]

            if not data:
                raise RuntimeError(f"Bybit v5: empty list for {symbol} {interval}")

            # Собираем DataFrame
            rows: List[Dict[str, Any]] = []
            for it in data:
                # ожидаем массив строк
                if isinstance(it, (list, tuple)) and len(it) >= 6:
                    ts = int(it[0])  # миллисекунды
                    # если вдруг секунды — приведём
                    if ts < 10**11:  # очень маленькое — секунды
                        ts = ts * 1000
                    row = {
                        "open_time": pd.to_datetime(ts, unit="ms"),
                        "open": float(it[1]),
                        "high": float(it[2]),
                        "low": float(it[3]),
                        "close": float(it[4]),
                        "volume": float(it[5]),
                    }
                    rows.append(row)
                elif isinstance(it, dict):
                    # защитный путь на случай другого формата
                    ts = int(it.get("start", it.get("timestamp", it.get("open_time", 0))))
                    if ts < 10**11:
                        ts = ts * 1000
                    row = {
                        "open_time": pd.to_datetime(ts, unit="ms"),
                        "open": float(it.get("open", 0.0)),
                        "high": float(it.get("high", 0.0)),
                        "low": float(it.get("low", 0.0)),
                        "close": float(it.get("close", 0.0)),
                        "volume": float(it.get("volume", 0.0)),
                    }
                    rows.append(row)

            if not rows:
                raise RuntimeError("Bybit v5 parsed 0 rows")

            df = pd.DataFrame(rows).sort_values("open_time")
            df.set_index("open_time", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]
        except Exception as e:
            last_exc = e
            # 403/429 → иногда помогает небольшой бэкоф
            time.sleep(0.4)
    # если не получилось
    raise RuntimeError(f"Bybit fetch failed: {last_exc}")

def fetch_binance_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
    r = client.get(BINANCE_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Empty kline data from Binance")
    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Unified fetch: сначала выбранная EXCHANGE, при неудаче — фолбэк на другую.
    """
    first = EXCHANGE
    second = "BINANCE" if EXCHANGE == "BYBIT" else "BYBIT"

    # 1) первая попытка
    try:
        if first == "BYBIT":
            df = fetch_bybit_klines(symbol, interval, limit=limit, category=BYBIT_CATEGORY)
        else:
            df = fetch_binance_klines(symbol, interval, limit=limit)
        if df.empty:
            raise RuntimeError(f"{first} returned empty")
        return df
    except Exception as e1:
        logger.warning("fetch_klines primary %s failed for %s %s: %s", first, symbol, interval, e1)

    # 2) фолбэк
    try:
        if second == "BYBIT":
            df = fetch_bybit_klines(symbol, interval, limit=limit, category=BYBIT_CATEGORY)
        else:
            df = fetch_binance_klines(symbol, interval, limit=limit)
        if df.empty:
            raise RuntimeError(f"{second} returned empty")
        logger.info("fetch_klines fallback used: %s for %s %s", second, symbol, interval)
        return df
    except Exception as e2:
        logger.error("fetch_klines fallback %s failed for %s %s: %s", second, symbol, interval, e2)
        raise

# ------------------------------
# Indicators
# ------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    # RSI
    out["rsi"] = RSIIndicator(close=close, window=14, fillna=True).rsi()
    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=True)
    out["macd_line"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    # Bollinger
    bb = BollingerBands(close=close, window=20, window_dev=2, fillna=True)
    out["bb_mavg"] = bb.bollinger_mavg()
    out["bb_hband"] = bb.bollinger_hband()
    out["bb_lband"] = bb.bollinger_lband()
    out["bb_pct"] = (close - out["bb_lband"]) / (out["bb_hband"] - out["bb_lband"] + 1e-12)
    # SMA/EMA
    out["sma20"] = SMAIndicator(close=close, window=20, fillna=True).sma_indicator()
    out["sma50"] = SMAIndicator(close=close, window=50, fillna=True).sma_indicator()
    out["ema20"] = EMAIndicator(close=close, window=20, fillna=True).ema_indicator()
    out["ema50"] = EMAIndicator(close=close, window=50, fillna=True).ema_indicator()
    # Stoch RSI
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
    if diff <= 0:
        # защитный случай
        return {
            "0.0%": last_high,
            "23.6%": last_high,
            "38.2%": last_high,
            "50.0%": last_high,
            "61.8%": last_high,
            "100.0%": last_low,
        }
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
        # RSI
        if row["rsi"] < 30:
            votes["buy"] += 1
        elif row["rsi"] > 70:
            votes["sell"] += 1
        # MACD hist
        if row["macd_hist"] > 0:
            votes["buy"] += 1
        elif row["macd_hist"] < 0:
            votes["sell"] += 1
        # EMA trend
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
        # Volume momentum
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
    lines = [f"⏱️ Hourly report for {symbol} — {now}\n"]
    votes_summary: Dict[str, Dict[str, Any]] = {}
    chosen_image: bytes = b""

    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = fetch_klines(symbol, tf_conf["interval"], limit=tf_conf["limit"])
            if df.empty or len(df) < 30:
                lines.append(f"TF: {tf_label} — error computing data: No data (empty dataframe)")
                continue

            ind = compute_indicators(df)
            rec, votes = score_signals(ind)

            last = df.tail(3)
            lastc = {"high": last["high"].iloc[-1], "low": last["low"].iloc[-1], "close": last["close"].iloc[-1]}
            piv = pivot_support_resistance(lastc)

            window = df.tail(120) if len(df) >= 120 else df
            low = float(window["low"].min())
            high = float(window["high"].max())
            fibs = fib_levels(low, high)

            lines.append(
                f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.6f} | Rec: {rec} | "
                f"RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}"
            )
            lines.append(
                f"  Pivot:{piv['pivot']:.6f}, R1:{piv['r1']:.6f}, S1:{piv['s1']:.6f}"
            )
            lines.append(
                f"  Fib 23.6%:{fibs['23.6%']:.6f} 38.2%:{fibs['38.2%']:.6f} 50%:{fibs['50.0%']:.6f}"
            )

            votes_summary[tf_label] = {"rec": rec, "votes": votes}

            # картинку делаем один раз (предпочтительно 4h)
            if (tf_label == "4h" and not chosen_image) or not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)

        except Exception as e:
            logger.exception("build_hourly_report: error for %s %s", symbol, tf_label)
            lines.append(f"TF: {tf_label} — error computing data: {e}")

    # Aggregate final
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for tf, d in votes_summary.items():
        counts[d["rec"]] = counts.get(d["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"

    # SL/TP только если есть цена
    sl_text = "n/a"
    tp_text = "n/a"
    try:
        latest = fetch_klines(symbol, "1h", limit=3)
        if not latest.empty:
            latest_price = float(latest["close"].iloc[-1])
            sl = latest_price * (0.98 if final == "BUY" else 1.02)
            tp = latest_price * (1.04 if final == "BUY" else 0.96)
            sl_text = f"{sl:.8f}"
            tp_text = f"{tp:.8f}"
    except Exception as e:
        logger.warning("SL/TP calc failed for %s: %s", symbol, e)

    header = f"Final recommendation for {symbol}: {final}\nSL: {sl_text} TP: {tp_text}\n"
    text = header + "\n".join(lines)
    return text, chosen_image

# ------------------------------
# Telegram helpers (HTTP)
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
        return None, "missing token/chat"
    if not image_bytes:
        logger.warning("telegram_send_photo skipped: empty image")
        return None, "empty image"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    try:
        r = client.post(url, data=data, files=files, timeout=30.0)
        r.raise_for_status()
        return r.status_code, r.text
    except Exception:
        logger.exception("telegram_send_photo failed")
        return None, "error"

# ------------------------------
# Tasks
# ------------------------------
def hourly_task():
    logger.info("Hourly task started")
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            telegram_send_text(text[:3800])
            if img:
                telegram_send_photo(img, caption=f"{sym} hourly chart")
            time.sleep(0.5)
        except Exception:
            logger.exception("hourly_task error for %s", sym)
    logger.info("Hourly task finished")

def daily_task():
    logger.info("Daily task started")
    for sym in SYMBOLS:
        try:
            text, img = build_hourly_report(sym)
            telegram_send_text("📈 DAILY REPORT\n" + text[:3800])
            if img:
                telegram_send_photo(img, caption=f"{sym} daily chart")
            time.sleep(0.5)
        except Exception:
            logger.exception("daily_task error for %s", sym)
    logger.info("Daily task finished")

# ------------------------------
# Optional: Telegram /test через getUpdates
# ------------------------------
_tg_offset = 0

def _poll_telegram_updates():
    global _tg_offset
    if not TELEGRAM_POLL_UPDATES or not TELEGRAM_BOT_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
        params = {"timeout": 0, "offset": _tg_offset}
        r = client.post(url, json=params, timeout=15.0)
        r.raise_for_status()
        upd = r.json()
        if not upd or not upd.get("ok"):
            return
        for item in upd.get("result", []):
            _tg_offset = max(_tg_offset, int(item.get("update_id", 0)) + 1)
            msg = item.get("message") or item.get("edited_message")
            if not msg:
                continue
            chat_id = str(msg.get("chat", {}).get("id"))
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            # Обрабатываем только свой чат (безопаснее)
            if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
                continue

            if text.startswith("/test"):
                parts = text.split()
                sym = SYMBOLS[0]
                if len(parts) >= 2 and parts[1].upper() in SYMBOLS:
                    sym = parts[1].upper()
                telegram_send_text(f"🔄 Running test report for {sym} ...")
                try:
                    report, img = build_hourly_report(sym)
                    telegram_send_text(report[:3800])
                    if img:
                        telegram_send_photo(img, caption=f"{sym} test chart")
                except Exception as e:
                    telegram_send_text(f"❌ Test failed: {e}")

    except Exception:
        logger.exception("poll_telegram_updates failed")

# ------------------------------
# Scheduler
# ------------------------------
scheduler = BackgroundScheduler(timezone=SCHED_TZ)
scheduler.add_job(hourly_task, "cron", minute=1, id="hourly_task")
scheduler.add_job(daily_task, "cron", hour=4, minute=2, id="daily_task")
# Пуллинг Telegram команд — каждые 10 сек (если включён)
if TELEGRAM_POLL_UPDATES:
    scheduler.add_job(_poll_telegram_updates, "interval", seconds=10, id="tg_poll")

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Crypto Signal Bot")

# GET и HEAD на корне
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "crypto-signal-bot running",
        "time": datetime.datetime.utcnow().isoformat(),
        "service": "crypto-signal-bot",
        "exchange": EXCHANGE,
        "bybit_category": BYBIT_CATEGORY,
    }

@app.get("/health")
async def health_check():
    try:
        # Делаем простую проверку на 15m
        df = fetch_klines("BTCUSDT", "15m", 5)
        ok = (not df.empty) and np.isfinite(df["close"].iloc[-1])
        return {
            "status": "healthy" if ok else "no-data",
            "exchange": EXCHANGE,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "rows": int(len(df)),
        }
    except Exception as e:
        logger.exception("health_check failure")
        return Response(
            content=json.dumps(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }
            ),
            status_code=500,
            media_type="application/json",
        )

@app.get("/status")
async def status():
    return {"status": "ok", "message": "Service is running"}

@app.get("/ping")
async def ping():
    return Response(content="pong", media_type="text/plain")

@app.post("/test")
async def test_endpoint(symbol: Optional[str] = None):
    """
    Ручной тест без Telegram-команд (HTTP): POST /test?symbol=ADAUSDT
    """
    sym = symbol.upper() if symbol else SYMBOLS[0]
    if sym not in SYMBOLS:
        return Response(
            content=json.dumps({"ok": False, "error": f"Symbol {sym} not in list: {SYMBOLS}"}),
            status_code=400,
            media_type="application/json",
        )
    try:
        text, img = build_hourly_report(sym)
        telegram_send_text("🧪 TEST REPORT\n" + text[:3800])
        if img:
            telegram_send_photo(img, caption=f"{sym} test chart")
        return {"ok": True, "symbol": sym}
    except Exception as e:
        logger.exception("test_endpoint failed")
        return Response(
            content=json.dumps({"ok": False, "error": str(e)}),
            status_code=500,
            media_type="application/json",
        )

# ------------------------------
# Lifespan events
# ------------------------------
@app.on_event("startup")
def on_startup():
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")
    try:
        telegram_send_text("✅ Crypto signal bot deployed and running 24/7.")
    except Exception:
        logger.exception("startup telegram message failed")

@app.on_event("shutdown")
def on_shutdown():
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Scheduler shutdown")
    except Exception:
        logger.exception("Failed to shutdown scheduler")

# ------------------------------
# Local run
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:%s (exchange=%s)", PORT, EXCHANGE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
