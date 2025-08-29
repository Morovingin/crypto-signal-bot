# main.py
import os
import io
import time
import json
import logging
import datetime
from typing import Dict, Tuple, List, Optional

import httpx
import numpy as np
import pandas as pd
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Response, Request, HTTPException
from apscheduler.schedulers.background import BackgroundScheduler

# TA indicators
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

# Telegram webhook (aiogram 2.x)
from aiogram import Bot, Dispatcher, types
from aiogram.utils.exceptions import TelegramAPIError

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crypto-signal-bot")

# ------------------------------
# Config / env
# ------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Если хотите авто-установку вебхука – задайте TELEGRAM_WEBHOOK_URL
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()  # полный URL до /telegram_webhook
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()  # ?token=...

PORT = int(os.getenv("PORT", "10000"))
EXCHANGE = os.getenv("EXCHANGE", "BINANCE").upper()  # BINANCE или BYBIT
SCHED_TZ = os.getenv("SCHED_TZ", "Europe/Moscow")

SYMBOLS: List[str] = os.getenv("SYMBOLS", "DOGEUSDT,ADAUSDT").replace(" ", "").split(",")

# Таймфреймы: 15m, 1h, 4h, 1d
TIMEFRAMES = {
    "15m": {"interval": "15m", "limit": 200},
    "1h":  {"interval": "1h",  "limit": 200},
    "4h":  {"interval": "4h",  "limit": 200},
    "1d":  {"interval": "1d",  "limit": 400},
}

# Binance/Bybit endpoints
BINANCE_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"
BYBIT_V5_KLINES_ENDPOINT = "https://api.bybit.com/v5/market/kline"  # category=spot

# Optional proxies (Render: set HTTP_PROXY/HTTPS_PROXY if нужен обход)
HTTP_PROXY = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
HTTPS_PROXY = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
proxies = None
if HTTP_PROXY or HTTPS_PROXY:
    proxies = {
        "http://": HTTP_PROXY or HTTPS_PROXY,
        "https://": HTTPS_PROXY or HTTP_PROXY,
    }
    logger.info("HTTP(S) proxy configured")

# Persistent clients
client = httpx.Client(timeout=30.0, proxies=proxies)
client_short = httpx.Client(timeout=10.0, proxies=proxies)

# Telegram bot/dispatcher
bot: Optional[Bot] = None
dp: Optional[Dispatcher] = None
if TELEGRAM_BOT_TOKEN:
    bot = Bot(token=TELEGRAM_BOT_TOKEN, parse_mode="HTML")
    dp = Dispatcher(bot)
else:
    logger.warning("TELEGRAM_BOT_TOKEN not set – Telegram features disabled")

# ------------------------------
# Utility: chunk long text for Telegram (4096 limit)
# ------------------------------
def chunk_text(s: str, limit: int = 4096) -> List[str]:
    parts = []
    while s:
        parts.append(s[:limit])
        s = s[limit:]
    return parts

# ------------------------------
# Fetch klines: Binance -> Bybit -> yfinance
# ------------------------------
def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    # Leaves only open/high/low/close/volume indexed by datetime
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].astype(float)
    df = df[cols]
    df = df.dropna()
    return df

def fetch_binance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = client.get(BINANCE_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise RuntimeError("Empty from Binance")
    df = pd.DataFrame(
        data,
        columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","num_trades",
            "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df = df.rename(columns=str.lower)
    return _normalize_ohlcv_df(df)

def fetch_bybit_v5_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    # Bybit v5 for spot: category=spot; interval in {"1","3","5","15","30","60","120","240","360","720","D","W","M"}
    bybit_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    if interval not in bybit_map:
        raise ValueError("Unsupported interval for Bybit v5: " + interval)
    params = {
        "category": "spot",
        "symbol": symbol.upper(),
        "interval": bybit_map[interval],
        "limit": str(limit),
    }
    r = client.get(BYBIT_V5_KLINES_ENDPOINT, params=params)
    r.raise_for_status()
    jd = r.json()
    if not isinstance(jd, dict) or jd.get("retCode") not in (0, "0"):
        raise RuntimeError(f"Bybit v5 retCode != 0: {str(jd)[:200]}")
    lst = (jd.get("result") or {}).get("list") or []
    if not lst:
        raise RuntimeError("Empty from Bybit v5")
    # Bybit v5: each row [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(lst, columns=["start","open","high","low","close","volume","turnover"])
    df["start"] = pd.to_datetime(df["start"], unit="s", utc=True)
    df = df.set_index("start").rename(columns=str.lower)
    return _normalize_ohlcv_df(df)

def fetch_yfinance_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    import yfinance as yf

    # map symbol → Yahoo ticker
    mapping = {
        "BTCUSDT": "BTC-USD",
        "ETHUSDT": "ETH-USD",
        "BNBUSDT": "BNB-USD",
        "DOGEUSDT": "DOGE-USD",
        "ADAUSDT": "ADA-USD",
        "XRPUSDT": "XRP-USD",
        "SOLUSDT": "SOL-USD",
    }
    ticker = mapping.get(symbol.upper(), None)
    if not ticker:
        raise RuntimeError(f"yfinance: no mapping for {symbol}")

    # choose period big enough
    if interval == "15m":
        yf_interval, period = "15m", "7d"
    elif interval == "1h":
        yf_interval, period = "1h", "30d"
    elif interval == "4h":
        # fetch 1h and resample later
        yf_interval, period = "1h", "60d"
    else:  # 1d
        yf_interval, period = "1d", "730d"

    df = yf.Ticker(ticker).history(period=period, interval=yf_interval, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned empty for {ticker}")

    df = df.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })
    df.index = pd.to_datetime(df.index, utc=True)

    if interval == "4h":
        # resample 1h → 4h
        agg = {
            "open":"first","high":"max","low":"min","close":"last","volume":"sum"
        }
        df = df.resample("4H").agg(agg).dropna(how="any")

    df = df[["open","high","low","close","volume"]].dropna(how="any")
    if len(df) > limit:
        df = df.iloc[-limit:]
    return df

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """Cascade: preferred EXCHANGE first, then the other, then yfinance."""
    last_exc = None
    sources: List[str] = []

    try:
        if EXCHANGE == "BINANCE":
            sources = ["BINANCE", "BYBIT", "YF"]
        else:
            sources = ["BYBIT", "BINANCE", "YF"]
    except Exception:
        sources = ["BINANCE", "BYBIT", "YF"]

    for src in sources:
        try:
            if src == "BINANCE":
                return fetch_binance_klines(symbol, interval, limit)
            elif src == "BYBIT":
                return fetch_bybit_v5_klines(symbol, interval, limit)
            else:
                return fetch_yfinance_klines(symbol, interval, limit)
        except Exception as e:
            last_exc = e
            logger.warning("fetch_klines %s %s failed on %s: %s", symbol, interval, src, str(e))

    raise RuntimeError(f"All sources failed for {symbol} {interval}. Last error: {last_exc}")

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
        high = float(series["high"]); low = float(series["low"]); close = float(series["close"])
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
            votes["buy" if row["macd_hist"] > 0 else "sell"] += 1

        score = votes["buy"] - votes["sell"]
        if score >= 2: return "BUY", votes
        if score <= -2: return "SELL", votes
        return "HOLD", votes
    except Exception:
        return "HOLD", {"buy": 0, "sell": 0, "neutral": 0}

# ------------------------------
# Charting
# ------------------------------
def plot_price_and_indicators(df: pd.DataFrame, symbol: str, timeframe: str) -> bytes:
    try:
        plt.switch_backend("Agg")
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
        ax_price, ax_macd, ax_rsi = axes
        ax_price.plot(df.index, df["close"], label=f"{symbol} close")
        if "sma20" in df.columns: ax_price.plot(df.index, df["sma20"], label="SMA20", linewidth=0.8)
        if "ema20" in df.columns: ax_price.plot(df.index, df["ema20"], label="EMA20", linewidth=0.8)
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
# Fundamental (light) via CoinGecko (best effort, не ломает отчёт)
# ------------------------------
def get_fundamentals(symbol: str) -> str:
    coin_id = None
    if symbol.upper().startswith("DOGE"): coin_id = "dogecoin"
    elif symbol.upper().startswith("ADA"): coin_id = "cardano"
    if not coin_id:
        return ""

    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "ids": coin_id}
        r = client_short.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if not data: return ""
        item = data[0]
        mc = item.get("market_cap")
        ch24 = item.get("price_change_percentage_24h")
        vol = item.get("total_volume")
        return f"Fundamentals: MC=${mc:,} | 24h Δ={ch24:.2f}% | 24h Vol=${vol:,}"
    except Exception as e:
        logger.warning("Fundamentals fetch failed %s: %s", symbol, e)
        return ""

# ------------------------------
# Report builder
# ------------------------------
def build_report(symbol: str) -> Tuple[str, bytes]:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"⏱ Report for {symbol} — {now}\n"]
    votes_summary = {}
    chosen_image = b""

    for tf_label, tf_conf in TIMEFRAMES.items():
        try:
            df = fetch_klines(symbol, tf_conf["interval"], limit=tf_conf["limit"])
            if df.empty or len(df) < 30:
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

            lines.append(
                f"TF: {tf_label} | Price: {df['close'].iloc[-1]:.6f} | Rec: <b>{rec}</b> "
                f"| RSI:{ind['rsi'].iloc[-1]:.1f} | MACD_hist:{ind['macd_hist'].iloc[-1]:.6f}"
            )
            lines.append(
                f"  Pivot:{piv['pivot']:.6f}, R1:{piv['r1']:.6f}, S1:{piv['s1']:.6f}"
            )
            lines.append(
                f"  Fib 23.6%:{fibs['23.6%']:.6f} 38.2%:{fibs['38.2%']:.6f} 50%:{fibs['50.0%']:.6f} 61.8%:{fibs['61.8%']:.6f}"
            )

            votes_summary[tf_label] = {"rec": rec, "votes": votes}

            # картинка — предпочтительно 4h
            if (tf_label == "4h" and not chosen_image) or not chosen_image:
                chosen_image = plot_price_and_indicators(ind.tail(200), symbol, tf_label)

        except Exception:
            logger.exception("build_report: error for %s %s", symbol, tf_label)
            lines.append(f"TF: {tf_label} — error computing data")

    # Итог
    counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for _, d in votes_summary.items():
        counts[d["rec"]] = counts.get(d["rec"], 0) + 1
    final = max(counts.items(), key=lambda x: x[1])[0] if counts else "HOLD"

    # SL/TP (1h close)
    try:
        latest_price = float(fetch_klines(symbol, "1h", limit=3)["close"].iloc[-1])
        sl = latest_price * (0.98 if final == "BUY" else 1.02)
        tp = latest_price * (1.04 if final == "BUY" else 0.96)
    except Exception:
        sl = 0.0; tp = 0.0

    fundamentals = get_fundamentals(symbol)
    header = (
        f"<b>Final recommendation for {symbol}: {final}</b>\n"
        f"SL: {sl:.6f}  TP: {tp:.6f}\n"
    )
    if fundamentals:
        header += fundamentals + "\n"

    text = header + "\n".join(lines)
    return text, chosen_image

# ------------------------------
# Telegram helpers (HTTP API, для задач планировщика)
# ------------------------------
def telegram_send_text(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("telegram_send_text skipped: missing token/chat_id")
        return None, "missing token/chat"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        for chunk in chunk_text(text, 4096):
            r = client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "HTML"}, timeout=15.0)
            r.raise_for_status()
        return 200, "ok"
    except Exception:
        logger.exception("telegram_send_text failed")
        return None, "error"

def telegram_send_photo(image_bytes: bytes, caption: str = ""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("telegram_send_photo skipped: missing token/chat_id")
        return None, "missing token/chat"
    if not image_bytes:
        return None, "empty image"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", image_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
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
            text, img = build_report(sym)
            telegram_send_text(text[:3800])
            if img:
                telegram_send_photo(img, caption=f"{sym} chart")
            time.sleep(0.5)
        except Exception:
            logger.exception("hourly_task error for %s", sym)
    logger.info("Hourly task finished")

def daily_task():
    logger.info("Daily task started")
    for sym in SYMBOLS:
        try:
            text, img = build_report(sym)
            telegram_send_text("📈 DAILY REPORT\n" + text[:3800])
            if img:
                telegram_send_photo(img, caption=f"{sym} daily chart")
            time.sleep(0.5)
        except Exception:
            logger.exception("daily_task error for %s", sym)
    logger.info("Daily task finished")

# ------------------------------
# Scheduler
# ------------------------------
scheduler = BackgroundScheduler(timezone=SCHED_TZ)
scheduler.add_job(hourly_task, "cron", minute=1, id="hourly_task")
scheduler.add_job(daily_task, "cron", hour=4, minute=2, id="daily_task")

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Crypto Signal Bot")

# Root for UptimeRobot — поддерживаем HEAD и GET
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "ok",
        "service": "crypto-signal-bot",
        "time": datetime.datetime.utcnow().isoformat() + "Z",
        "exchange": EXCHANGE,
    }

# Health — НИКАКИХ внешних вызовов, всегда 200
@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "healthy", "time": datetime.datetime.utcnow().isoformat() + "Z"}

# Ping — текстовый ответ, тоже GET/HEAD
@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return Response(content="pong", media_type="text/plain")

# (Опционально) HTTP-триггер быстрого отчёта — удобно для ручного теста
@app.get("/fast")
async def fast_http():
    parts = []
    for sym in SYMBOLS:
        try:
            text, _ = build_report(sym)
            parts.append(text)
        except Exception as e:
            parts.append(f"{sym}: error: {e}")
    return Response("\n\n".join(parts), media_type="text/plain")

# ------------------------------
# Telegram webhook endpoint
# ------------------------------
@app.post("/telegram_webhook")
async def telegram_webhook(request: Request):
    if not dp or not bot:
        raise HTTPException(status_code=503, detail="Telegram not configured")

    # Проверка секретного токена (если задан)
    if TELEGRAM_WEBHOOK_SECRET:
        token = request.query_params.get("token", "")
        if token != TELEGRAM_WEBHOOK_SECRET:
            raise HTTPException(status_code=403, detail="Forbidden")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")

    update = types.Update(**data)

    # Aiogram 2.x: фиксируем текущие объекты в контексте
    from aiogram import Bot as _Bot
    from aiogram.dispatcher.dispatcher import Dispatcher as _Dispatcher
    _Bot.set_current(bot)           # важно!
    _Dispatcher.set_current(dp)     # важно!

    try:
        await dp.process_update(update)
    except TelegramAPIError as e:
        logger.error("TelegramAPIError: %s", e)
    except Exception as e:
        logger.exception("Webhook processing failed: %s", e)
    return Response(status_code=200)

# ------------------------------
# Aiogram handlers (/start, /fast)
# ------------------------------
if dp:
    @dp.message_handler(commands=["start", "help"])
    async def cmd_start(message: types.Message):
        await message.answer(
            "👋 Привет! Доступные команды:\n"
            "/fast — быстрый отчёт по DOGE/USDT и ADA/USDT (15m, 1h, 4h, 1d)\n"
            "Ежечасные сигналы приходят автоматически."
        )

    @dp.message_handler(commands=["fast"])
    async def cmd_fast(message: types.Message):
        from aiogram import Bot as _Bot
        from aiogram.dispatcher.dispatcher import Dispatcher as _Dispatcher
        _Bot.set_current(bot)
        _Dispatcher.set_current(dp)

        for sym in SYMBOLS:
            try:
                text, img = build_report(sym)
                for chunk in chunk_text(text, 4096):
                    await message.answer(chunk)
                if img:
                    # aiogram photo upload
                    await message.answer_photo(types.InputFile(io.BytesIO(img), filename="chart.png"))
            except Exception as e:
                logger.error("Ошибка /fast for %s: %s", sym, e)
                await message.answer(f"⚠️ Ошибка при получении данных для {sym}.")

# ------------------------------
# Startup / Shutdown
# ------------------------------
@app.on_event("startup")
def on_startup():
    try:
        if not scheduler.running:
            scheduler.start()
            logger.info("Scheduler started")
    except Exception:
        logger.exception("Failed to start scheduler")

    # Telegram webhook autoconfigure (если URL задан)
    if bot and TELEGRAM_WEBHOOK_URL:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"
            wh_url = TELEGRAM_WEBHOOK_URL
            if TELEGRAM_WEBHOOK_SECRET:
                delim = "&" if "?" in wh_url else "?"
                wh_url = f"{wh_url}{delim}token={TELEGRAM_WEBHOOK_SECRET}"
            payload = {"url": wh_url, "allowed_updates": ["message"]}
            r = client.post(url, json=payload, timeout=10.0)
            r.raise_for_status()
            logger.info("Telegram webhook set to %s", wh_url)
        except Exception:
            logger.exception("Failed to set Telegram webhook")

    # Стартовый пинг в чат
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
# Entrypoint
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server on 0.0.0.0:%s (exchange=%s)", PORT, EXCHANGE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, timeout_keep_alive=300)
