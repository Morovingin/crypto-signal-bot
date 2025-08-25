import os, io, math, time, json, asyncio, random, textwrap
from datetime import datetime, timezone, timedelta
import pytz
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# ТА-библиотека (без компиляции, подходит для фри-хостинга)
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Конфиг --------------------
PAIRS = os.getenv("PAIRS", "DOGEUSDT,ADAUSDT").upper().split(",")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")  # опционально
BYBIT_BASE = "https://api.bybit.com"
TZ_MOSCOW = pytz.timezone("Europe/Moscow")
APP_TZ = timezone.utc  # всё планируем в UTC (04:00 МСК = 01:00 UTC)
HEADERS = {"User-Agent": "crypto-signal-bot/1.0"}

# Интервалы Bybit: '15','60','240','720' для 15m/1h/4h/12h
TIMEFRAMES = {"15m": "15", "1h": "60", "4h": "240", "12h": "720"}

# Хранилище последнего отчёта в памяти
LAST_REPORT_TEXT = ""
LAST_DAILY_IMAGE = {}  # {"DOGEUSDT": bytes, ...}

# -------------------- Веб-приложение --------------------
app = FastAPI()

@app.get("/healthz")
async def healthz():
    return {"ok": True, "utc": datetime.now(timezone.utc).isoformat()}

@app.get("/last_report")
async def last_report():
    return Response(LAST_REPORT_TEXT or "Пока нет отчётов", media_type="text/plain; charset=utf-8")

# -------------------- Вспомогательные функции --------------------
def ts_to_dt_ms(ms):
    return datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc)

async def bybit_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """
    Bybit v5 Market Kline (spot)
    https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=60
    """
    url = f"{BYBIT_BASE}/v5/market/kline"
    params = {"category": "spot", "symbol": symbol, "interval": interval, "limit": str(limit)}
    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"Bybit error: {data}")
    rows = data["result"]["list"]  # [[start,open,high,low,close,volume,turnover], ...] newest first
    rows = list(reversed(rows))    # делаем от старых к новым
    cols = ["start","open","high","low","close","volume","turnover"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","turnover"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["datetime"] = pd.to_datetime(df["start"].astype("int64"), unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Скользящие
    df["ema20"]  = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"]  = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()
    df["sma20"]  = SMAIndicator(df["close"], window=20).sma_indicator()
    # RSI/StochRSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    st = StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
    df["stochrsi_k"] = st.stochrsi_k()
    df["stochrsi_d"] = st.stochrsi_d()
    # MACD
    macd = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    # Bollinger
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_up"]  = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    # ATR (для SL/TP)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()
    # MA по объёмам (простая)
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    return df

def recent_swing_levels(df: pd.DataFrame, lookback: int = 250):
    d = df.tail(lookback)
    hi = float(d["high"].max())
    lo = float(d["low"].min())
    # На случай вырожденности
    if not math.isfinite(hi) or not math.isfinite(lo) or hi <= lo:
        return None
    # Уровни Фибоначчи (0..1 на отрезке lo-hi)
    levels = {
        "0%": lo,
        "23.6%": lo + 0.236*(hi-lo),
        "38.2%": lo + 0.382*(hi-lo),
        "50.0%": lo + 0.5*(hi-lo),
        "61.8%": lo + 0.618*(hi-lo),
        "78.6%": lo + 0.786*(hi-lo),
        "100%": hi,
    }
    return levels

def local_support_resistance(df: pd.DataFrame, window: int = 10, topn: int = 3):
    # Простая детекция локальных экстремумов
    lows  = df["low"]
    highs = df["high"]
    sups = []
    ress = []
    for i in range(window, len(df)-window):
        if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
            sups.append((df.index[i], float(lows.iloc[i])))
        if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
            ress.append((df.index[i], float(highs.iloc[i])))
    # Берём последние и уникализируем уровни (по расстоянию)
    def pick(levels):
        levels = sorted(levels, key=lambda x: x[0], reverse=True)
        res = []
        for _, price in levels:
            if all(abs(price - p) / p > 0.01 for p in res):  # не ближе 1%
                res.append(price)
            if len(res) >= topn:
                break
        return sorted(res)
    return pick(sups), pick(ress)

def signal_from_indicators(df: pd.DataFrame):
    """Возвращает ('BUY'|'SELL'|'NEUTRAL', причины:list[str]) по последней свече."""
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row
    reasons = []
    score = 0

    # Тренд MA/EMA
    if row.close > row.ema50 > row.ema200:
        score += 2; reasons.append("цена>EMA50>EMA200 (бычий тренд)")
    elif row.close < row.ema50 < row.ema200:
        score -= 2; reasons.append("цена<EMA50<EMA200 (медвежий тренд)")

    # MACD
    if row.macd_hist > 0 and prev.macd_hist <= 0:
        score += 2; reasons.append("MACD: бычий кроссовер")
    elif row.macd_hist < 0 and prev.macd_hist >= 0:
        score -= 2; reasons.append("MACD: медвежий кроссовер")
    else:
        if row.macd_hist > 0: score += 0.5
        if row.macd_hist < 0: score -= 0.5

    # RSI
    if row.rsi < 30: score += 1.5; reasons.append("RSI<30 (перепроданность)")
    elif row.rsi > 70: score -= 1.5; reasons.append("RSI>70 (перекупленность)")
    elif row.rsi > 55: score += 0.5
    elif row.rsi < 45: score -= 0.5

    # StochRSI
    if row.stochrsi_k > 0.8 and row.stochrsi_k < prev.stochrsi_k:
        score -= 1; reasons.append("StochRSI разворот сверху")
    if row.stochrsi_k < 0.2 and row.stochrsi_k > prev.stochrsi_k:
        score += 1; reasons.append("StochRSI разворот снизу")

    # Bollinger Bands (контртренд)
    if row.close <= row.bb_low: score += 1; reasons.append("касание нижней BB")
    if row.close >= row.bb_up:  score -= 1; reasons.append("касание верхней BB")

    # Объёмы
    if row.volume > (row.vol_sma20 * 1.3):
        reasons.append("объём выше среднего (импульс)")
        score += 0.5

    decision = "NEUTRAL"
    if score >= 2: decision = "BUY"
    if score <= -2: decision = "SELL"
    return decision, reasons, row

def sl_tp_from_atr(side: str, price: float, atr: float, supports: list, resistances: list):
    """Рекомендации по SL/TP на основе ATR и ближайших S/R."""
    rr = 1.5
    if side == "BUY":
        # SL чуть ниже ближайшей поддержки или 1.2*ATR
        sl = min([s for s in supports if s < price] + [price - 1.2*atr])
        tp1 = price + rr*atr
        tp2 = price + 2*rr*atr
    elif side == "SELL":
        sl = max([r for r in resistances if r > price] + [price + 1.2*atr])
        tp1 = price - rr*atr
        tp2 = price - 2*rr*atr
    else:
        sl = None; tp1=None; tp2=None
    return sl, tp1, tp2

async def telegram_send_text(text: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID): return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text[:3900], "parse_mode":"HTML"})

async def telegram_send_photo(png_bytes: bytes, caption: str):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID): return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    files = {"photo": ("chart.png", png_bytes, "image/png")}
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        await client.post(url, data=data, files=files)

async def fetch_m2_latest():
    """Опционально: последний M2 (M2SL, monthly, SA) из FRED API."""
    if not FRED_API_KEY:
        return None
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": "M2SL", "api_key": FRED_API_KEY, "file_type": "json"}
    async with httpx.AsyncClient(timeout=30.0, headers=HEADERS) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    obs = [o for o in data.get("observations", []) if o.get("value") not in (".", None)]
    if not obs:
        return None
    last = obs[-1]
    return {"date": last["date"], "value": float(last["value"])}

def make_daily_simulation_chart(symbol: str, df_1h: pd.DataFrame) -> bytes:
    """Строим PNG: свечи (1ч), индикаторы и 24-часовую стох. симуляцию."""
    close = df_1h["close"]
    # Волатильность по последним ~7 дням (24*7 часов)
    hourly_ret = close.pct_change().dropna()
    sigma = float(hourly_ret.tail(24*7).std())
    price0 = float(close.iloc[-1])

    steps = 24
    paths = 200
    sims = np.zeros((paths, steps+1))
    sims[:,0] = price0
    for p in range(paths):
        shocks = np.random.normal(loc=0.0, scale=sigma, size=steps)
        series = [price0]
        for e in shocks:
            series.append(series[-1] * (1.0 + e))
        sims[p,:] = series
    median_path = np.median(sims, axis=0)
    p10 = np.percentile(sims, 10, axis=0)
    p90 = np.percentile(sims, 90, axis=0)

    # График
    fig = plt.figure(figsize=(10,5), dpi=150)
    ax = plt.gca()

    # Свечной упрощённый (OHLC -> линии high/low+close)
    dfc = df_1h.tail(24*7)  # последние 7 дней
    t = range(len(dfc))
    ax.plot(t, dfc["close"].values, label="Close (1h)")
    # Индикаторы
    dfi = compute_indicators(dfc)
    ax.plot(t, dfi["ema50"].values, label="EMA50")
    ax.plot(t, dfi["ema200"].values, label="EMA200")
    # Диапазон симуляции
    future_x = range(len(dfc), len(dfc)+steps+1)
    ax.plot(future_x, median_path, linestyle="--", label="Sim median (24h)")
    ax.fill_between(future_x, p10, p90, alpha=0.2, label="Sim P10–P90")

    ax.set_title(f"{symbol} — модель движения на 24ч (данные Bybit, 1h)")
    ax.set_xlabel("Часы (последние 7 дней + 24ч вперёд)")
    ax.set_ylabel("Цена (USDT)")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":")

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def fmt_price(x: float) -> str:
    if x >= 1: return f"{x:.4f}"
    if x >= 0.1: return f"{x:.5f}"
    return f"{x:.8f}"

async def build_report_for_symbol(symbol: str) -> str:
    lines = [f"<b>{symbol}</b>"]
    for tf_name, interval in TIMEFRAMES.items():
        df = await bybit_klines(symbol, interval, limit=600)
        dfi = compute_indicators(df)
        fib = recent_swing_levels(dfi, lookback=250)
        sups, ress = local_support_resistance(dfi, window=10, topn=3)
        decision, reasons, row = signal_from_indicators(dfi)

        atr = float(row.atr) if math.isfinite(row.atr) else None
        sl, tp1, tp2 = (None, None, None)
        if atr and decision in ("BUY","SELL"):
            sl, tp1, tp2 = sl_tp_from_atr(decision, float(row.close), atr, sups, ress)

        # Короткая строка по таймфрейму
        fib_str = ""
        if fib:
            fib_str = f"Fibo 38.2%={fmt_price(fib['38.2%'])}, 61.8%={fmt_price(fib['61.8%'])}"
        sr_str = ""
        if sups or ress:
            if sups: sr_str += "S:" + ",".join(fmt_price(x) for x in sups[:2])
            if ress: sr_str += (" " if sr_str else "") + "R:" + ",".join(fmt_price(x) for x in ress[:2])

        sltp = ""
        if sl and tp1 and tp2:
            sltp = f" | SL {fmt_price(sl)} / TP1 {fmt_price(tp1)} / TP2 {fmt_price(tp2)}"

        lines.append(
            f"• {tf_name}: {decision} @ {fmt_price(float(row.close))} | "
            f"RSI {row.rsi:.1f} | MACD {'+' if row.macd_hist>0 else '-'} | BB [{fmt_price(float(row.bb_low))}…{fmt_price(float(row.bb_up))}] | {fib_str} | {sr_str}{sltp}"
        )
        # Добавим «почему»
        if reasons:
            lines.append("  └ причины: " + "; ".join(reasons[:3]))

    return "\n".join(lines)

async def hourly_job():
    global LAST_REPORT_TEXT
    parts = []
    for sym in PAIRS:
        try:
            parts.append(await build_report_for_symbol(sym))
        except Exception as e:
            parts.append(f"{sym}: ошибка данных ({e})")
    m2 = await fetch_m2_latest()
    if m2:
        parts.append(f"M2 (FRED): {m2['value']:.1f} млрд, дата {m2['date']}")
    text = "📊 <b>Ежечасный отчёт</b>\n" + "\n\n".join(parts)
    LAST_REPORT_TEXT = text
    await telegram_send_text(text)

async def daily_job_0400msk():
    """Раз в сутки 04:00 МСК (01:00 UTC): 1) симуляции 2) краткий вывод по покупкам/продажам."""
    global LAST_DAILY_IMAGE
    for sym in PAIRS:
        df_1h = await bybit_klines(sym, TIMEFRAMES["1h"], limit=800)
        png = make_daily_simulation_chart(sym, df_1h)
        LAST_DAILY_IMAGE[sym] = png
        caption = f"{sym}: модель 24ч вперёд. Это НЕ финансовый совет."
        await telegram_send_photo(png, caption)

    # Короткое резюме (по 1h)
    parts = []
    for sym in PAIRS:
        dfi = compute_indicators(await bybit_klines(sym, TIMEFRAMES["1h"], limit=400))
        dec, reasons, row = signal_from_indicators(dfi)
        parts.append(f"{sym} 1h: {dec} @ {fmt_price(float(row.close))} — " + "; ".join(reasons[:3]))
    msg = "🕓 04:00 МСК — дневной отчёт\n" + "\n".join(parts) + "\nM2 учитывается при наличии ключа."
    await telegram_send_text(msg)

# -------------------- Планировщик --------------------
scheduler = AsyncIOScheduler(timezone=APP_TZ)
# Каждый час в ноль минут
scheduler.add_job(hourly_job, CronTrigger(minute="0"))
# Ежедневно в 01:00 UTC == 04:00 МСК
scheduler.add_job(daily_job_0400msk, CronTrigger(hour="1", minute="0"))

@app.on_event("startup")
async def on_startup():
    scheduler.start()

# Локальный запуск для отладки
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
