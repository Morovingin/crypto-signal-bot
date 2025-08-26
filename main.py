import os
import asyncio
import logging
import datetime
import pandas as pd
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from tvDatafeed import TvDatafeed, Interval

# ---------------------------------------------------
# ЛОГИРОВАНИЕ
# ---------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# TELEGRAM
# ---------------------------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # твой render URL

application = Application.builder().token(TELEGRAM_TOKEN).build()

# ---------------------------------------------------
# TRADINGVIEW API
# ---------------------------------------------------
tv = TvDatafeed()  # анонимный вход, можно добавить логин/пароль если надо

interval_map = {
    "15m": Interval.in_15_minute,
    "1h": Interval.in_1_hour,
    "4h": Interval.in_4_hour,
    "12h": Interval.in_12_hour,
    "1d": Interval.in_daily,
}

async def fetch_tradingview_klines(symbol: str, interval: str, limit: int = 200):
    try:
        # Приводим к виду ADA/USDT → ADAUSDT
        if symbol.endswith("USDT"):
            ticker = symbol.replace("USDT", "/USDT")
        else:
            ticker = symbol

        bars = tv.get_hist(symbol=ticker, exchange="BINANCE", interval=interval_map[interval], n_bars=limit)
        if bars is None or bars.empty:
            raise RuntimeError("TradingView returned empty data")
        return bars
    except Exception as e:
        logger.error(f"TradingView fetch failed for {symbol} {interval}: {e}")
        raise

async def fetch_klines(symbol: str, interval: str, limit: int = 200):
    return await fetch_tradingview_klines(symbol, interval, limit)

# ---------------------------------------------------
# ОБРАБОТЧИКИ КОМАНД
# ---------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для анализа криптовалют. Используй /fast для проверки сигнала.")

async def fast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol = "DOGEUSDT"
    try:
        df = await fetch_klines(symbol, "1h", limit=100)
        last_close = df["close"].iloc[-1]
        await update.message.reply_text(f"{symbol} (1H) последняя цена: {last_close}")
    except Exception as e:
        await update.message.reply_text(f"Ошибка получения данных: {e}")

# ---------------------------------------------------
# TELEGRAM ROUTES
# ---------------------------------------------------
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("fast", fast))

# ---------------------------------------------------
# FASTAPI
# ---------------------------------------------------
app = FastAPI()

@app.post(f"/telegram_webhook/{TELEGRAM_TOKEN}")
async def telegram_webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, application.bot)
    await application.process_update(update)
    return {"ok": True}

@app.get("/")
async def root():
    return {
        "status": "ok",
        "time": datetime.datetime.utcnow().isoformat()
    }

# ---------------------------------------------------
# СТАРТ
# ---------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
