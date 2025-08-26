import os
import logging
import datetime
import httpx
import matplotlib.pyplot as plt
from io import BytesIO

from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# -------------------
# Логирование
# -------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# -------------------
# Конфигурация
# -------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # ID твоего телеграм-чата
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# -------------------
# FastAPI
# -------------------
app = FastAPI()

# -------------------
# Telegram Bot
# -------------------
tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# -------------------
# Функции работы с Bybit
# -------------------
async def fetch_klines(symbol: str, interval: str = "60", limit: int = 3):
    """Получение свечей с Bybit (публичные данные)"""
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        if "result" in data and "list" in data["result"]:
            return data["result"]["list"]
        return []

async def generate_report(symbol: str = "BTCUSDT"):
    """Формирование текста отчёта"""
    try:
        candles = await fetch_klines(symbol, interval="60", limit=3)
        if not candles:
            return f"⛔ Ошибка: пустые данные для {symbol}"

        # простая обработка (берём цены закрытия)
        closes = [float(c[4]) for c in candles]
        avg_price = sum(closes) / len(closes)

        text = (
            f"📊 Report for {symbol}\n"
            f"⏱ Time: {datetime.datetime.utcnow().isoformat()}\n"
            f"Last close prices: {closes}\n"
            f"Average: {avg_price:.4f}\n"
        )
        return text
    except Exception as e:
        logger.error(f"Ошибка в generate_report: {e}")
        return f"⚠ Ошибка при генерации отчёта для {symbol}: {e}"

# -------------------
# Команды Telegram
# -------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я бот для отчётов с Bybit.")

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /test — вручную получить отчёт"""
    report = await generate_report("BTCUSDT")
    await update.message.reply_text(report)

# Регистрируем команды
tg_app.add_handler(CommandHandler("start", start_command))
tg_app.add_handler(CommandHandler("test", test_command))

# -------------------
# Планировщик задач
# -------------------
scheduler = AsyncIOScheduler()

async def hourly_task():
    """Ежечасный отчёт"""
    report = await generate_report("BTCUSDT")
    await tg_app.bot.send_message(chat_id=CHAT_ID, text="⏱ Hourly Report:\n" + report)

async def daily_task():
    """Ежедневный отчёт"""
    report = await generate_report("BTCUSDT")
    await tg_app.bot.send_message(chat_id=CHAT_ID, text="📌 Daily Report:\n" + report)

scheduler.add_job(hourly_task, "cron", minute=0)
scheduler.add_job(daily_task, "cron", hour=0, minute=0)

# -------------------
# FastAPI события
# -------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Запуск приложения")
    scheduler.start()
    await tg_app.initialize()
    await tg_app.start()
    logger.info("✅ Telegram Bot и Scheduler запущены")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Остановка приложения")
    scheduler.shutdown()
    await tg_app.stop()
    await tg_app.shutdown()
    logger.info("✅ Telegram Bot и Scheduler остановлены")

# -------------------
# HTTP endpoint для Render
# -------------------
@app.get("/")
async def root():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}
