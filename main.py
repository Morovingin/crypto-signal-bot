import os
import logging
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from dotenv import load_dotenv

# =======================
# 🔹 Настройки логирования
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

# =======================
# 🔹 Загрузка переменных окружения
# =======================
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN не найден в .env")

if not CHAT_ID:
    raise ValueError("❌ CHAT_ID не найден в .env")

# =======================
# 🔹 FastAPI приложение
# =======================
app = FastAPI(title="Trading Bot Reporter")

# =======================
# 🔹 Планировщик
# =======================
scheduler = AsyncIOScheduler()

# =======================
# 🔹 Telegram Application
# =======================
tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# =======================
# 🔹 Функции отчёта
# =======================
def generate_fake_report():
    """Фейковые данные для теста"""
    now = datetime.now()
    times = [now - timedelta(hours=i) for i in range(24)][::-1]
    prices = np.cumsum(np.random.randn(24)) + 100

    df = pd.DataFrame({"time": times, "price": prices})
    return df

def plot_report(df: pd.DataFrame, filename: str = "report.png") -> str:
    """Сохраняет график в PNG"""
    plt.figure(figsize=(10, 4))
    plt.plot(df["time"], df["price"], label="Price")
    plt.title("Test Report")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

async def send_report(context: ContextTypes.DEFAULT_TYPE, test: bool = False):
    """Отправка отчёта в Telegram"""
    try:
        df = generate_fake_report()
        filename = plot_report(df, "report.png")

        title = "📊 Тестовый отчёт" if test else "📈 Ежечасный отчёт"
        await context.bot.send_message(chat_id=CHAT_ID, text=title)
        await context.bot.send_photo(chat_id=CHAT_ID, photo=open(filename, "rb"))

        logger.info("✅ Отчёт отправлен в Telegram (%s)", title)
    except Exception as e:
        logger.error("❌ Ошибка при отправке отчёта: %s", str(e))

# =======================
# 🔹 Команды Telegram
# =======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Привет! Я бот-репортёр. Доступные команды:\n/test — тестовый отчёт")

async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Генерация тестового отчёта...")
    await send_report(context, test=True)

# =======================
# 🔹 Задачи планировщика
# =======================
async def hourly_task():
    """Ежечасный отчёт"""
    class DummyContext:
        bot = tg_app.bot
    await send_report(DummyContext(), test=False)

async def daily_task():
    """Ежедневный отчёт"""
    class DummyContext:
        bot = tg_app.bot
    await send_report(DummyContext(), test=False)

# =======================
# 🔹 FastAPI события
# =======================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Запуск приложения...")

    # Регистрируем команды
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(CommandHandler("test", test))

    # Планировщик
    scheduler.add_job(hourly_task, IntervalTrigger(hours=1), id="hourly_task")
    scheduler.add_job(daily_task, IntervalTrigger(days=1), id="daily_task")
    scheduler.start()

    # Запускаем Telegram-бота внутри uvicorn event loop
    asyncio.create_task(tg_app.run_polling())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Остановка приложения...")
    scheduler.shutdown()

# =======================
# 🔹 Root endpoint
# =======================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Trading bot reporter is running"}
