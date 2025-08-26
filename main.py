import os
import io
import logging
import httpx
import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# =======================
# Настройки
# =======================
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI()

# Планировщик
scheduler = AsyncIOScheduler()

# Telegram Bot
tg_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()


# =======================
# Bybit API
# =======================
async def fetch_klines(symbol: str, interval: str = "60", limit: int = 3, category: str = "spot"):
    """
    Загружаем свечи с Bybit API
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        data = resp.json()

        # Логируем ответ в Render
        logger.info(f"Bybit API response for {symbol}/{category}: {data}")

        return data.get("result", {}).get("list", [])


# =======================
# Генерация отчёта
# =======================
async def generate_report(symbol: str = "BTCUSDT"):
    klines = await fetch_klines(symbol, interval="60", limit=3, category="spot")

    if not klines:
        return f"⚠️ Нет данных от Bybit для {symbol}"

    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    df["close"] = df["close"].astype(float)

    # Строим график
    plt.figure(figsize=(6, 3))
    plt.plot(df["close"], marker="o")
    plt.title(f"{symbol} — последние {len(df)} свечей")
    plt.xlabel("Свечи")
    plt.ylabel("Цена")
    plt.grid()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return buf


# =======================
# Telegram Handlers
# =======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Бот работает! Используй /test для проверки отчёта.")


async def test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    buf = await generate_report("BTCUSDT")
    if isinstance(buf, str):
        await update.message.reply_text(buf)
    else:
        await update.message.reply_photo(photo=buf, caption="Тестовый отчёт ✅")


# =======================
# Задачи APScheduler
# =======================
async def hourly_task():
    buf = await generate_report("BTCUSDT")
    if isinstance(buf, str):
        await tg_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=buf)
    else:
        await tg_app.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=buf, caption="Ежечасный отчёт")


async def daily_task():
    buf = await generate_report("BTCUSDT")
    if isinstance(buf, str):
        await tg_app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=buf)
    else:
        await tg_app.bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=buf, caption="Ежедневный отчёт")


# =======================
# FastAPI Events
# =======================
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Запуск приложения...")

    # Команды Telegram
    tg_app.add_handler(CommandHandler("start", start))
    tg_app.add_handler(CommandHandler("test", test))

    # Планировщик
    scheduler.add_job(hourly_task, "interval", hours=1, id="hourly_task")
    scheduler.add_job(daily_task, "interval", days=1, id="daily_task")
    scheduler.start()

    # Запускаем Telegram бота в фоне
    tg_app.create_task(tg_app.run_polling())


@app.get("/")
async def root():
    return {"status": "ok", "message": "FastAPI сервер работает 🚀"}
