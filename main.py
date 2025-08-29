import os
import logging
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.types import Update
from aiogram.utils.executor import start_webhook

# ----------------------------
# Конфигурация
# ----------------------------
TOKEN = os.getenv("BOT_TOKEN")  # Твой токен бота из Render Secret
WEBHOOK_HOST = os.getenv("RENDER_EXTERNAL_URL")  # Render сам выдаёт HTTPS-домен
WEBHOOK_PATH = f"/webhook/{TOKEN}"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

logging.basicConfig(level=logging.INFO)

# ----------------------------
# FastAPI приложение
# ----------------------------
app = FastAPI()


@app.on_event("startup")
async def on_startup():
    """Устанавливаем вебхук в Telegram при запуске"""
    logging.info("Setting webhook to %s", WEBHOOK_URL)
    await bot.set_webhook(WEBHOOK_URL)


@app.on_event("shutdown")
async def on_shutdown():
    """Удаляем вебхук при остановке"""
    logging.info("Deleting webhook")
    await bot.delete_webhook()


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    """Обработка апдейтов от Telegram"""
    data = await request.json()
    update = Update(**data)
    await dp.process_update(update)
    return {"ok": True}

# ----------------------------
# Хэндлеры бота
# ----------------------------

@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я твой бот на Render 🚀")


@dp.message_handler(commands=["ping"])
async def ping(message: types.Message):
    await message.answer("Pong 🏓")


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(f"Ты написал: {message.text}")
