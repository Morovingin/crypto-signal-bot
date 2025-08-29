import os
import logging
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Update

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("BOT_TOKEN")  # положи сюда токен через Render secrets
WEBHOOK_PATH = f"/webhook/{TOKEN}"
WEBHOOK_URL = f"https://crypto-signal-bot-1-md7v.onrender.com{WEBHOOK_PATH}"

bot = Bot(token=TOKEN)
dp = Dispatcher()

app = FastAPI()


# === Хэндлеры ===
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я живой и работаю через вебхуки 🚀")


@dp.message()
async def echo(message: types.Message):
    await message.answer(f"Ты написал: {message.text}")


# === FastAPI routes ===
@app.on_event("startup")
async def on_startup():
    await bot.set_webhook(WEBHOOK_URL)
    logging.info(f"Webhook set to {WEBHOOK_URL}")


@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook()
    logging.info("Webhook deleted")


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.model_validate(data)
    await dp.feed_update(bot, update)
    return {"ok": True}
