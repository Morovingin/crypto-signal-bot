import os
import asyncio
import logging
import datetime
from fastapi import FastAPI, Request
from aiogram import Bot, Dispatcher, types
from aiogram.utils.executor import start_webhook
import yfinance as yf

# ----------------------------
# Конфигурация
# ----------------------------
API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # токен телеграм-бота
WEBHOOK_HOST = os.getenv("RENDER_EXTERNAL_URL", "https://crypto-signal-bot.onrender.com")
WEBHOOK_PATH = "/telegram_webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = int(os.getenv("PORT", 10000))

# ----------------------------
# Логирование
# ----------------------------
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Telegram bot
# ----------------------------
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()


@app.get("/")
async def root():
    """Health-check для UptimeRobot"""
    return {
        "status": "ok",
        "time": datetime.datetime.utcnow().isoformat()
    }


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    """Обработка апдейтов от Telegram"""
    data = await request.json()
    update = types.Update(**data)
    await dp.process_update(update)
    return {"ok": True}

# ----------------------------
# Команды бота
# ----------------------------
@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я бот с криптосигналами.\n"
                         "Доступные команды:\n"
                         "/test — проверить соединение\n"
                         "/fast — быстрый курс BTC/ETH\n")


@dp.message_handler(commands=["test"])
async def cmd_test(message: types.Message):
    await message.answer("✅ Бот работает! Webhook активен.")


@dp.message_handler(commands=["fast"])
async def cmd_fast(message: types.Message):
    try:
        btc = yf.Ticker("BTC-USD").history(period="1d")["Close"].iloc[-1]
        eth = yf.Ticker("ETH-USD").history(period="1d")["Close"].iloc[-1]
        await message.answer(f"⚡ Быстрый курс:\n"
                             f"BTC: {btc:.2f} USD\n"
                             f"ETH: {eth:.2f} USD")
    except Exception as e:
        logging.error(f"Ошибка /fast: {e}")
        await message.answer("Ошибка при получении данных с TradingView (yfinance).")

# ----------------------------
# Запуск бота
# ----------------------------
async def on_startup(dp):
    logging.info("Устанавливаю webhook...")
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(dp):
    logging.warning("Выключение бота...")
    await bot.delete_webhook()

def main():
    # aiogram через FastAPI
    import uvicorn
    loop = asyncio.get_event_loop()
    loop.create_task(on_startup(dp))
    uvicorn.run(app, host=WEBAPP_HOST, port=WEBAPP_PORT)

if __name__ == "__main__":
    main()
