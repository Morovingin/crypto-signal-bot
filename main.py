import os
import logging
import datetime
import io
import httpx
import pandas as pd
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# === ENV VARIABLES ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# === Init ===
bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()
scheduler = AsyncIOScheduler()

# === Telegram bot with handlers ===
tg_app = Application.builder().token(TELEGRAM_TOKEN).build()

# === SYMBOLS TO TRACK ===
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"]

# ========================
#   FETCH DATA FROM BYBIT
# ========================
async def fetch_ohlcv(symbol: str, interval: str, limit: int = 200):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": str(limit)
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("retCode") != 0:
                logging.error(f"Bybit API error: {data}")
                return None

            rows = data["result"]["list"]
            df = pd.DataFrame(rows, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df = df.astype({
                "open": "float",
                "high": "float",
                "low": "float",
                "close": "float",
                "volume": "float"
            })
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            return df
    except Exception as e:
        logging.error(f"fetch_ohlcv error for {symbol} {interval}: {e}")
        return None

# ========================
#   INDICATORS
# ========================
def compute_indicators(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    try:
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["MA50"] = df["close"].rolling(window=50).mean()

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    except Exception as e:
        logging.error(f"compute_indicators error: {e}")
        return None

# ========================
#   REPORT BUILDER
# ========================
async def build_report(symbol: str, interval: str):
    df = await fetch_ohlcv(symbol, interval, 200)
    df = compute_indicators(df)
    if df is None or df.empty:
        return f"{symbol} — error computing data", None

    last = df.iloc[-1]
    text = (
        f"📊 Report for {symbol} ({interval})\n"
        f"Close: {last['close']:.4f}\n"
        f"MA20: {last['MA20']:.4f}\n"
        f"MA50: {last['MA50']:.4f}\n"
        f"RSI: {last['RSI']:.2f}\n"
    )

    # === chart ===
    fig, ax = plt.subplots(figsize=(8, 4))
    df["close"].plot(ax=ax, label="Close", color="blue")
    df["MA20"].plot(ax=ax, label="MA20", color="orange")
    df["MA50"].plot(ax=ax, label="MA50", color="red")
    ax.set_title(f"{symbol} {interval} chart")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return text, buf

# ========================
#   TASKS
# ========================
async def hourly_task():
    logging.info("Running hourly task...")
    for sym in SYMBOLS:
        try:
            text, img = await build_report(sym, "60")
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
            if img:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img)
        except Exception as e:
            logging.error(f"hourly_task error {sym}: {e}")
    logging.info("Hourly task finished")

async def daily_task():
    logging.info("Running daily task...")
    for sym in SYMBOLS:
        try:
            text, img = await build_report(sym, "D")
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
            if img:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img)
        except Exception as e:
            logging.error(f"daily_task error {sym}: {e}")
    logging.info("Daily task finished")

# ========================
#   TEST ENDPOINT (BROWSER)
# ========================
@app.get("/test")
async def test_report():
    text, img = await build_report("BTCUSDT", "60")
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    if img:
        await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img)
    return {"status": "test report sent"}

# ========================
#   TELEGRAM COMMAND /test
# ========================
async def command_test(update: Update, context):
    text, img = await build_report("BTCUSDT", "60")
    await update.message.reply_text(text)
    if img:
        await update.message.reply_photo(img)

tg_app.add_handler(CommandHandler("test", command_test))

# ========================
#   SCHEDULER
# ========================
@app.on_event("startup")
async def startup_event():
    scheduler.add_job(hourly_task, "cron", minute=1)
    scheduler.add_job(daily_task, "cron", hour=0, minute=5)
    scheduler.start()
    tg_app.initialize()
    tg_app.post_init()
    logging.info("Scheduler + Telegram bot started")

# ========================
#   ROOT (UPTIME ROBOT)
# ========================
@app.get("/")
async def root():
    return JSONResponse({
        "status": "ok",
        "time": datetime.datetime.utcnow().isoformat()
    })

# ========================
#   RUN SERVER
# ========================
if __name__ == "__main__":
    import uvicorn, asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(tg_app.run_polling())
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
