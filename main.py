import logging
import os
import pandas as pd
from aiogram import Bot, Dispatcher, executor, types
from tradingview_ta import TA_Handler, Interval

# Логирование
logging.basicConfig(level=logging.INFO)

# Настройки Telegram (токен из переменной окружения)
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("Ошибка: не найден TELEGRAM_API_TOKEN в переменных окружения")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Символы для анализа
SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]

# Поддерживаемые таймфреймы
TIMEFRAMES = {
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
}

# === Получение данных из TradingView ===
def fetch_klines(symbol: str, interval: str):
    try:
        tv_interval = TIMEFRAMES.get(interval, Interval.INTERVAL_1_HOUR)

        handler = TA_Handler(
            symbol=symbol,
            screener="crypto",
            exchange="BINANCE",  # можно заменить на "COINBASE" или "KUCOIN"
            interval=tv_interval,
        )

        analysis = handler.get_analysis()
        indicators = analysis.indicators
        summary = analysis.summary

        # Делаем DataFrame для удобства
        df = pd.DataFrame([indicators])
        df["BUY"] = summary.get("BUY", 0)
        df["SELL"] = summary.get("SELL", 0)
        df["NEUTRAL"] = summary.get("NEUTRAL", 0)

        return df
    except Exception as e:
        logging.error(f"TradingView fetch failed for {symbol} {interval}: {e}")
        raise RuntimeError(f"TradingView fetch failed for {symbol} {interval}: {e}")

# === Формирование отчёта ===
def build_report(symbol: str):
    report = f"📊 Отчёт по {symbol}\n"
    for tf in TIMEFRAMES.keys():
        try:
            df = fetch_klines(symbol, tf)
            if df is not None and not df.empty:
                buy = df['BUY'].iloc[0]
                sell = df['SELL'].iloc[0]
                neutral = df['NEUTRAL'].iloc[0]
                report += f"\n⏱ {tf}:\n✅ BUY: {buy}\n❌ SELL: {sell}\n➖ NEUTRAL: {neutral}\n"
            else:
                report += f"\n⏱ {tf}: Нет данных\n"
        except Exception as e:
            report += f"\n⏱ {tf}: Ошибка ({e})\n"
    return report

# === Telegram-команды ===
@dp.message_handler(commands=["start"])
async def send_welcome(message: types.Message):
    await message.reply("👋 Привет! Я бот для анализа криптовалют с TradingView.\n"
                        "Напиши /report, чтобы получить отчёт.")

@dp.message_handler(commands=["report"])
async def send_report(message: types.Message):
    reply = ""
    for symbol in SYMBOLS:
        reply += build_report(symbol) + "\n\n"
    await message.reply(reply)

# === Запуск бота ===
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
