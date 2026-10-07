import os
import json
import asyncio
import logging
from pathlib import Path

import requests
import pandas as pd
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

TOKEN = os.getenv("TELEGRAM_TOKEN")
SYMBOL = "BTCUSDT"
SUBSCRIBERS_FILE = Path("subscribers.json")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

if not TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN environment variable is missing.")


def load_subscribers():
    if not SUBSCRIBERS_FILE.exists():
        return set()
    try:
        return set(json.loads(SUBSCRIBERS_FILE.read_text()))
    except Exception:
        return set()


def save_subscribers(subscribers):
    SUBSCRIBERS_FILE.write_text(json.dumps(sorted(subscribers)))


def get_klines(interval, limit=250):
    url = "https://api.binance.com/api/v3/klines"
    r = requests.get(
        url,
        params={"symbol": SYMBOL, "interval": interval, "limit": limit},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_base", "taker_quote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])
    return df


def indicators(df):
    x = df.copy()

    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["ema200"] = x["close"].ewm(span=200, adjust=False).mean()

    delta = x["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    x["rsi"] = 100 - (100 / (1 + rs))

    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    x["macd_hist"] = x["macd"] - x["macd_signal"]

    prev_close = x["close"].shift(1)
    tr = pd.concat(
        [
            x["high"] - x["low"],
            (x["high"] - prev_close).abs(),
            (x["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    x["atr"] = tr.rolling(14).mean()
    x["vol_ma20"] = x["volume"].rolling(20).mean()
    return x


def build_signal():
    d4 = indicators(get_klines("4h"))
    d1 = indicators(get_klines("1h"))
    d15 = indicators(get_klines("15m"))

    a = d4.iloc[-1]
    b = d1.iloc[-1]
    c = d15.iloc[-1]

    bull4 = a.close > a.ema200 and a.ema50 > a.ema200
    bear4 = a.close < a.ema200 and a.ema50 < a.ema200

    bull1 = b.close > b.ema50 and b.ema20 > b.ema50 and b.rsi >= 50
    bear1 = b.close < b.ema50 and b.ema20 < b.ema50 and b.rsi <= 50

    bull15 = c.close > c.ema20 and c.rsi >= 50 and c.macd_hist > 0
    bear15 = c.close < c.ema20 and c.rsi <= 50 and c.macd_hist < 0

    volume_ok = c.volume >= c.vol_ma20

    if bull4:
        side = "BUY"
        score = 40
        score += 25 if bull1 else 0
        score += 25 if bull15 else 0
        score += 10 if volume_ok else 0
    elif bear4:
        side = "SELL"
        score = 40
        score += 25 if bear1 else 0
        score += 25 if bear15 else 0
        score += 10 if volume_ok else 0
    else:
        side = "WAIT"
        score = 0

    price = float(c.close)
    atr = float(c.atr)

    if side == "BUY" and bull1 and bull15 and atr > 0:
        sl = price - 1.5 * atr
        tp1 = price + 1.0 * (price - sl)
        tp2 = price + 2.0 * (price - sl)
        tp3 = price + 3.0 * (price - sl)
    elif side == "SELL" and bear1 and bear15 and atr > 0:
        sl = price + 1.5 * atr
        tp1 = price - 1.0 * (sl - price)
        tp2 = price - 2.0 * (sl - price)
        tp3 = price - 3.0 * (sl - price)
    else:
        sl = tp1 = tp2 = tp3 = None

    return {
        "side": side,
        "score": int(score),
        "price": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rsi4": float(a.rsi),
        "rsi1": float(b.rsi),
        "rsi15": float(c.rsi),
        "volume_ok": bool(volume_ok),
    }


def format_signal(s):
    if s["side"] == "WAIT" or s["score"] < 70:
        return (
            "🟡 BTC SIGNAL\n\n"
            "وضعیت: ⏳ WAIT\n"
            f"قیمت: ${s['price']:,.2f}\n"
            f"امتیاز: {s['score']}/100\n\n"
            f"RSI 4H: {s['rsi4']:.1f}\n"
            f"RSI 1H: {s['rsi1']:.1f}\n"
            f"RSI 15M: {s['rsi15']:.1f}\n\n"
            "فعلاً شرایط ورود کامل نیست."
        )

    emoji = "🟢" if s["side"] == "BUY" else "🔴"
    return (
        f"{emoji} BTC {s['side']} SIGNAL\n\n"
        f"Entry: ${s['price']:,.2f}\n"
        f"SL: ${s['sl']:,.2f}\n"
        f"TP1: ${s['tp1']:,.2f}\n"
        f"TP2: ${s['tp2']:,.2f}\n"
        f"TP3: ${s['tp3']:,.2f}\n\n"
        f"Score: {s['score']}/100\n"
        f"RSI 4H: {s['rsi4']:.1f}\n"
        f"RSI 1H: {s['rsi1']:.1f}\n"
        f"RSI 15M: {s['rsi15']:.1f}\n"
        f"Volume: {'OK' if s['volume_ok'] else 'Weak'}\n\n"
        "⚠️ این سیگنال آموزشی است؛ سود تضمینی نیست."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Btcai آماده است.\n\n"
        "/signal — دریافت سیگنال BTC\n"
        "/subscribe — دریافت خودکار سیگنال‌ها\n"
        "/unsubscribe — توقف سیگنال خودکار\n"
        "/help — راهنما"
    )


async def signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        s = await asyncio.to_thread(build_signal)
        await update.message.reply_text(format_signal(s))
    except Exception as e:
        logging.exception("Signal error")
        await update.message.reply_text("❌ خطا در دریافت داده بازار. چند لحظه بعد دوباره تلاش کن.")


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers = load_subscribers()
    subscribers.add(update.effective_chat.id)
    save_subscribers(subscribers)
    await update.message.reply_text("✅ فعال شد. سیگنال‌های خودکار برای این چت ارسال می‌شوند.")


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    subscribers = load_subscribers()
    subscribers.discard(update.effective_chat.id)
    save_subscribers(subscribers)
    await update.message.reply_text("⛔ سیگنال خودکار غیرفعال شد.")


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)


async def broadcast_job(context: ContextTypes.DEFAULT_TYPE):
    subscribers = load_subscribers()
    if not subscribers:
        return

    try:
        s = await asyncio.to_thread(build_signal)
        # فقط وقتی شرایط نسبتاً قوی است پیام خودکار بفرست
        if s["side"] == "WAIT" or s["score"] < 70:
            return

        msg = format_signal(s)
        for chat_id in list(subscribers):
            try:
                await context.bot.send_message(chat_id=chat_id, text=msg)
            except Exception:
                logging.exception("Could not send to %s", chat_id)
    except Exception:
        logging.exception("Broadcast error")


def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("subscribe", subscribe))
    app.add_handler(CommandHandler("unsubscribe", unsubscribe))
    app.add_handler(CommandHandler("help", help_cmd))

    # هر 15 دقیقه بررسی می‌کند
    app.job_queue.run_repeating(broadcast_job, interval=900, first=30)

    logging.info("Btcai bot is starting...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
