"""Live low-timeframe alert bot.

Polls Binance for HTF/LTF candles, evaluates the multi-timeframe channel
breakout strategy (bot/strategy.py) on the last *closed* candle, and pushes
a Telegram alert the first time a new signal appears for that candle.

This process needs outbound network access to api.binance.com and
api.telegram.org, so it must run somewhere that has it (your machine, a
VPS, a cron job, etc.) -- not from a network-restricted sandbox.

Examples:
    # single check, good for cron or a manual test
    python -m bot.live_bot --symbol BTC/USDT --ltf 5m --htf 1h --once

    # keep polling every 60s
    python -m bot.live_bot --symbol BTC/USDT --ltf 5m --htf 1h --interval 60
"""
import argparse
import json
import time
from pathlib import Path

from .data import fetch_ohlcv_ccxt
from .strategy import generate_signal
from .telegram import TelegramNotifier

STATE_FILE = Path(__file__).resolve().parent.parent / ".bot_state.json"


def make_exchange(market: str):
    import ccxt

    return ccxt.binance({"options": {"defaultType": "future" if market == "futures" else "spot"}})


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def load_last_alert(symbol: str, ltf: str):
    return _load_state().get(f"{symbol}:{ltf}")


def save_last_alert(symbol: str, ltf: str, timestamp: str):
    state = _load_state()
    state[f"{symbol}:{ltf}"] = timestamp
    STATE_FILE.write_text(json.dumps(state))


def format_message(symbol: str, signal) -> str:
    arrow = "\U0001F7E2 LONG" if signal.direction == "long" else "\U0001F534 SHORT"
    return (
        f"{arrow} {symbol}\n"
        f"زمان کندل: {signal.timestamp}\n"
        f"روند تایم بالا: {signal.htf_trend}\n"
        f"ورود: {signal.entry:.6g}\n"
        f"حد ضرر: {signal.stop_loss:.6g}\n"
        f"حد سود: {signal.take_profit:.6g}\n"
        f"نسبت ریسک به ریوارد: 1:{signal.r_multiple:g}"
    )


def check_once(symbol, ltf, htf, exchange, notifier, limit_ltf=300, limit_htf=300) -> bool:
    df_htf = fetch_ohlcv_ccxt(exchange, symbol, htf, limit=limit_htf).iloc[:-1]
    df_ltf = fetch_ohlcv_ccxt(exchange, symbol, ltf, limit=limit_ltf).iloc[:-1]

    signal = generate_signal(df_htf, df_ltf)
    if signal is None:
        return False

    ts = str(signal.timestamp)
    if load_last_alert(symbol, ltf) == ts:
        return False  # already alerted for this candle

    notifier.send(format_message(symbol, signal))
    save_last_alert(symbol, ltf, ts)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--market", choices=["spot", "futures"], default="spot")
    parser.add_argument("--ltf", default="5m")
    parser.add_argument("--htf", default="1h")
    parser.add_argument("--interval", type=int, default=300, help="seconds between checks when polling")
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    parser.add_argument("--dry-run", action="store_true", help="never call the Telegram API, just print")
    args = parser.parse_args()

    notifier = TelegramNotifier(dry_run=args.dry_run)
    exchange = make_exchange(args.market)

    if args.once:
        fired = check_once(args.symbol, args.ltf, args.htf, exchange, notifier)
        print("signal sent" if fired else "no new signal")
        return

    print(f"polling {args.symbol} every {args.interval}s (ltf={args.ltf}, htf={args.htf})... Ctrl+C to stop")
    while True:
        try:
            check_once(args.symbol, args.ltf, args.htf, exchange, notifier)
        except Exception as exc:  # keep the loop alive across transient errors
            print(f"check failed: {exc}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
