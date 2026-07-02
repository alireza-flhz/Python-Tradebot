"""Offline smoke test for the bot/ package -- no network access required.

Generates synthetic 5m OHLCV, resamples it to 1h for the HTF trend filter,
then exercises the three pieces that matter most:

1. generate_signal() called incrementally the way live_bot.py would call
   it, to make sure the live code path never crashes.
2. The Telegram message formatter, via a dry-run notifier (no network).
3. The full vectorized backtest (bot/backtest.py), including the
   lookahead-safe HTF-trend-onto-LTF alignment.

Run with: python3 scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from bot.backtest import run_backtest
from bot.live_bot import format_message
from bot.strategy import generate_signal
from bot.telegram import TelegramNotifier


def synth_ohlcv(n=4000, start="2024-01-01", freq="5min", seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=n, freq=freq)
    # random walk with a slow drift so trends actually form
    drift = np.linspace(0, 0.35, n)
    noise = rng.normal(0, 0.006, n).cumsum()
    close = 100 * np.exp(drift + noise)
    high = close * (1 + rng.uniform(0, 0.003, n))
    low = close * (1 - rng.uniform(0, 0.003, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"timestamp": ts, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}
    )


def resample_htf(df_ltf: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    htf = (
        df_ltf.set_index("timestamp")
        .resample(rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
        .reset_index()
    )
    return htf


def main():
    df_ltf = synth_ohlcv()
    df_htf = resample_htf(df_ltf)
    print(f"synthetic data: {len(df_ltf)} LTF (5m) candles, {len(df_htf)} HTF (1h) candles")

    # 1) live-style incremental signal generation must not crash and must
    #    never look at candles past the current point in time.
    signals = []
    for i in range(400, len(df_ltf), 25):
        window_ltf = df_ltf.iloc[:i]
        cutoff = window_ltf["timestamp"].iloc[-1]
        window_htf = df_htf[df_htf["timestamp"] <= cutoff]
        if len(window_htf) < 5:
            continue
        sig = generate_signal(window_htf, window_ltf)
        if sig is not None:
            signals.append(sig)
    print(f"generate_signal: ok, {len(signals)} synthetic signals found across the incremental scan")
    assert all(s.direction in ("long", "short") for s in signals)
    assert all(s.stop_loss != s.entry for s in signals)

    # 2) Telegram formatting + dry-run send (no network call made)
    notifier = TelegramNotifier(dry_run=True)
    if signals:
        ok = notifier.send(format_message("BTC/USDT", signals[0]))
        assert ok
    print("telegram dry-run: ok")

    # 3) full vectorized backtest must run end-to-end without exceptions
    stats, _ = run_backtest(df_htf, df_ltf, commission=0.0006)
    print(stats[["# Trades", "Win Rate [%]", "Return [%]", "Max. Drawdown [%]"]])
    assert stats["# Trades"] >= 0
    print("backtest: ok")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
