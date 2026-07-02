"""Vectorized backtest of the multi-timeframe channel breakout strategy.

Fetches historical OHLCV from Binance via ccxt for both the LTF and HTF
(or reads a local CSV with --csv for an offline smoke test), aligns the
HTF trend onto each LTF bar *without lookahead* (a HTF candle's trend is
only known once that candle closes), and runs backtesting.Backtest with a
realistic commission.

Example:
    python -m bot.backtest --symbol BTC/USDT --ltf 5m --htf 1h --limit 1000
"""
import argparse

import pandas as pd
from backtesting import Backtest, Strategy

from .channel import BREAKOUT_DOWN, BREAKOUT_UP, is_breakout
from .data import fetch_ohlcv_ccxt, load_ohlcv_csv
from .indicators import atr, ema
from .strategy import ATR_LENGTH, ATR_SL_MULT, CHANNEL_BACKCANDLES, CHANNEL_WINDOW, REWARD_RISK, TREND_EMA_LENGTH


def build_dataset(df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> pd.DataFrame:
    df_htf = df_htf.copy()
    df_htf["trend_ema"] = ema(df_htf["Close"], TREND_EMA_LENGTH)
    df_htf["trend"] = (df_htf["Close"] > df_htf["trend_ema"]).map({True: 1, False: -1})

    # A HTF candle's trend is only known once it closes -- shift it onto the
    # timestamp of the *next* HTF candle so merge_asof can never look ahead.
    trend_known_at = df_htf[["timestamp", "trend"]].copy()
    trend_known_at["timestamp"] = trend_known_at["timestamp"].shift(-1)
    trend_known_at = trend_known_at.dropna(subset=["timestamp"]).sort_values("timestamp")

    df_ltf = df_ltf.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(df_ltf, trend_known_at, on="timestamp", direction="backward")

    breakout = [0] * len(merged)
    for i in range(len(merged)):
        breakout[i] = is_breakout(merged, i, CHANNEL_BACKCANDLES, CHANNEL_WINDOW)
    merged["breakout"] = breakout

    merged["atr"] = atr(merged, ATR_LENGTH)

    merged["signal"] = 0
    long_mask = (merged["breakout"] == BREAKOUT_UP) & (merged["trend"] == 1)
    short_mask = (merged["breakout"] == BREAKOUT_DOWN) & (merged["trend"] == -1)
    merged.loc[long_mask, "signal"] = 1
    merged.loc[short_mask, "signal"] = -1

    merged = merged.dropna(subset=["atr", "trend"]).reset_index(drop=True)
    merged = merged.rename(columns={"timestamp": "Date"}).set_index("Date")
    return merged


class ChannelBreakoutStrategy(Strategy):
    atr_mult = ATR_SL_MULT
    reward_risk = REWARD_RISK

    def init(self):
        self.signal = self.I(lambda: self.data.signal, name="signal")

    def next(self):
        if len(self.trades) > 0:
            return
        risk = self.atr_mult * self.data.atr[-1]
        if risk <= 0:
            return
        if self.data.signal[-1] == 1:
            price = self.data.Close[-1]
            self.buy(sl=price - risk, tp=price + self.reward_risk * risk)
        elif self.data.signal[-1] == -1:
            price = self.data.Close[-1]
            self.sell(sl=price + risk, tp=price - self.reward_risk * risk)


def run_backtest(df_htf: pd.DataFrame, df_ltf: pd.DataFrame, cash: float = 10_000, commission: float = 0.0006):
    data = build_dataset(df_htf, df_ltf)
    bt = Backtest(data, ChannelBreakoutStrategy, cash=cash, commission=commission)
    stats = bt.run()
    return stats, bt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--market", choices=["spot", "futures"], default="spot")
    parser.add_argument("--ltf", default="5m")
    parser.add_argument("--htf", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--commission", type=float, default=0.0006, help="round-trip fee fraction, e.g. 0.0006 = 0.06%")
    parser.add_argument("--csv", help="use a local OHLCV CSV instead of live ccxt data (offline test)")
    parser.add_argument("--notify", action="store_true", help="send the summary to Telegram")
    args = parser.parse_args()

    if args.csv:
        df = load_ohlcv_csv(args.csv)
        df_htf = df_ltf = df
    else:
        import ccxt

        exchange = ccxt.binance({"options": {"defaultType": "future" if args.market == "futures" else "spot"}})
        df_htf = fetch_ohlcv_ccxt(exchange, args.symbol, args.htf, limit=args.limit)
        df_ltf = fetch_ohlcv_ccxt(exchange, args.symbol, args.ltf, limit=args.limit)

    stats, _ = run_backtest(df_htf, df_ltf, commission=args.commission)
    print(stats)

    if args.notify:
        from .telegram import TelegramNotifier

        summary = (
            f"بک‌تست {args.symbol} ({args.ltf}/{args.htf})\n"
            f"معاملات: {stats['# Trades']}\n"
            f"نرخ برد: {stats['Win Rate [%]']:.1f}%\n"
            f"بازده: {stats['Return [%]']:.1f}%\n"
            f"افت سرمایه: {stats['Max. Drawdown [%]']:.1f}%"
        )
        TelegramNotifier().send(summary)


if __name__ == "__main__":
    main()
