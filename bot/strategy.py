"""Multi-timeframe trend-filtered channel-breakout strategy.

A higher timeframe (HTF) EMA200 sets the dominant trend. A channel
breakout on the lower timeframe (LTF) is only taken when it agrees with
that trend -- this is the main defense against low-timeframe noise.
Stop-loss / take-profit are sized off LTF ATR(14), not a fixed pip/percent
value, so risk adapts to current volatility.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .channel import BREAKOUT_DOWN, BREAKOUT_UP, is_breakout
from .indicators import atr, ema

TREND_EMA_LENGTH = 200
CHANNEL_WINDOW = 3
CHANNEL_BACKCANDLES = 40
ATR_LENGTH = 14
ATR_SL_MULT = 1.5
REWARD_RISK = 2.0


@dataclass
class Signal:
    direction: str  # "long" or "short"
    entry: float
    stop_loss: float
    take_profit: float
    htf_trend: str
    r_multiple: float
    timestamp: object


def htf_trend(df_htf: pd.DataFrame) -> str:
    """Trend of the last *closed* HTF candle. Caller must have already
    dropped the still-forming candle."""
    trend_ema = ema(df_htf["Close"], TREND_EMA_LENGTH)
    if len(trend_ema) == 0 or pd.isna(trend_ema.iloc[-1]):
        return "unknown"
    return "up" if df_htf["Close"].iloc[-1] > trend_ema.iloc[-1] else "down"


def generate_signal(df_htf: pd.DataFrame, df_ltf: pd.DataFrame) -> Optional[Signal]:
    """Evaluate the strategy on the last *closed* LTF candle.

    Both frames must be sorted ascending by time with OHLC columns and
    have the still-forming candle already dropped by the caller.
    """
    trend = htf_trend(df_htf)
    if trend == "unknown":
        return None

    df_ltf = df_ltf.reset_index(drop=True)
    candle = len(df_ltf) - 1
    if candle - CHANNEL_BACKCANDLES - CHANNEL_WINDOW < 0:
        return None

    breakout = is_breakout(df_ltf, candle, CHANNEL_BACKCANDLES, CHANNEL_WINDOW)
    if breakout == 0:
        return None
    if (breakout == BREAKOUT_UP and trend != "up") or (breakout == BREAKOUT_DOWN and trend != "down"):
        return None  # against the higher-timeframe trend -- skip

    ltf_atr = atr(df_ltf, ATR_LENGTH).iloc[-1]
    if pd.isna(ltf_atr) or ltf_atr <= 0:
        return None

    entry = float(df_ltf["Close"].iloc[-1])
    risk = ATR_SL_MULT * float(ltf_atr)
    if breakout == BREAKOUT_UP:
        direction = "long"
        stop_loss, take_profit = entry - risk, entry + REWARD_RISK * risk
    else:
        direction = "short"
        stop_loss, take_profit = entry + risk, entry - REWARD_RISK * risk

    ts_col = "timestamp" if "timestamp" in df_ltf.columns else df_ltf.columns[0]
    return Signal(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        htf_trend=trend,
        r_multiple=REWARD_RISK,
        timestamp=df_ltf[ts_col].iloc[-1],
    )
