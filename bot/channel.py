"""Pivot / channel-breakout detection.

Adapted from Straategy.Two/ChannelBreakOut.ipynb: instead of the original
O(n * window) python loop, pivots are found with a vectorized rolling
max/min (same semantics -- a candle is a pivot high/low if its high/low
is the extreme within the surrounding `window` candles on both sides).

All functions expect a DataFrame with a plain RangeIndex (0..n-1) and
'High'/'Low'/'Open'/'Close' columns, sorted ascending by time.
"""
import numpy as np
import pandas as pd
from scipy import stats

PIVOT_HIGH = 1
PIVOT_LOW = 2
PIVOT_BOTH = 3

BREAKOUT_NONE = 0
BREAKOUT_UP = 1
BREAKOUT_DOWN = -1


def find_pivots(df: pd.DataFrame, window: int = 3) -> pd.Series:
    high, low = df["High"], df["Low"]
    span = 2 * window + 1
    roll_max = high.rolling(span, center=True).max()
    roll_min = low.rolling(span, center=True).min()
    is_high = high >= roll_max
    is_low = low <= roll_min

    pivot = pd.Series(0, index=df.index, dtype=int)
    pivot[is_high & is_low] = PIVOT_BOTH
    pivot[is_high & ~is_low] = PIVOT_HIGH
    pivot[~is_high & is_low] = PIVOT_LOW
    return pivot


def collect_channel(df: pd.DataFrame, candle: int, backcandles: int, window: int):
    """Fit trend lines through the pivot highs/lows in the lookback window
    ending `window` candles before `candle`. Returns
    (slope_lows, intercept_lows, slope_highs, intercept_highs, r2_lows, r2_highs).
    """
    start = candle - backcandles - window
    end = candle - window
    if start < 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    localdf = df.iloc[start:end].copy()
    localdf["pivot"] = find_pivots(localdf, window)
    highs = localdf.loc[localdf["pivot"] == PIVOT_HIGH, "High"]
    lows = localdf.loc[localdf["pivot"] == PIVOT_LOW, "Low"]

    if len(lows) < 2 or len(highs) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    sl_lows, ic_lows, r_l, _, _ = stats.linregress(lows.index.values, lows.values)
    sl_highs, ic_highs, r_h, _, _ = stats.linregress(highs.index.values, highs.values)
    return sl_lows, ic_lows, sl_highs, ic_highs, r_l**2, r_h**2


def is_breakout(df: pd.DataFrame, candle: int, backcandles: int, window: int) -> int:
    """1 if candle broke down through the lower channel line, -1 if it broke
    up through the upper channel line, 0 otherwise. Only looks at candles
    up to and including `candle` -- safe to call on the last closed bar.
    """
    if candle < 1 or candle - backcandles - window < 0:
        return BREAKOUT_NONE

    sl_lows, ic_lows, sl_highs, ic_highs, _, _ = collect_channel(df, candle, backcandles, window)
    if sl_lows == 0 and sl_highs == 0:
        return BREAKOUT_NONE

    prev, curr = df.iloc[candle - 1], df.iloc[candle]
    prev_idx, curr_idx = candle - 1, candle

    lower_prev, lower_curr = sl_lows * prev_idx + ic_lows, sl_lows * curr_idx + ic_lows
    upper_prev, upper_curr = sl_highs * prev_idx + ic_highs, sl_highs * curr_idx + ic_highs

    if prev.High > lower_prev and prev.Close < lower_prev and curr.Open < lower_curr and curr.Close < lower_prev:
        return BREAKOUT_DOWN
    if prev.Low < upper_prev and prev.Close > upper_prev and curr.Open > upper_curr and curr.Close > upper_prev:
        return BREAKOUT_UP
    return BREAKOUT_NONE
