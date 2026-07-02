"""OHLCV data sources: live via ccxt, or a local CSV for offline testing."""
import pandas as pd

COLUMNS = ["timestamp", "Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv_ccxt(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV candles from a ccxt exchange instance. The most recent
    candle returned by the exchange is usually still forming -- callers
    that need only closed candles should drop the last row."""
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """Load a CSV with Gmt time/Date/Open/High/Low/Close[/Volume] columns
    (matches the format already used under Straategy.Two/) into the
    standard OHLCV frame used by the rest of bot/."""
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in ("gmt time", "date", "time", "timestamp"):
            rename[col] = "timestamp"
    df = df.rename(columns=rename)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[COLUMNS]
