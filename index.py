import time
import pandas as pd
import pandas_ta as ta
import ccxt



exchange = ccxt.binance()

def get_ohlcv(symbol, interval='1m', limit=10) -> pd.DataFrame:

    # headers
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    # get candles from exchange
    bars = exchange.fetch_ohlcv(symbol, timeframe=interval, limit=limit)

    # convert list of list to pandas dataframe
    df = pd.DataFrame(bars, columns=columns)

    # convert millisecond to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

if __name__ == '__main__':
    symbolinpyt=input('enter symbol:')
    symbol = f'{symbolinpyt.upper()}/USDT'
    quantity = 1.0
    dataframe = get_ohlcv(symbol, limit=50, interval='4h')

    # calculate last row index
    last_row = len(dataframe.index) - 1
    print (last_row)
    dataframe['rsi3'] = ta.rsi(dataframe['close'], length=14) 