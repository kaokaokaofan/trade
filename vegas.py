import pandas as pd
import numpy as np
import talib as ta
import vectorbt as vbt
import matplotlib.pyplot as plt
from datetime import datetime

import os
from binance.client import Client
api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')
client = Client(api_key, api_secret)

# ====  設定起始與結束時間（2025年5月～7月） ====
start_str = '2024-07-26 00:00:00'
end_str = '2025-07-27 23:59:59'
start_ts = int(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
end_ts = int(datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
bars = client.get_historical_klines('BTCUSDT', '5m', start_ts, end_ts)

with open('5min.csv', 'w', newline='') as f:
    for line in bars:
        del line[6:]
    df = pd.DataFrame(bars, columns=['date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['seconds'] = np.floor_divide(df['date'], 1000)
    df['date'] = df['seconds'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))
    df.set_index('date', inplace=True)
    print(df.head()) ##確認資料是否正確
    df.to_csv('5min.csv')

df = pd.read_csv('5min.csv', index_col=0)

##轉資料為浮點數
df['Open'] = df['Open'].astype(float)
df['High'] = df['High'].astype(float)
df['Low'] = df['Low'].astype(float)
df['Close'] = df['Close'].astype(float)
df['Volume'] = df['Volume'].astype(float)

def crossover(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))

def crossdown(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))

# 定義Vegas通道

df = df[['High', 'Low', 'Open', 'Close', 'Volume']]

df['ema12'] = ta.EMA(df.Close, timeperiod = 12)
df['ema144'] = ta.EMA(df.Close, timeperiod = 144)
df['ema169'] = ta.EMA(df.Close, timeperiod = 169)
df['ema576'] = ta.EMA(df.Close, timeperiod = 576)
df['ema676'] = ta.EMA(df.Close, timeperiod = 676)

# 定義動向指標(DMI)
len = 14
lensig = 14

# Calculate +DM and -DM
df['up'] = df.High.diff()
df['down'] = -df.Low.diff()
df['upmove'] = np.where((df.up > df.down) & (df.up > 0), df.up, 0)
df['downmove'] = np.where((df.down > df.up) & (df.down > 0), df.down, 0)

# Calculate +DI and -DI
df['plusDI'] = 100 * ta.EMA(df.upmove / ta.ATR(df.High, df.Low, df.Close, timeperiod=len), timeperiod=len)
df['minusDI'] = 100 * ta.EMA(df.downmove / ta.ATR(df.High, df.Low, df.Close, timeperiod=len), timeperiod=len)

# Calculate ADX
df['adx'] = 100 * ta.EMA(np.abs(df.plusDI - df.minusDI) / (df.plusDI + df.minusDI), timeperiod=lensig)

# 定義ATR為止損點
df['atr'] = ta.ATR(df.High, df.Low, df.Close, timeperiod=14)
# 定義多空排列
df['vegasLong'] = np.minimum(df['ema144'], df['ema169']) > np.maximum(df['ema576'], df['ema676'])
df['vegasShort'] = np.minimum(df['ema144'], df['ema169']) < np.maximum(df['ema576'], df['ema676'])

df = df[['Open', 'Close', 'ema12', 'ema144', 'ema169', 'ema576', 'ema676', 'atr', 'vegasLong', 'vegasShort', 'High', 'Low','plusDI', 'minusDI', 'adx']]

# 過濾需要的資料
# row + columns
df = df.iloc[700: , : ]
# df.head(20) 

# 定義交易訊號
# 一倍ATR設置停損點
signal = pd.DataFrame()

signal['long_entry'] = (df['vegasLong']) & crossover(df.Close, df.ema144 ) & (df['ema12'] > df['ema144'])  & (df.plusDI > df.minusDI) 
signal['short_entry'] = (df['vegasShort']) & crossover(df.Close, df.ema144 ) & (df['ema12'] < df['ema144'])  & (df.plusDI < df.minusDI) 
signal['long_exit'] =  (df.adx < 20 & (df['ema12'] < df['ema144'])) 
signal['short_exit'] = (df.adx < 20 & (df['ema12'] > df['ema144'])) 

# 通常會以收盤價作為進出場的起始K棒的價格，而開單則應該要是下一根K棒的開盤價
price = df['Open'].shift(-1)

# 回測
pf = vbt.Portfolio.from_signals(price,
                            entries = signal['long_entry'],
                            exits = signal['long_exit'],
                            short_entries = signal['short_entry'],
                            short_exits = signal['short_exit'],
                            freq = '5m',
                            fees = 0.0005)

# 統計結果
print(pf.stats())