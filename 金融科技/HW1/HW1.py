import pandas as pd
import matplotlib.pyplot as plt


# 讀入你已過濾好的檔案
df_raw = pd.read_csv("TX.csv", encoding="utf-8-sig")

# 只取四欄並改英文欄名
rename_map = {
    "成交日期": "Date",
    "成交時間": "Time",
    "成交價格": "Price",
    "成交數量(B+S)": "Volume",
}
cols = [c for c in rename_map if c in df_raw.columns]
data = df_raw[cols].rename(columns=rename_map).copy()

# 正規化日期/時間
def norm_date(s):
    s = str(s)
    if "-" in s or "/" in s:      # 已經是 2025-09-17 或 2025/09/17
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    return pd.to_datetime(s, format="%Y%m%d").strftime("%Y-%m-%d")

def norm_time(s):
    s = str(s)
    if ":" in s:                   # 已經是 11:17:43
        return pd.to_datetime(s).strftime("%H:%M:%S")
    s = "".join(ch for ch in s if ch.isdigit()).zfill(6)[:6]  # 111743/90301 -> 09:03:01
    return f"{s[:2]}:{s[2:4]}:{s[4:6]}"

data["Date"] = data["Date"].map(norm_date)
data["Time"] = data["Time"].map(norm_time)

# 建立時間戳並排序
data["DateTime"] = pd.to_datetime(data["Date"] + " " + data["Time"])
data = data.sort_values("DateTime").reset_index(drop=True)

# 轉數值型態
data["Price"]  = pd.to_numeric(data["Price"], errors="coerce")
data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0).astype(int)

"""
# ✅ 印幾筆資料檢查
print("欄位名稱：", list(data.columns))
print("\n前 5 筆資料：")
print(data.head(5))
print("\n資料筆數：", len(data))
print("\n時間範圍：", data["DateTime"].min(), "→", data["DateTime"].max())
"""

# (a) Time Bar: 以1天為單位的K線 (每日的Open, High, Low, Close)
bars_time = []
for date, grp in data.groupby(data['DateTime'].dt.date):
    open_price = grp.iloc[0]['Price']
    close_price = grp.iloc[-1]['Price']
    high_price = grp['Price'].max()
    low_price = grp['Price'].min()
    ticks_count = len(grp)
    last_dt = grp.iloc[-1]['DateTime']  # 取該日最後一筆交易時間作為該K棒的時間戳
    bars_time.append({
        'timestamp': last_dt,
        'open': open_price,
        'close': close_price,
        'high': high_price,
        'low': low_price,
        'ticks': ticks_count
    })
bars_time_df = pd.DataFrame(bars_time).set_index('timestamp')

"""
print("✅ bars_time_df 資料筆數：", len(bars_time_df))
print("✅ 欄位：", list(bars_time_df.columns))
print("\n📊 前5筆資料預覽：")
print(bars_time_df.head())

print("\n📊 各欄位資料型態：")
print(bars_time_df.dtypes)
"""


# (b) Tick Bar: 每 10,000 筆交易畫一根 K 棒（不分日）
bars_tick = []
tick_threshold = 10000

# 確保資料按時間排序
data = data.sort_values('DateTime').reset_index(drop=True)
n = len(data)

# 依序每一萬筆一組切分
for start in range(0, n, tick_threshold):
    end = min(start + tick_threshold - 1, n - 1)
    subset = data.iloc[start:end+1]
    bars_tick.append({
        'timestamp': subset.iloc[-1]['DateTime'],   # 該組最後一筆交易時間
        'open': subset.iloc[0]['Price'],            # 第一筆成交價
        'close': subset.iloc[-1]['Price'],          # 最後一筆成交價
        'high': subset['Price'].max(),              # 最高價
        'low': subset['Price'].min(),               # 最低價
        'ticks': len(subset)                        # 該組筆數（通常為 10,000）
    })

# 建立 DataFrame 並設定時間索引
bars_tick_df = pd.DataFrame(bars_tick).set_index('timestamp')

# (c) Volume Bar: 每累積 100,000 交易量畫一根K棒（跨日連續累積）
vol_threshold = 100_000

# 只取需要欄位、轉型、排序
_df = data[['DateTime', 'Price', 'Volume']].copy()
_df['DateTime'] = pd.to_datetime(_df['DateTime'])
_df['Price']    = pd.to_numeric(_df['Price'], errors='coerce')
_df['Volume']   = pd.to_numeric(_df['Volume'], errors='coerce')
_df = _df.dropna(subset=['DateTime','Price','Volume']).sort_values('DateTime').reset_index(drop=True)

# 連續累積成交量 → 以門檻分桶（最後一桶可能不足門檻也會保留）
cvol = _df['Volume'].cumsum()
gid  = ((cvol - 1) // vol_threshold).astype('int64')   # 每達門檻切一組
_df['_gid'] = gid

# 聚合成每棒 OHLC / ticks / volume，時間戳取該組最後一筆
bars_volume_df = (
    _df.groupby('_gid', sort=True)
       .agg(
           open     = ('Price', 'first'),
           high     = ('Price', 'max'),
           low      = ('Price', 'min'),
           close    = ('Price', 'last'),
           ticks    = ('Price', 'size'),
           volume   = ('Volume', 'sum'),
           timestamp= ('DateTime', 'max')
       )
       .set_index('timestamp')
       .sort_index()
)

# (d) Dollar Bar: 每累積 10 億新台幣成交金額畫一根K棒
# 計算成交金額時需考慮TX契約每點新台幣200元的乘數
bars_dollar = []
dollar_threshold = 1e9  # 10億
for date, grp in data.groupby(data['DateTime'].dt.date):
    cumulative_value = 0.0
    start_idx = 0
    for j, row in grp.reset_index(drop=True).iterrows():
        trade_value = row['Price'] * row['Volume'] * 200.0  # 該筆交易的成交金額
        cumulative_value += trade_value
        if cumulative_value >= dollar_threshold:
            subset = grp.iloc[start_idx:j+1]
            bars_dollar.append({
                'timestamp': subset.iloc[-1]['DateTime'],
                'open': subset.iloc[0]['Price'],
                'close': subset.iloc[-1]['Price'],
                'high': subset['Price'].max(),
                'low': subset['Price'].min(),
                'ticks': len(subset)
            })
            start_idx = j + 1
            cumulative_value = 0.0
    if start_idx < len(grp):
        subset = grp.iloc[start_idx:]
        bars_dollar.append({
            'timestamp': subset.iloc[-1]['DateTime'],
            'open': subset.iloc[0]['Price'],
            'close': subset.iloc[-1]['Price'],
            'high': subset['Price'].max(),
            'low': subset['Price'].min(),
            'ticks': len(subset)
        })
bars_dollar_df = pd.DataFrame(bars_dollar).set_index('timestamp')

# 5. 定義一個函式來繪製 K-Bar 圖（包括上方K線及下方對應的交易次數柱狀圖）
def plot_candlestick(bars_df, title, use_time_axis=False):
    """
    繪製 K-Bar（上方K線 + 下方 tick 柱）。預設使用等距索引軸，避免黏在一起與時間空白。
    - use_time_axis=False：等距索引軸（推薦用於 Tick / Volume / Dollar bars）
    - use_time_axis=True ：時間軸（Time Bar 用，會保留真實時間間隔）
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- 準備資料 ---
    df = bars_df.copy()
    # 必要欄位檢查與數值化
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"缺少欄位：{c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ticks" not in df.columns:
        df["ticks"] = np.nan
    else:
        df["ticks"] = pd.to_numeric(df["ticks"], errors="coerce")

    # X 軸：等距或時間
    if use_time_axis:
        if not isinstance(df.index, (pd.DatetimeIndex, pd.core.indexes.datetimes.DatetimeIndex)):
            raise ValueError("use_time_axis=True 時，index 必須是 DatetimeIndex")
        df = df.sort_index()
        x = df.index

        # 動態棒寬：取最小時間差的 80%，並做上下限夾住，避免過寬/過細
        if len(x) > 1:
            td = pd.Series(x).diff().dropna()
            min_diff = td.min()
            mean_diff = td.mean()
            if pd.isna(min_diff) or min_diff == pd.Timedelta(0):
                bar_td = pd.Timedelta(minutes=1)
            else:
                bar_td = max(pd.Timedelta(minutes=1),
                             min(min_diff * 0.8, mean_diff * 0.9))
        else:
            bar_td = pd.Timedelta(minutes=10)
        width = bar_td / pd.Timedelta(days=1)  # matplotlib 的日單位寬度
        wick_lw = 0.8
    else:
        # 等距索引軸：沒有空白、不會黏在一起
        df = df.reset_index(drop=False)
        x = np.arange(len(df))
        width = 0.8         # 棒寬
        wick_lw = 0.8

    # 顏色：紅漲綠跌
    up_mask = df["close"] >= df["open"]
    up_color, down_color = "red", "green"
    body_colors = np.where(up_mask, up_color, down_color)

    # --- 畫圖 ---
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, sharex=True, figsize=(11, 6),
        gridspec_kw={'height_ratios':[3,1]}
    )

    # 先畫實體（不會被影線或邊框蓋掉）
    heights = df["close"] - df["open"]
    ax1.bar(x[up_mask], heights[up_mask], width=width,
            bottom=df.loc[up_mask, "open"],
            color=up_color, edgecolor=up_color, linewidth=0.6)
    ax1.bar(x[~up_mask], heights[~up_mask], width=width,
            bottom=df.loc[~up_mask, "open"],
            color=down_color, edgecolor=down_color, linewidth=0.6)

    # 再畫影線（高低）
    ax1.vlines(x, df["low"], df["high"], colors=body_colors, linewidth=wick_lw)

    ax1.set_title(title)
    ax1.set_ylabel("Price")

    # 下方 tick（若缺就顯示 0，不畫也可）
    if df["ticks"].notna().any():
        ax2.bar(x, df["ticks"].fillna(0), width=width,
                color=body_colors, edgecolor=body_colors, linewidth=0.4)
        ax2.set_ylabel("Number of Ticks")
    else:
        ax2.text(0.5, 0.5, "No 'ticks' column", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=10, alpha=0.7)
        ax2.set_yticks([])

    ax2.set_xlabel("Time" if use_time_axis else "Bar Index")

    # X 標籤
    if use_time_axis:
        plt.setp(ax1.get_xticklabels(), rotation=45)
        plt.setp(ax2.get_xticklabels(), rotation=45)
    else:
        # 等距軸：顯示對應日期/時間的標籤（抽樣顯示避免擁擠）
        if isinstance(bars_df.index, (pd.DatetimeIndex, pd.core.indexes.datetimes.DatetimeIndex)):
            labels = bars_df.index.strftime("%Y-%m-%d").tolist()
        else:
            labels = [str(i) for i in range(len(df))]
        # 抽樣顯示（最多 10 個標籤）
        step = max(1, len(labels)//10)
        ax2.set_xticks(np.arange(0, len(df), step))
        ax2.set_xticklabels(labels[::step], rotation=45, ha="right")

    plt.tight_layout()
    fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()

# 分別繪製四種 K-Bar 圖表
plot_candlestick(bars_time_df, "Time Bar (1 Day)")
plot_candlestick(bars_tick_df, "Tick Bar (10000 ticks)")
plot_candlestick(bars_volume_df, "Volume Bar (100000 volume)")
plot_candlestick(bars_dollar_df, "Dollar Bar (1B NTD)")
