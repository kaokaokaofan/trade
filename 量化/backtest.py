# backtest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SECONDS_PER_YEAR = 365 * 24 * 3600

def _build_signal(long_entry: pd.Series,
                  long_exit: pd.Series,
                  short_entry: pd.Series,
                  short_exit: pd.Series) -> pd.Series:
    """用狀態機把四個布林訊號轉成部位 (-1/0/+1)，並 shift(1) 下一根生效。"""
    le = long_entry.fillna(False).astype(bool).to_numpy()
    lx = long_exit.fillna(False).astype(bool).to_numpy()
    se = short_entry.fillna(False).astype(bool).to_numpy()
    sx = short_exit.fillna(False).astype(bool).to_numpy()

    n = len(le)
    pos = np.zeros(n, dtype=float)  # -1/0/+1

    for i in range(1, n):
        prev = pos[i-1]
        if prev == 0:
            if le[i] and not se[i]:
                pos[i] = 1.0
            elif se[i] and not le[i]:
                pos[i] = -1.0
            else:
                pos[i] = 0.0
        elif prev == 1:  # 多單中
            if se[i]:         # 反手優先（兩筆：平多+開空）
                pos[i] = -1.0
            elif lx[i]:
                pos[i] = 0.0
            else:
                pos[i] = 1.0
        else:            # 空單中
            if le[i]:         # 反手優先（兩筆：平空+開多）
                pos[i] = 1.0
            elif sx[i]:
                pos[i] = 0.0
            else:
                pos[i] = -1.0

    signal = pd.Series(pos, index=long_entry.index).shift(1).fillna(0.0)
    return signal

def backtest(df: pd.DataFrame,
             long_entry: pd.Series,
             long_exit: pd.Series,
             short_entry: pd.Series,
             short_exit: pd.Series,
             fee_per_side: float = 0.0005,
             step_sec: float = 15*60.0):
    """
    以 close-to-close 回測：
    - df 需至少含 'Close'
    - 四個訊號為布林 Series，index 與 df 對齊
    - fee_per_side：單邊費用（比例）
    - step_sec：一根K線秒數（年化用）
    回傳：(df_bt, stats)
    """
    df = df.copy()
    # 產生部位 signal（已 shift(1)）
    df['signal'] = _build_signal(long_entry, long_exit, short_entry, short_exit)

    # 報酬 & 費用（反手自動=2×單邊；只有 signal 變化才扣）
    df['ret'] = df['Close'].pct_change().fillna(0.0)
    pos_change = (df['signal'] - df['signal'].shift(1).fillna(0)).abs()
    fees = fee_per_side * pos_change

    df['pos_change'] = pos_change
    df['fees'] = fees
    df['strat_ret_gross'] = df['signal'] * df['ret']
    df['strat_ret'] = df['strat_ret_gross'] - fees

    # 權益曲線
    df['equity'] = (1 + df['strat_ret']).cumprod()
    df['bh'] = (1 + df['ret']).cumprod()

    # ===== 指標 =====
    if len(df.index) > 1:
        years = max((df.index[-1] - df.index[0]).total_seconds() / SECONDS_PER_YEAR, 1e-9)
    else:
        years = 1e-9
    periods_per_year = SECONDS_PER_YEAR / step_sec

    total_return = float(df['equity'].iloc[-1] - 1)
    cagr = float(df['equity'].iloc[-1] ** (1 / years) - 1)
    std = float(df['strat_ret'].std())
    sharpe = (df['strat_ret'].mean() / std * np.sqrt(periods_per_year)) if std > 0 else np.nan
    roll_max = df['equity'].cummax()
    max_dd = float((df['equity'] / roll_max - 1).min())

    # 交易統計：反手算 2（pos_change=2）
    num_orders = int(df['pos_change'].sum())
    num_round_trips = num_orders // 2  # 開+平=一回（若最後留倉，可能有 ±1 誤差）

    # ===== 逐筆交易勝率（含費用） =====
    # 將開倉費與平倉費分別歸屬到該筆交易：
    # 規則：在變化那一根先處理「關舊倉扣一次費 → 結算 trade」，
    # 再「新倉開倉先扣一次費」，最後用當根的新持倉去累積當根報酬。
    sig = df['signal']
    trade_returns = []
    prev_pos = 0.0
    cum_ret = 0.0

    for i in range(len(df)):
        cur_pos = float(sig.iloc[i])

        # 若部位改變，先關舊倉（若有），扣平倉費並結算；再開新倉（若有），先扣開倉費
        if cur_pos != prev_pos:
            if prev_pos != 0.0:
                # 關舊倉：扣一次費用
                cum_ret -= fee_per_side
                trade_returns.append(cum_ret)
                cum_ret = 0.0
            if cur_pos != 0.0:
                # 開新倉：先扣一次費用
                cum_ret -= fee_per_side

        # 用當根「生效後的」持倉（cur_pos）累積當根報酬
        if cur_pos != 0.0:
            cum_ret += cur_pos * float(df['ret'].iloc[i])

        prev_pos = cur_pos

    # 若最後仍有留倉，不把未平倉計入勝率（避免偏誤）
    win_rate_trades = np.mean(np.array(trade_returns) > 0) if len(trade_returns) > 0 else np.nan

    stats = dict(
        periods=len(df),
        step_minutes=step_sec/60.0,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        std=std,
        max_dd=max_dd,
        win_rate=win_rate_trades,      # ← 已改：逐筆交易勝率
        num_orders=num_orders,
        num_round_trips=num_round_trips,
        fee_per_side=fee_per_side,
    )
    return df, stats

def plot_equity(df_bt: pd.DataFrame, title="Strategy vs Buy & Hold"):
    plt.figure(figsize=(11,5))
    plt.plot(df_bt['equity'], label='Strategy')
    plt.plot(df_bt['bh'], label='Buy & Hold', alpha=0.7)
    plt.xlabel('Time'); plt.ylabel('Equity'); plt.title(title)
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def report(stats: dict, name: str = "回測報告") -> None:
    """列印回測報告（逐筆交易勝率版）。"""
    def pct(x): return f"{x*100:.2f}%"
    print(f"====== {name} ======")
    print(f"資料根數: {stats['periods']:,} | 估計週期: {stats['step_minutes']:.2f} 分/根")
    print(f"交易次數: {stats['num_round_trips']}")
    print(f"總報酬率: {pct(stats['total_return'])}")
    print(f"CAGR年化: {pct(stats['cagr'])}")
    print(f"Sharpe  : {stats['sharpe']:.2f}")
    print(f"最大回撤: {pct(stats['max_dd'])}")
    print(f"勝率: {pct(stats['win_rate'])}")
    print(f"單邊費用 : {stats['fee_per_side']*100:.3f}%")
