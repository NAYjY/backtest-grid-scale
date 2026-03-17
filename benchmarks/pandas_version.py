import numpy as np
import pandas as pd
import os
from data import StrategyParams
from dataclasses import asdict
from config import load_config

def run_pandas_version(
    df: pd.DataFrame,
    params: StrategyParams,
    output_path: str,
    filename: str,
) -> pd.DataFrame:


    results  = simulate_trades(df, params.sl_pct_l, params.sl_pct_s)
    results  = add_trade_stats(results)
    summary  = generate_performance_report(results, params)

    out = output_path + 'pandas_' + filename
    summary.to_csv(out, mode='a', index=False, 
                   header=not os.path.exists(out))

    return results


def generate_performance_report(df_Y: pd.DataFrame,
                                params: StrategyParams)-> pd.DataFrame:
    cfg = load_config()
    months   = pd.date_range(start=cfg['backtest_start'], 
                          end=cfg['backtest_end'], freq='MS')
    quarters = pd.date_range(start=cfg['backtest_start'], 
                            end=cfg['backtest_end'], freq='QS')
    years    = pd.date_range(start=cfg['backtest_start'], 
                            end=cfg['backtest_end'], freq='YS')
    data = asdict(params)

    # QUARTERLY SLICING
    for i in range(len(quarters)-1):
        start, end = quarters[i].strftime('%Y-%m-%d'), quarters[i+1].strftime('%Y-%m-%d')
        q_name = f"{quarters[i].year}Q{pd.Period(quarters[i], freq='Q').quarter}"
        data[q_name] = df_Y.loc[start:end].pnl.sum()

    # YEARLY SLICING & RISK
    for i in range(len(years)-1):
        start, end = years[i].strftime('%Y-%m-%d'), years[i+1].strftime('%Y-%m-%d')
        y_name = f"{years[i].year}"
        subset = df_Y.loc[start:end]
        data[y_name] = subset.pnl.sum()

    # MONTHLY SLICING
    for i in range(len(months)-1):
        start, end = months[i].strftime('%Y-%m-%d'), months[i+1].strftime('%Y-%m-%d')
        m_name = f"{months[i].year}M{pd.Period(months[i], freq='M').month}"
        subset = df_Y.loc[start:end]
        data[m_name] = subset.pnl.sum()


    # TOTALS
    total_range = df_Y.loc[cfg['backtest_start']:cfg['backtest_end']]
    data['Total'] = total_range.pnl.sum()
    data['Total_DD'] = total_range.DrawDown.min()

    new_df = pd.DataFrame(data, index=[0])

    return new_df

# ─── Trade Simulation ─────────────────────────────────────────────────────────

def simulate_trades(df,SL_PCT_L,SL_PCT_S):
    records = []
    i = 1
    n = len(df)

    while i < n:
        if df['position'].iloc[i - 1] == 0:
            i += 1
            continue

        entry_idx = i
        entry_pos = df['position'].iloc[i - 1]
        entry_price = df['open'].iloc[i]

        if entry_pos == 1:
            stop_loss = entry_price * (1 - SL_PCT_L / 100)
        else:
            stop_loss = entry_price * (1 + SL_PCT_S / 100)

        trade_closed = False
        exit_price = None
        exit_reason = None

        while not trade_closed and i < n:
            current_close = df['close'].iloc[i]
            current_pos   = df['position'].iloc[i]

            # Signal flip
            if current_pos != 0 and current_pos != entry_pos:
                exit_price  = current_close
                exit_reason = 'change'
                trade_closed = True

            # Stop loss — long
            elif entry_pos == 1 and df['low'].iloc[i] <= stop_loss:
                exit_price  = df['open'].iloc[i] if df['open'].iloc[i] <= stop_loss else stop_loss
                exit_reason = 'STP'
                trade_closed = True

            # Stop loss — short
            elif entry_pos == -1 and df['high'].iloc[i] >= stop_loss:
                exit_price  = df['open'].iloc[i] if df['open'].iloc[i] >= stop_loss else stop_loss
                exit_reason = 'STP'
                trade_closed = True

            i += 1

        if exit_price is None:
            exit_price = df['close'].iloc[i - 1]
            exit_reason = 'end'

        pnl = (exit_price - entry_price) * entry_pos

        trade_slice = df.iloc[entry_idx:i]
        records.append({
            'OpenTrade':  df.index[entry_idx],
            'CloseTrade': df.index[i - 1],
            'position':   entry_pos,
            'open':       entry_price,
            'close':      exit_price,
            'pnl':        pnl,
            'stop_loss':  stop_loss,
            'high':       trade_slice['high'].max(),
            'low':        trade_slice['low'].min(),
            'exit_reason': exit_reason,
        })

    results = pd.DataFrame(records)
    if results.empty:
        return results

    results.index = results['OpenTrade']
    results.drop(columns=['OpenTrade'], inplace=True)
    return results

def add_trade_stats(result):
    # TIME PERIOD LABELS FOR GROUPING
    result.index = pd.to_datetime(result.index)
    YEAR = result.index.strftime('%Y').astype(int)

    # BASE TRADE METRICS (DRAWDOWN AND RUNUP)
    result['DrawDown'] = np.where(
        result['position'] == 1,
        result['low'] - result['open'],
        result['open'] - result['high']
    )
    result['Runup'] = np.where(
        result['position'] == 1,
        result['high'] - result['open'],
        result['open'] - result['low']
    )
    result['cumulative_pnl'] = result['pnl'].cumsum()
    result['yearly_pnl_cumsum'] = result.groupby(YEAR)['pnl'].cumsum()
    result['position_label'] = result['position'].map({1: 'L', -1: 'S', 0: 'X'})
    
    return result
