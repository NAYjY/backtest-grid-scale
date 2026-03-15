import numpy as np
import pandas as pd
from talib import abstract
from data import StrategyParams

def calculate_indicators(df: pd.DataFrame, 
                         params: StrategyParams) -> pd.DataFrame:
    
    df['atr_l'] = params.atr_mult_l * abstract.ATR(
        df['high'], df['low'], df['close'], timeperiod=params.atr_len_l)
    df['atr_s'] = params.atr_mult_s * abstract.ATR(
        df['high'], df['low'], df['close'], timeperiod=params.atr_len_s)
    df.dropna(subset=['atr_l', 'atr_s'], inplace=True)

    midpoint = (df['high'] + df['low']) / 2
    df['long_stop_l']  = midpoint - df['atr_l']
    df['short_stop_l'] = midpoint + df['atr_l']
    df['long_stop_s']  = midpoint - df['atr_s']
    df['short_stop_s'] = midpoint + df['atr_s']

    roc_l = 100 * (df['close'] - df['close'].shift(params.roc_len_l)) / df['close'].shift(params.roc_len_l)
    roc_s = 100 * (df['close'] - df['close'].shift(params.roc_len_s)) / df['close'].shift(params.roc_len_s)
    df['ema_roc_l'] = abstract.EMA(np.array(roc_l), timeperiod=params.roc_len_l // 2)
    df['ema_roc_s'] = abstract.EMA(np.array(roc_s), timeperiod=params.roc_len_s // 2)

    df['long_stop_prev_l']  = _rolling_max_stop(df['close'].values, df['long_stop_l'].values)
    df['short_stop_prev_l'] = _rolling_min_stop(df['close'].values, df['short_stop_l'].values)
    df['long_stop_prev_s']  = _rolling_max_stop(df['close'].values, df['long_stop_s'].values)
    df['short_stop_prev_s'] = _rolling_min_stop(df['close'].values, df['short_stop_s'].values)

    df['dir_l'] = _calculate_direction(
        df['close'].values, df['short_stop_prev_l'], df['long_stop_prev_l'])
    df['dir_s'] = _calculate_direction(
        df['close'].values, df['short_stop_prev_s'], df['long_stop_prev_s'])

    roc_strong_l = (df['ema_roc_l'] > params.roc_thresh_l / 2) | (df['ema_roc_l'] < -(params.roc_thresh_l / 2))
    roc_strong_s = (df['ema_roc_s'] > params.roc_thresh_s / 2) | (df['ema_roc_s'] < -(params.roc_thresh_s / 2))

    df['long_signal']  = (df['dir_l'] == 1)  & (df['dir_l'].shift(1) == -1) & roc_strong_l
    df['short_signal'] = (df['dir_s'] == -1) & (df['dir_s'].shift(1) == 1)  & roc_strong_s

    df['position'] = 0
    df['position'] = np.where(df['long_signal'],   1, df['position'])
    df['position'] = np.where(df['short_signal'], -1, df['position'])

    return df
# ─── Stops ────────────────────────────────────────────────────────────────────
def _rolling_max_stop(close, stop):
    n = len(close)
    prev = np.zeros(n)
    prev[0] = stop[0]
    for i in range(1, n):
        prev[i] = stop[i - 1]
        if close[i - 1] > prev[i]:
            stop[i] = max(stop[i], prev[i])
    return prev

def _rolling_min_stop(close, stop):
    n = len(close)
    prev = np.zeros(n)
    prev[0] = stop[0]
    for i in range(1, n):
        prev[i] = stop[i - 1]
        if close[i - 1] < prev[i]:
            stop[i] = min(stop[i], prev[i])
    return prev

# ─── Direction ────────────────────────────────────────────────────────────────
def _calculate_direction(close, stop_up, stop_down, initial_dir=1):
    n = len(close)
    out = np.zeros(n)
    direction = initial_dir
    for i in range(n):
        if direction == -1 and close[i] > stop_up[i]:
            direction = 1
        elif direction == 1 and close[i] < stop_down[i]:
            direction = -1
        out[i] = direction
    return out