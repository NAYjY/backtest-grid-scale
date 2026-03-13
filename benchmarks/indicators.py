def calculate_indicators(df,ATR_LEN_L,ATR_LEN_S,ATR_MULT_L,ATR_MULT_S,ROC_LEN_L,ROC_LEN_S,ROC_THRESH_L,ROC_THRESH_S):       
    df['atr_l'] = ATR_MULT_L * abstract.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_LEN_L)
    df['atr_s'] = ATR_MULT_S * abstract.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_LEN_S)
    df.dropna(subset=['atr_l', 'atr_s'], inplace=True)

    midpoint = (df['high'] + df['low']) / 2
    df['long_stop_l']  = midpoint - df['atr_l']
    df['short_stop_l'] = midpoint + df['atr_l']
    df['long_stop_s']  = midpoint - df['atr_s']
    df['short_stop_s'] = midpoint + df['atr_s']

    roc_l = 100 * (df['close'] - df['close'].shift(ROC_LEN_L)) / df['close'].shift(ROC_LEN_L)
    roc_s = 100 * (df['close'] - df['close'].shift(ROC_LEN_S)) / df['close'].shift(ROC_LEN_S)
    df['ema_roc_l'] = abstract.EMA(np.array(roc_l), timeperiod=ROC_LEN_L // 2)
    df['ema_roc_s'] = abstract.EMA(np.array(roc_s), timeperiod=ROC_LEN_S // 2)

    df['long_stop_prev_l']  = _rolling_max_stop(df['close'].values, df['long_stop_l'].values)
    df['short_stop_prev_l'] = _rolling_min_stop(df['close'].values, df['short_stop_l'].values)
    df['long_stop_prev_s']  = _rolling_max_stop(df['close'].values, df['long_stop_s'].values)
    df['short_stop_prev_s'] = _rolling_min_stop(df['close'].values, df['short_stop_s'].values)

    df['dir_l'] = _calculate_direction(
        df['close'].values, df['short_stop_prev_l'], df['long_stop_prev_l']
    )
    df['dir_s'] = _calculate_direction(
        df['close'].values, df['short_stop_prev_s'], df['long_stop_prev_s']
    )

    roc_strong_l = (df['ema_roc_l'] > ROC_THRESH_L / 2) | (df['ema_roc_l'] < -(ROC_THRESH_L / 2))
    roc_strong_s = (df['ema_roc_s'] > ROC_THRESH_S / 2) | (df['ema_roc_s'] < -(ROC_THRESH_S / 2))

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