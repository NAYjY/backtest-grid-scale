import pandas as pd

from dataclasses import dataclass

@dataclass
class StrategyParams:
    atr_len_l:   int
    atr_len_s:   int
    atr_mult_l:  float
    atr_mult_s:  float
    roc_len_l:   int
    roc_len_s:   int
    roc_thresh_l: float
    roc_thresh_s: float
    sl_pct_l:    float
    sl_pct_s:    float

def load_ohlcv(symbol, interval, data_dir: str) -> pd.DataFrame:
    path = f'{data_dir}/{symbol[1]}_{symbol[0]}_{interval}.csv'
    df = pd.read_csv(path, index_col='time')
    df.index = pd.to_datetime(df.index)
    return df