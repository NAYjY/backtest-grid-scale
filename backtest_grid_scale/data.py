"""
data.py
-------
Core data structures and OHLCV data loading.

Provides
~~~~~~~~
* :class:`StrategyParams` — typed dataclass holding one complete set of
  strategy parameters for a single backtest run.
* :func:`load_ohlcv` — read a symbol's OHLCV CSV from disk into a
  time-indexed ``pandas.DataFrame``.
"""

import pandas as pd

from dataclasses import dataclass


@dataclass
class StrategyParams:
    """One concrete parameter combination for the SuperTrend + ROC strategy.

    Instances are produced by :func:`~backtest_grid_scale.config.get_search_grid`
    and :func:`~backtest_grid_scale.config.get_sample_grid`, then passed to
    :func:`~backtest_grid_scale.indicators.calculate_indicators` and the
    simulation functions.

    Attributes
    ----------
    atr_len_l : int
        ATR look-back period for the *long* SuperTrend band.
    atr_len_s : int
        ATR look-back period for the *short* SuperTrend band.
    atr_mult_l : float
        ATR multiplier for the *long* SuperTrend band.
    atr_mult_s : float
        ATR multiplier for the *short* SuperTrend band.
    roc_len_l : int
        ROC look-back period for the *long* momentum filter.
    roc_len_s : int
        ROC look-back period for the *short* momentum filter.
    roc_thresh_l : float
        Minimum absolute EMA(ROC) required to confirm a *long* signal.
    roc_thresh_s : float
        Minimum absolute EMA(ROC) required to confirm a *short* signal.
    sl_pct_l : float
        Stop-loss distance as a percentage of entry price for *long* trades.
    sl_pct_s : float
        Stop-loss distance as a percentage of entry price for *short* trades.
    """

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
    """Load OHLCV bar data from a CSV file.

    The expected filename format is ``{exchange}_{ticker}_{interval}.csv``
    where *symbol* is a two-element sequence ``[ticker, exchange]``.

    Parameters
    ----------
    symbol : sequence of str
        ``[ticker, exchange]`` — e.g. ``["S501!", "TFEX"]``.
    interval : str
        Bar interval string — e.g. ``"1h"``.
    data_dir : str
        Directory that contains the CSV file.

    Returns
    -------
    pandas.DataFrame
        OHLCV DataFrame with a ``DatetimeIndex`` named ``"time"`` and at
        least the columns ``open``, ``high``, ``low``, ``close``.
    """
    path = f'{data_dir}/{symbol[1]}_{symbol[0]}_{interval}.csv'
    df = pd.read_csv(path, index_col='time')
    df.index = pd.to_datetime(df.index)
    return df