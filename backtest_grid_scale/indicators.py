"""
indicators.py
-------------
SuperTrend and ROC indicator computation.

The indicators are ported from the open-source Pine Script strategy
`Super Trend Daily 2.0 <https://www.tradingview.com/script/1aNKOSH3>`_
by bennef.  Each call to :func:`calculate_indicators` is stateless — it
receives a *copy* of the raw OHLCV DataFrame and returns an enriched
DataFrame with all indicator columns attached.

Column glossary
~~~~~~~~~~~~~~~
``atr_l / atr_s``
    Scaled ATR values for the long / short bands.
``long_stop_l / short_stop_l``
    Raw (pre-trail) lower / upper stop levels using the *long* ATR.
``long_stop_s / short_stop_s``
    Raw (pre-trail) lower / upper stop levels using the *short* ATR.
``long_stop_prev_l / short_stop_prev_l``
    Trailing stop levels for the *long-band* SuperTrend.
``long_stop_prev_s / short_stop_prev_s``
    Trailing stop levels for the *short-band* SuperTrend.
``dir_l / dir_s``
    Trend direction signals: ``1`` = uptrend, ``-1`` = downtrend.
``ema_roc_l / ema_roc_s``
    Smoothed rate-of-change used as a momentum filter.
``long_signal / short_signal``
    Boolean entry signals (direction flip + ROC filter confirmed).
``position``
    Signed position: ``1`` = long, ``-1`` = short, ``0`` = flat.
"""

import numpy as np
import pandas as pd
from talib import abstract
from .data import StrategyParams


def calculate_indicators(df: pd.DataFrame,
                         params: StrategyParams) -> pd.DataFrame:
    """Compute all SuperTrend + ROC indicators and entry signals.

    Modifies *df* in-place and returns it.  The caller should pass a copy
    (``df_raw.copy()``) when the original must be preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV DataFrame with columns ``open``, ``high``, ``low``, ``close``.
    params : StrategyParams
        Strategy parameters for this particular grid point.

    Returns
    -------
    pandas.DataFrame
        The same *df* with indicator and signal columns appended.  Rows
        where ATR cannot be computed (initial look-back) are dropped.
    """

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
        df['close'].values, df['short_stop_prev_l'].values, df['long_stop_prev_l'].values)
    df['dir_s'] = _calculate_direction(
        df['close'].values, df['short_stop_prev_s'].values, df['long_stop_prev_s'].values)

    roc_strong_l = (df['ema_roc_l'] > params.roc_thresh_l / 2) | (df['ema_roc_l'] < -(params.roc_thresh_l / 2))
    roc_strong_s = (df['ema_roc_s'] > params.roc_thresh_s / 2) | (df['ema_roc_s'] < -(params.roc_thresh_s / 2))

    df['long_signal']  = (df['dir_l'] == 1)  & (df['dir_l'].shift(1) == -1) & roc_strong_l
    df['short_signal'] = (df['dir_s'] == -1) & (df['dir_s'].shift(1) == 1)  & roc_strong_s

    df['position'] = 0
    df['position'] = np.where(df['long_signal'],   1, df['position'])
    df['position'] = np.where(df['short_signal'], -1, df['position'])

    return df


# ─── Stops ────────────────────────────────────────────────────────────────────

def _rolling_max_stop(close: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Compute trailing *support* stop levels (ratchets upward only).

    For each bar the trailing stop is the maximum of the raw stop value and
    the previous bar's stop, provided the previous close was **above** the
    previous trailing stop (i.e. the uptrend is intact).

    Parameters
    ----------
    close : np.ndarray
        Close prices, shape ``(n,)``.
    stop : np.ndarray
        Raw (untrailed) lower stop values, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Trailing stop values aligned to the same bar indices, shape ``(n,)``.
    """
    stop = stop.copy()
    n = len(close)
    prev = np.zeros(n)
    prev[0] = stop[0]
    for i in range(1, n):
        prev[i] = stop[i - 1]
        if close[i - 1] > prev[i]:
            stop[i] = max(stop[i], prev[i])
    return prev


def _rolling_min_stop(close: np.ndarray, stop: np.ndarray) -> np.ndarray:
    """Compute trailing *resistance* stop levels (ratchets downward only).

    For each bar the trailing stop is the minimum of the raw stop value and
    the previous bar's stop, provided the previous close was **below** the
    previous trailing stop (i.e. the downtrend is intact).

    Parameters
    ----------
    close : np.ndarray
        Close prices, shape ``(n,)``.
    stop : np.ndarray
        Raw (untrailed) upper stop values, shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Trailing stop values aligned to the same bar indices, shape ``(n,)``.
    """
    stop = stop.copy()
    n = len(close)
    prev = np.zeros(n)
    prev[0] = stop[0]
    for i in range(1, n):
        prev[i] = stop[i - 1]
        if close[i - 1] < prev[i]:
            stop[i] = min(stop[i], prev[i])
    return prev


# ─── Direction ────────────────────────────────────────────────────────────────

def _calculate_direction(
    close: np.ndarray,
    stop_up: np.ndarray,
    stop_down: np.ndarray,
    initial_dir: int = 1,
) -> np.ndarray:
    """Determine bar-by-bar SuperTrend direction from trailing stops.

    Direction flips from ``-1`` to ``1`` when close crosses above *stop_up*,
    and from ``1`` to ``-1`` when close crosses below *stop_down*.

    Parameters
    ----------
    close : np.ndarray
        Close prices, shape ``(n,)``.
    stop_up : np.ndarray
        Trailing resistance stop (output of :func:`_rolling_min_stop`),
        shape ``(n,)``.
    stop_down : np.ndarray
        Trailing support stop (output of :func:`_rolling_max_stop`),
        shape ``(n,)``.
    initial_dir : int, optional
        Starting direction before the first bar is processed.  Default ``1``.

    Returns
    -------
    np.ndarray
        Direction array of ``1`` (uptrend) and ``-1`` (downtrend),
        shape ``(n,)``.
    """
    close = close.copy()
    stop_up = stop_up.copy()
    stop_down = stop_down.copy()
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