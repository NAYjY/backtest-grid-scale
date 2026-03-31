"""
njit_version.py
---------------
Numba JIT-compiled trade simulation and performance screening.

This module implements the same strategy logic as
:mod:`backtest_grid_scale.pandas_version` but uses ``@nb.njit``-decorated
functions for the hot loops, yielding a 4.5–100× speedup on large grids.

Architecture
~~~~~~~~~~~~
The pipeline has three layers:

1. **Python / Pandas** — orchestration, date arithmetic, CSV I/O.
2. **Numba JIT** — :func:`simulate_trades` and :func:`screen_parameter_set`
   operate on raw ``numpy.ndarray`` objects to avoid Python overhead.
3. **DataFrame reconstruction** — :func:`build_trades_df` maps the Numba
   output array back to a time-indexed Pandas DataFrame.

Public functions
~~~~~~~~~~~~~~~~
run_njit_version       -- End-to-end runner: simulate → screen → write CSV.
simulate_trades        -- ``@njit`` bar-by-bar trade simulation.
screen_parameter_set   -- ``@njit`` per-period statistics and pass/fail gate.
build_period_indices   -- Map a DatetimeIndex to integer period boundaries.
build_report_row       -- Assemble a single-row summary DataFrame.
append_to_csv          -- Append a DataFrame row to a CSV file.
run_screening          -- Orchestrate screening for one parameter set.
build_trades_df        -- Reconstruct a DataFrame from the Numba output array.
"""

import os
import numpy as np
import numba as nb
import pandas as pd
import csv
from .data import StrategyParams
from dataclasses import asdict
from .config import load_config
import warnings
warnings.filterwarnings("ignore")


def run_njit_version(
    df: pd.DataFrame,
    params: StrategyParams,
    output_path: str,
    filename: str,
) -> pd.DataFrame:
    """Run a full Numba-accelerated simulation and optionally write results.

    Parameters
    ----------
    df : pandas.DataFrame
        OHLCV + indicator DataFrame produced by
        :func:`~backtest_grid_scale.indicators.calculate_indicators`.
    params : StrategyParams
        Strategy parameters for this grid point.
    output_path : str
        Directory path where the output CSV is written.
    filename : str
        Base filename (without directory) for the output CSV.

    Returns
    -------
    pandas.DataFrame
        Trade-level results DataFrame produced by :func:`build_trades_df`.
    """
    arr = df[['open','high','low','close','position']].values.astype(np.float32)
    trades, n_trades = simulate_trades(arr, params.sl_pct_l, params.sl_pct_s)
    results = build_trades_df(trades, n_trades, df.index)
    #
    cfg = load_config()
    s = pd.to_datetime(cfg['backtest_start'])
    e = pd.to_datetime(cfg['backtest_end'])
    total_type = cfg['total_type']
    condition = cfg['condition']
    #
    run_screening(results[['high','low','open','close','position','pnl']],s,e,total_type,params,condition,output_path,filename)

    return results


# ─── Numba: all math ──────────────────────────────────────────────────────────

@nb.njit
def screen_parameter_set(
    trades:         np.ndarray,
    period_indices: np.ndarray,   # CSV column boundaries — user freq (M/Q/Y)
    thresholds:     np.ndarray,   # [min_pnl, winR, pf, min_n, max_n]
) -> tuple:
    """Compute per-period statistics and apply pass/fail screening.

    This ``@njit`` function operates entirely on raw arrays to avoid Python
    overhead inside tight grid-search loops.

    Parameters
    ----------
    trades : np.ndarray, shape (n_trades, 7+)
        Trade array with at least 7 columns.  Column layout:

        * 0 ``COL_HIGH``     — trade high
        * 1 ``COL_LOW``      — trade low
        * 2 ``COL_OPEN``     — entry price
        * 3 ``COL_CLOSE``    — exit price
        * 4 ``COL_POSITION`` — direction (``1`` long, ``-1`` short)
        * 5 ``COL_PNL``      — trade PnL
        * 6 ``COL_DRAWDOWN`` — computed in-place (initialised to 0)
    period_indices : np.ndarray of int64, shape (n_periods + 1,)
        Row boundaries for each reporting period.  The last element must
        equal ``len(trades)``.
    thresholds : np.ndarray, shape (5,)
        Screening thresholds:
        ``[min_total_pnl, min_win_rate, min_profit_factor, min_trades, max_trades]``.

    Returns
    -------
    tuple
        ``(passed, period_pnl, period_dd, win_rate, profit_factor,
        pnl_long, pnl_short)``

        * *passed*        — ``True`` if all thresholds are met.
        * *period_pnl*    — PnL sum per reporting period.
        * *period_dd*     — worst draw-down per reporting period.
        * *win_rate*      — fraction of winning trades.
        * *profit_factor* — gross profit / |gross loss|.
        * *pnl_long*      — total PnL from long trades.
        * *pnl_short*     — total PnL from short trades.
    """

    # ─── Column indices ────────────────────────────────────────────────────────────
    COL_HIGH      = 0
    COL_LOW       = 1
    COL_OPEN      = 2
    COL_CLOSE     = 3
    COL_POSITION  = 4
    COL_PNL       = 5
    COL_DRAWDOWN  = 6

    n = trades.shape[0]
    wins         = 0
    gross_profit = 0.0
    gross_loss   = 0.0
    pnl_long  = 0.0
    pnl_short = 0.0

    for i in range(1, n):
        if trades[i, COL_PNL] >= 0:
            wins += 1
            gross_profit += trades[i, COL_PNL]
        else:
            gross_loss += trades[i, COL_PNL]

        if trades[i, COL_POSITION] == 1:
            trades[i, COL_DRAWDOWN] = trades[i, COL_LOW] - trades[i, COL_OPEN]
            pnl_long += trades[i, COL_PNL]
        elif trades[i, COL_POSITION] == -1:
            trades[i, COL_DRAWDOWN] = trades[i, COL_OPEN] - trades[i, COL_HIGH]
            pnl_short += trades[i, COL_PNL]

    # ── Per-period stats (user freq — for CSV columns) ────────────────────────
    n_periods    = len(period_indices) - 1
    period_pnl   = np.zeros(n_periods, dtype=np.float64)
    period_dd    = np.zeros(n_periods, dtype=np.float64)

    for p in range(n_periods):
        lo = period_indices[p]
        hi = period_indices[p + 1]

        period_pnl[p] = np.sum(trades[lo:hi, COL_PNL])
        period_dd[p]  = np.min(trades[lo:hi, COL_DRAWDOWN])

    win_rate      = wins / n if n > 0 else 0.0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0.0 else 999.0

    # ── Pass/fail ─────────────────────────────────────────────────────────────
    passed = (
        np.sum(period_pnl)               >= thresholds[0]
        and win_rate                     >= thresholds[1]
        and profit_factor                >= thresholds[2]
        and n                            >= thresholds[3]
        and n                            <= thresholds[4]
    )

    return (
        passed,
        period_pnl, period_dd,
        win_rate, profit_factor, pnl_long, pnl_short
    )


# ─── Python: build period indices ─────────────────────────────────────────────

def build_period_indices(df: pd.DataFrame, freq: str, periods) -> tuple:
    """Map a period-indexed array to integer row-boundary arrays.

    Parameters
    ----------
    df : pandas.DataFrame
        Trade DataFrame whose row positions define the index.
    freq : str
        Reporting frequency: ``'M'`` (monthly), ``'Q'`` (quarterly),
        or ``'Y'`` (yearly).
    periods : pandas.PeriodIndex
        Pre-computed period index matching *df*'s datetime index at *freq*.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        * ``period_indices`` — integer array of shape ``(n_periods + 1,)``;
          the last element is ``len(df)``.
        * ``period_labels`` — list of human-readable column name strings.
    """
    unique  = periods.unique().sort_values()
    labels  = []
    indices = []

    for p in unique:
        rows = np.where(periods == p)[0]
        if len(rows) == 0:
            continue
        indices.append(rows[0])
        if freq == 'M':
            labels.append(f"{p.year}M{p.month}")
        elif freq == 'Q':
            labels.append(f"{p.year}Q{p.quarter}")
        else:
            labels.append(str(p.year))

    indices.append(len(df))
    return np.array(indices, dtype=np.int64), labels


# ─── Python: assemble CSV row ─────────────────────────────────────────────────

def build_report_row(
    params:        dict,
    period_labels: list,
    period_pnl:    np.ndarray,
    period_dd:     np.ndarray,
    total_pnl:     float,
    total_pnl_l:   float,
    total_pnl_s:   float,
    total_dd:      float,
    win_rate:      float,
    profit_factor: float,
    n_trades:      int,
) -> pd.DataFrame:
    """Assemble a single-row summary DataFrame for one parameter set.

    Parameters
    ----------
    params : dict
        Strategy parameter values (from ``dataclasses.asdict``).
    period_labels : list[str]
        Column names for per-period PnL/DD columns.
    period_pnl : np.ndarray
        PnL sum for each reporting period.
    period_dd : np.ndarray
        Worst draw-down for each reporting period.
    total_pnl : float
        Aggregate PnL across all periods.
    total_pnl_l : float
        Aggregate PnL from long trades.
    total_pnl_s : float
        Aggregate PnL from short trades.
    total_dd : float
        Worst draw-down across all periods.
    win_rate : float
        Fraction of winning trades in ``[0, 1]``.
    profit_factor : float
        Gross profit divided by absolute gross loss.
    n_trades : int
        Total number of trades.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame ready to be appended to a results CSV.
    """
    data = asdict(params)
    for i, label in enumerate(period_labels):
        data[label]            = period_pnl[i]
        data[label + '_DD']    = period_dd[i]
    data.update({
        'win_rate':      win_rate,
        'profit_factor': profit_factor,
        'n_trades':      n_trades,
        'Total':         total_pnl,
        'Total_L':       total_pnl_l,
        'Total_S':       total_pnl_s,
        'Total_DD':      total_dd,
    })

    return pd.DataFrame(data, index=[0])


def append_to_csv(row: pd.DataFrame, path: str, filename: str) -> None:
    """Append a single-row DataFrame to a CSV file.

    Creates the file with a header on the first call; subsequent calls append
    without repeating the header.

    Parameters
    ----------
    row : pandas.DataFrame
        Single-row summary DataFrame produced by :func:`build_report_row`.
    path : str
        Directory where the CSV file is located (or will be created).
    filename : str
        Filename within *path*.
    """
    filepath = os.path.join(path, filename)
    header   = not os.path.isfile(filepath)
    row.to_csv(filepath, index=False, header=header, mode='a')


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_screening(
    df_trades:   pd.DataFrame,
    date_start,
    date_end,
    freq:        str,
    params:      StrategyParams,
    thresholds:  np.ndarray,
    output_path: str,
    filename:    str,
    dot:         int = 2,
):
    """Orchestrate the full screening pipeline for one parameter set.

    1. Slice the trade DataFrame to the backtest date range.
    2. Apply a quick PnL pre-filter to skip obviously failing combinations.
    3. Build period indices and call :func:`screen_parameter_set` via Numba.
    4. If the parameter set passes all thresholds, write the summary row to CSV.

    Parameters
    ----------
    df_trades : pandas.DataFrame
        Trade-level DataFrame with columns ``high``, ``low``, ``open``,
        ``close``, ``position``, ``pnl``.
    date_start : datetime-like
        Start of the evaluation window (inclusive).
    date_end : datetime-like
        End of the evaluation window (inclusive).
    freq : str
        Reporting frequency for CSV columns: ``'M'``, ``'Q'``, or ``'Y'``.
    params : StrategyParams
        Strategy parameters written to every CSV row.
    thresholds : np.ndarray, shape (5,)
        Screening thresholds passed through to :func:`screen_parameter_set`.
    output_path : str
        Directory for the output CSV file.
    filename : str
        Base filename for the output CSV.
    dot : int, optional
        Number of decimal places for rounding the trades array.  Default ``2``.
    """
    if df_trades.empty:
        return None, None

    df = df_trades.loc[date_start.strftime('%Y-%m-%d %H:%M:%S'):date_end.strftime('%Y-%m-%d %H:%M:%S')].copy()
    if df.empty or df['pnl'].sum() < thresholds[0]:
        return None, None

    # ── Build array ───────────────────────────────────────────────────────────
    cols = ['high', 'low', 'open', 'close', 'position', 'pnl']
    arr  = df[cols].values.astype(np.float32)
    arr  = np.hstack([arr, np.zeros((len(arr), 1), dtype=np.float32)])  # drawdown column
    arr  = np.round(arr, dot)

    # ── Date components — computed once, used for both F1Y and period indices ─
    year_periods  = pd.to_datetime(df.index).to_period('Y')
    month_periods = pd.to_datetime(df.index).to_period('M')
    qtr_periods   = pd.to_datetime(df.index).to_period('Q')

    # ── Period indices — two calls, two purposes ──────────────────────────────
    period_indices, period_labels = build_period_indices(
        df, freq,
        qtr_periods if freq == 'Q' else month_periods if freq == 'M' else year_periods
    )

    # ── Numba ─────────────────────────────────────────────────────────────────
    (passed,
     period_pnl, period_dd,
     win_rate, profit_factor,
     pnl_long, pnl_short) = screen_parameter_set(
        arr, period_indices, thresholds
    )

    if not passed:
        return None, None

    # ── Report ────────────────────────────────────────────────────────────────
    row = build_report_row(
        params               = params,
        period_labels        = period_labels,
        period_pnl           = period_pnl,
        period_dd            = period_dd,
        total_pnl            = float(period_pnl.sum()),
        total_pnl_l          = pnl_long,
        total_pnl_s          = pnl_short,
        total_dd             = float(period_dd.min()),
        win_rate             = win_rate,
        profit_factor        = profit_factor,
        n_trades             = len(df),
    )

    append_to_csv(row, output_path, filename)


# ─── Trade Simulation ─────────────────────────────────────────────────────────

@nb.njit
def simulate_trades(arr, sl_pct_l, sl_pct_s, max_trades=8000):
    """Simulate bar-by-bar trade execution (Numba JIT).

    Parameters
    ----------
    arr : np.ndarray, shape (n_bars, 5)
        OHLCV bar data with columns:

        * 0 ``IN_OPEN``     — open price
        * 1 ``IN_HIGH``     — high price
        * 2 ``IN_LOW``      — low price
        * 3 ``IN_CLOSE``    — close price
        * 4 ``IN_POSITION`` — entry signal (``1`` long, ``-1`` short, ``0`` flat)
    sl_pct_l : float
        Stop-loss distance as a percentage of entry price for long trades.
    sl_pct_s : float
        Stop-loss distance as a percentage of entry price for short trades.
    max_trades : int, optional
        Pre-allocated number of output rows.  Increase if a grid point can
        produce more than 8 000 trades.  Default ``8000``.

    Returns
    -------
    tuple[np.ndarray, int]
        * ``trades`` — array of shape ``(max_trades, 7)`` with columns
          ``COL_HIGH``, ``COL_LOW``, ``COL_OPEN``, ``COL_CLOSE``,
          ``COL_POSITION``, ``COL_PNL``, ``COL_OPENTRADE``.
        * ``n_trades`` — number of valid (filled) rows in *trades*.
    """
    # ─── Input array columns (OHLCV bar data) ─────────────────────────────────────
    IN_OPEN     = 0
    IN_HIGH     = 1
    IN_LOW      = 2
    IN_CLOSE    = 3
    IN_POSITION = 4

    # ─── Output array columns (trade results → feeds screen_parameter_set) ────────
    COL_HIGH       = 0
    COL_LOW        = 1
    COL_OPEN       = 2
    COL_CLOSE      = 3
    COL_POSITION   = 4
    COL_PNL        = 5
    COL_OPENTRADE  = 6   # integer row index → mapped to timestamp in build_trades_df
    n        = arr.shape[0]
    trades   = np.zeros((max_trades, 7), dtype=np.float64)
    i_row    = 0
    i        = 1

    while i < n:

        # wait for a signal
        if arr[i - 1, IN_POSITION] == 0:
            i += 1
            continue

        entry_idx   = i
        entry_pos   = arr[i - 1, IN_POSITION]
        entry_price = arr[i, IN_OPEN]

        if entry_pos == 1:
            stop_loss = entry_price * (1 - sl_pct_l / 100)
        else:
            stop_loss = entry_price * (1 + sl_pct_s / 100)

        trade_closed = False
        exit_price   = arr[n - 1, IN_CLOSE]   # default: held to end of data

        while not trade_closed and i < n:
            cur_pos = arr[i, IN_POSITION]

            # signal flip — exit at close
            if cur_pos != 0 and cur_pos != entry_pos:
                exit_price   = arr[i, IN_CLOSE]
                trade_closed = True

            # stop loss — long
            elif entry_pos == 1 and arr[i, IN_LOW] <= stop_loss:
                exit_price   = arr[i, IN_OPEN] if arr[i, IN_OPEN] <= stop_loss else stop_loss
                trade_closed = True

            # stop loss — short
            elif entry_pos == -1 and arr[i, IN_HIGH] >= stop_loss:
                exit_price   = arr[i, IN_OPEN] if arr[i, IN_OPEN] >= stop_loss else stop_loss
                trade_closed = True

            i += 1

        trades[i_row, COL_HIGH]      = np.max(arr[entry_idx:i, IN_HIGH])
        trades[i_row, COL_LOW]       = np.min(arr[entry_idx:i, IN_LOW])
        trades[i_row, COL_OPEN]      = entry_price
        trades[i_row, COL_CLOSE]     = exit_price
        trades[i_row, COL_POSITION]  = entry_pos
        trades[i_row, COL_PNL]       = (exit_price - entry_price) * entry_pos
        trades[i_row, COL_OPENTRADE] = entry_idx   # integer — mapped to timestamp later
        i_row += 1

    return trades, i_row


def build_trades_df(trades: np.ndarray, n_trades: int, df_index: pd.Index) -> pd.DataFrame:
    """Reconstruct a labelled DataFrame from the Numba trade output array.

    Parameters
    ----------
    trades : np.ndarray, shape (max_trades, 7)
        Raw output array from :func:`simulate_trades`.
    n_trades : int
        Number of valid rows (as returned alongside *trades*).
    df_index : pandas.Index
        Original bar-data DatetimeIndex used to map integer row positions back
        to timestamps.

    Returns
    -------
    pandas.DataFrame
        One row per trade with columns ``high``, ``low``, ``open``, ``close``,
        ``position``, ``pnl``.  The index is named ``OpenTrade`` and contains
        the entry bar timestamps.
    """
    COL_HIGH       = 0
    COL_LOW        = 1
    COL_OPEN       = 2
    COL_CLOSE      = 3
    COL_POSITION   = 4
    COL_PNL        = 5
    COL_OPENTRADE  = 6

    t = trades[:n_trades]

    results = pd.DataFrame({
        'high':     t[:, COL_HIGH],
        'low':      t[:, COL_LOW],
        'open':     t[:, COL_OPEN],
        'close':    t[:, COL_CLOSE],
        'position': t[:, COL_POSITION].astype(int),
        'pnl':      t[:, COL_PNL],
    }, index=df_index[t[:, COL_OPENTRADE].astype(int)])

    results.index.name = 'OpenTrade'
    return results
