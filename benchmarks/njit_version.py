import os
import numpy as np
import numba as nb
import pandas as pd
import csv
import time
import psutil



s = pd.to_datetime(VIEW_START )
e = pd.to_datetime(VIEW_END)
total_type = 'Y'
sumDR_type = 'M'
condition = [0.0, 100000.0, 0.0, 0.0, 0.0, 0.0, 0.0,1.0,99999.0]


def run_njit_version():
    from benchmarks.indicators import calculate_indicators
    from benchmarks.config import (
        SYMBOL, INTERVAL, DATA_DIR,
        ATR_LEN_L_MAX, ATR_MULT_L_MAX, ATR_LEN_S_MAX, ATR_MULT_S_MAX,
        ROC_LEN_L_MAX, ROC_THRESH_L_MAX, ROC_LEN_S_MAX, ROC_THRESH_S_MAX,
        SL_PCT_L_MAX, SL_PCT_S_MAX,
        ATR_LEN_L_MIN, ATR_MULT_L_MIN, ATR_LEN_S_MIN, ATR_MULT_S_MIN,
        ROC_LEN_L_MIN, ROC_THRESH_L_MIN, ROC_LEN_S_MIN, ROC_THRESH_S_MIN,
        SL_PCT_L_MIN, SL_PCT_S_MIN
    )   
    calculate_indicators = __import__('benchmarks.indicators').calculate_indicators
    simulate_trades = __import__('benchmarks.njit_version').simulate_trades
    build_trades_df = __import__('benchmarks.njit_version').build_trades_df
    run_screening = __import__('benchmarks.njit_version').run_screening
# ─── Numba: all math ──────────────────────────────────────────────────────────

@nb.njit
def screen_parameter_set(
    trades:         np.ndarray,
    period_indices: np.ndarray,   # CSV column boundaries — user freq (M/Q/Y)
    thresholds:     np.ndarray,   # [min_pnl, max_dd, pct_Y, pct_S, pct_Q, winR, pf, min_n, max_n]
) -> tuple:

    # ─── Column indices ────────────────────────────────────────────────────────────
    COL_HIGH      = 0
    COL_LOW       = 1
    COL_OPEN      = 2
    COL_CLOSE     = 3
    COL_POSITION  = 4
    COL_PNL       = 5
    COL_DRAWDOWN  = 6

    n = trades.shape[0]

    # ── sumDR ─────────────────────────────────────────────────────────────────
    trades[0, COL_SUMDR] = trades[0, COL_PNL]
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
        win_rate, profit_factor,pnl_long, pnl_short
    )


# ─── Python: build period indices ─────────────────────────────────────────────

def build_period_indices(df: pd.DataFrame, freq: str, periods) -> tuple:
    """
    Returns (period_indices, period_labels).
    freq: 'M' | 'Q' | 'Y'
    period_indices shape: (n_periods + 1,) — last element is len(df).
    """
    # periods = df.index.to_period(freq)
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
    total_mincm:   float,
    win_rate:      float,
    profit_factor: float,
    n_trades:      int,
) -> pd.DataFrame:
    data = params.copy()
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


def append_to_csv(row: pd.DataFrame, path: str, filename: str):
    filepath = os.path.join(path, filename)
    header   = not os.path.isfile(filepath)
    row.to_csv(filepath, index=False, header=header, mode='a')


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def run_screening(
    df_trades:   pd.DataFrame,
    date_start:  str,
    date_end:    str,
    freq:        str,        # 'M' | 'Q' | 'Y' — CSV column granularity
    params:      dict,       # parameter metadata written into every CSV row
    thresholds:  np.ndarray,
    output_path: str,
    filename:    str,
    dot:         int = 2,
):
    if df_trades.empty:
        return None, None

    df = df_trades.loc[date_start.strftime('%Y-%m-%d %H:%M:%S'):date_end.strftime('%Y-%m-%d %H:%M:%S')].copy()
    if df.empty or df['pnl'].sum() < thresholds[0]:
        return None, None

    

    # ── Build array ───────────────────────────────────────────────────────────
    cols = ['high', 'low', 'open', 'close', 'position', 'pnl', 'sdr_reset']
    arr  = df[cols].values.astype(np.float32)
    arr  = np.hstack([arr, np.zeros((len(arr), 1), dtype=np.float32)])  # sumDR, drawdown
    arr  = np.round(arr, dot)

    '''
    fix this to choose one , cause del SDR
    '''
    # # ── Date components — computed once, used for both F1Y and period indices ─
    # compute once
    year_periods = pd.to_datetime(df.index).to_period('Y')
    month_periods = pd.to_datetime(df.index).to_period('M')
    qtr_periods   = pd.to_datetime(df.index).to_period('Q')

    # ── Period indices — two calls, two purposes ──────────────────────────────
    period_indices, period_labels = build_period_indices(df,freq,qtr_periods if freq=='Q' else month_periods if freq=='M' else year_periods)   # CSV columns

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
    """
    Simulate bar-by-bar trade execution.

    Parameters
    ----------
    arr        : np.ndarray (n_bars, 5) — IN_OPEN/HIGH/LOW/CLOSE/POSITION
    sl_pct_l   : float — stop loss % for long trades
    sl_pct_s   : float — stop loss % for short trades
    max_trades : int   — pre-allocated output rows (use len(arr) // 2)

    Returns
    -------
    trades : np.ndarray (max_trades, 9) — cols defined by COL_* above
    n_trades : int — valid row count
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


def build_trades_df(trades, n_trades, df_index):
    """
    Map numba output array to DataFrame with timestamps.

    Parameters
    ----------
    trades   : np.ndarray — output of simulate_trades
    n_trades : int        — valid row count from simulate_trades
    df_index : pd.Index   — original bar data index (timestamps)
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