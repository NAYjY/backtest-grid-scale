"""
test_correctness.py
-------------------
Correctness test suite for the ``backtest_grid_scale`` package.

Test phases
~~~~~~~~~~~
1. **CI sanity** — trivial assertion to confirm the test runner is working.
2. **Smoke tests** — verify that data loading, the Pandas implementation, and
   the Numba JIT implementation all run without errors.
3. **Numerical equivalence** — assert that the Pandas and Numba backends
   produce identical trade counts and PnL figures (rounded to 2 d.p.).
4. **TradingView validation** — (skipped) compare against a TradingView export
   once the reference CSV is available.

Run with::

    pytest benchmarks/test_correctness.py -v
"""

import os
import pytest
import numpy as np
import pandas as pd

from backtest_grid_scale.config import load_config, get_sample_grid

cfg = load_config()


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def real_df():
    """Load the test OHLCV CSV configured in ``config.yaml``."""
    from backtest_grid_scale.data import load_ohlcv
    return load_ohlcv(
        symbol   = [cfg['symbol'], cfg['exchange'], 1],
        interval = cfg['interval'],
        data_dir = cfg['data_dir']
    )


@pytest.fixture
def mock_params():
    """Return a fixed :class:`StrategyParams` instance for deterministic tests."""
    from backtest_grid_scale.data import StrategyParams
    return StrategyParams(
        atr_len_l=3,    atr_len_s=4,
        atr_mult_l=1.5, atr_mult_s=1.5,
        roc_len_l=30,   roc_len_s=76,
        roc_thresh_l=6, roc_thresh_s=6,
        sl_pct_l=5.0,   sl_pct_s=6.0,
    )


# ─── Phase 1: CI sanity ───────────────────────────────────────────────────────

def test_ci_is_working():
    """Trivial assertion — confirms the test runner is operational."""
    assert 1 + 1 == 2


# ─── Phase 2: versions run without error ─────────────────────────────────────

def test_data_loads(real_df):
    """OHLCV DataFrame loads with correct shape and OHLC ordering."""
    assert len(real_df) > 0
    assert all(col in real_df.columns
               for col in ['open', 'high', 'low', 'close'])
    assert (real_df['high'] >= real_df['close']).all()
    assert (real_df['low']  <= real_df['close']).all()


def test_pandas_version_runs(real_df, mock_params):
    """Pandas simulation completes and returns trade-level results."""
    from backtest_grid_scale.indicators import calculate_indicators
    from backtest_grid_scale.pandas_version import simulate_trades, add_trade_stats
    df      = calculate_indicators(real_df.copy(), mock_params)
    results = simulate_trades(df, mock_params.sl_pct_l, mock_params.sl_pct_s)
    results = add_trade_stats(results)
    assert len(results) > 0
    assert 'pnl' in results.columns
    assert 'DrawDown' in results.columns


def test_njit_version_runs(real_df, mock_params):
    """Numba JIT simulation completes and returns trade-level results."""
    from backtest_grid_scale.indicators import calculate_indicators
    from backtest_grid_scale.njit_version import simulate_trades, build_trades_df
    df        = calculate_indicators(real_df.copy(), mock_params)
    arr       = df[['open','high','low','close','position']].values.astype(np.float32)
    trades, n = simulate_trades(arr, mock_params.sl_pct_l, mock_params.sl_pct_s)
    results   = build_trades_df(trades, n, df.index)
    assert len(results) > 0
    assert 'pnl' in results.columns


# ─── Phase 3: correctness ─────────────────────────────────────────────────────

def test_pandas_equals_njit(real_df, mock_params):
    """Pandas and Numba backends produce numerically identical results."""
    from backtest_grid_scale.indicators import calculate_indicators
    from backtest_grid_scale.pandas_version import simulate_trades as pandas_sim, add_trade_stats
    from backtest_grid_scale.njit_version import simulate_trades as njit_sim, build_trades_df

    df            = calculate_indicators(real_df.copy(), mock_params)
    pandas_result = add_trade_stats(pandas_sim(df, mock_params.sl_pct_l, mock_params.sl_pct_s))

    arr           = df[['open','high','low','close','position']].values.astype(np.float32)
    trades, n     = njit_sim(arr, mock_params.sl_pct_l, mock_params.sl_pct_s)
    njit_result   = build_trades_df(trades, n, df.index)

    assert len(pandas_result) == len(njit_result), \
        f"trade count differs: pandas={len(pandas_result)} njit={len(njit_result)}"
    assert round(pandas_result.pnl.sum(), 2) == round(njit_result.pnl.sum(), 2), \
        f"total pnl differs: pandas={pandas_result.pnl.sum():.2f} njit={njit_result.pnl.sum():.2f}"
    assert round(pandas_result.pnl.max(), 2) == round(njit_result.pnl.max(), 2), \
        "best trade differs"
    assert round(pandas_result.pnl.min(), 2) == round(njit_result.pnl.min(), 2), \
        "worst trade differs"


# ─── Phase 4: TradingView validation ─────────────────────────────────────────

@pytest.mark.skip(reason="TradingView export not ready")
def test_matches_tradingview(real_df, mock_params):
    """Compare simulation output against a TradingView reference export."""
    from backtest_grid_scale.indicators import calculate_indicators
    from backtest_grid_scale.njit_version import simulate_trades, build_trades_df

    tv        = pd.read_csv('benchmarks/test_data/tv_reference.csv')
    df        = calculate_indicators(real_df.copy(), mock_params)
    arr       = df[['open','high','low','close','position']].values.astype(np.float32)
    trades, n = simulate_trades(arr, mock_params.sl_pct_l, mock_params.sl_pct_s)
    results   = build_trades_df(trades, n, df.index)

    assert round(results.pnl.sum(), 2) == round(tv['pnl'].sum(), 2), \
        f"total pnl differs: ours={results.pnl.sum():.2f} tv={tv['pnl'].sum():.2f}"
    assert len(results) == len(tv), \
        f"trade count differs: ours={len(results)} tv={len(tv)}"
