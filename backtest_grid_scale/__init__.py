"""
backtest_grid_scale
===================
High-performance backtesting framework for parameter grid search on trading
strategies.  Benchmarks a pure-Pandas reference implementation against a
Numba JIT-compiled implementation across a configurable grid of strategy
parameters applied to OHLCV data.

Public API
----------
load_config       -- Load ``config.yaml`` (or ``config.local.yaml``) into a dict.
cfg               -- Module-level config dict loaded at import time.
OptimizationSpace -- Dataclass describing the parameter search space.
get_search_grid   -- Enumerate every combination in the search space.
get_sample_grid   -- Draw a random sample from the full grid.
StrategyParams    -- Dataclass holding one concrete parameter combination.
load_ohlcv        -- Read OHLCV data from a CSV file into a DataFrame.
calculate_indicators -- Compute SuperTrend + ROC indicators on OHLCV data.
"""

from .config import load_config, get_search_grid, get_sample_grid, OptimizationSpace, cfg
from .data import StrategyParams, load_ohlcv
from .indicators import calculate_indicators