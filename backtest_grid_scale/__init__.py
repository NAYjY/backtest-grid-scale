"""
backtest_grid_scale
===================
High-performance backtesting framework for parameter grid search on trading
strategies.  Benchmarks a pure-Pandas reference implementation against a
Numba JIT-compiled implementation across a configurable grid of strategy
parameters applied to OHLCV data.
"""

__all__ = [
    "load_config",
    "cfg",
    "OptimizationSpace",
    "get_search_grid",
    "get_sample_grid",
    "StrategyParams",
    "load_ohlcv",
    "calculate_indicators",
]

from .config import load_config, get_search_grid, get_sample_grid, OptimizationSpace, cfg
from .data import StrategyParams, load_ohlcv
from .indicators import calculate_indicators