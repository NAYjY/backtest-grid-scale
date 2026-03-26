# backtest-grid-scale

Backtesting framework comparing pure Pandas vs Numba JIT implementations on trading strategy parameter grid search.

## Installation

```bash
pip install -e .
```

Requires Python 3.11+, TA-Lib, and dependencies listed in `pyproject.toml`.

## Quick Start

```bash
pytest benchmarks/test_correctness.py -v      # Validate correctness
python benchmarks/run_benchmark.py             # Run performance benchmark
```

Results saved to `fast_performance.csv`.

## Overview

- Grid searches strategy parameters (ATR, ROC, stop-loss %) across trading data
- Two implementations: Pandas (reference) + Numba JIT (optimized)
- Validates both produce identical results
- Measures performance (memory, CPU, execution time)
- Typical speedup: **10–100x** with Numba

## Configuration

Set data path, symbol, and parameter ranges in `config.yaml`. 

## Testing

```bash
pytest benchmarks/ -v
```

CI runs full test suite on Python 3.11 with TA-Lib via GitHub Actions.
