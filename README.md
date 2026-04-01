# backtest-grid-scale

[![CI](https://github.com/NAYjY/backtest-grid-scale/actions/workflows/main.yml/badge.svg)](https://github.com/NAYjY/backtest-grid-scale/actions/workflows/main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Production system running since 2022 under contract. Open-sourced as a portfolio reference with proprietary logic removed.

A high-performance backtesting framework for parameter grid search on trading strategies, benchmarking **pure Pandas** against **Numba JIT** implementations.

The Numba path runs **4.5–100×** faster than the Pandas baseline — the bigger the grid, the wider the gap.

---

## Strategy

Implements the open-source [Super Trend Daily 2.0](https://www.tradingview.com/script/1aNKOSH3-Super-Trend-Daily-2-0-Alerts-BF/) by bennef — ported from Pine Script to Python as the benchmark subject. Internal execution and infrastructure are proprietary.

---

## How It Works

The framework runs a full grid search over strategy parameters and compares two execution backends:

| Backend | Description |
|---|---|
| **Pandas** | Pure Python reference implementation |
| **Numba JIT** | Compiled implementation using `@njit` |

Each parameter combination is evaluated across historical OHLCV data, and results are validated for numerical equivalence between both backends before benchmarking.

---

## Installation

**Requirements:** Python 3.10+, TA-Lib
```bash
pip install -e .
```

TA-Lib requires native binaries — see the [TA-Lib installation guide](https://github.com/TA-Lib/ta-lib-python#dependencies) for your platform.

---

## Configuration

Edit `config.yaml` to set your data source, backtest range, and grid search space:
```yaml
symbol:   "S501!"
exchange: "TFEX"
interval: "1h"

data_dir:    "benchmarks/test_data"
output_path: "results/"

backtest_start: "2022-01-01"
backtest_end:   "2026-12-31"

# grid search space (min, max, step)
grid:
  atr_len_l:  { min: 2,   max: 3,   step: 1   }
  atr_mult_l: { min: 0.5, max: 1.0, step: 0.5 }
  roc_len_l:  { min: 10,  max: 11,  step: 5   }
  # ... etc
```

For local overrides create `config.local.yaml` — it takes precedence over `config.yaml` and is gitignored.

---

## Usage

**Validate correctness** (Pandas vs Numba outputs must match):
```bash
pytest benchmarks/test_correctness.py -v
```

**Run full benchmark:**
```bash
python benchmarks/run_benchmark.py
```

Results are saved to `results/performance.csv` with per-combination timing, memory, and CPU metrics.

---

## Performance

| Implementation | Typical Speedup |
|---|---|
| Pandas (baseline) | 1× |
| Numba JIT | **4.5–100×** |

Speedup scales with grid size — the larger the parameter search space, the greater the gain.

---

## Testing
```bash
pytest benchmarks/ -v
```

CI runs the full test suite on Python 3.11 with TA-Lib via GitHub Actions.

---

## Project Structure
```
backtest-grid-scale/
├── backtest_grid_scale/       # Core package — importable by external projects
│   ├── config.py              # Grid search space + parameter builder
│   ├── data.py                # OHLCV loader + StrategyParams
│   ├── indicators.py          # SuperTrend + ROC indicators
│   ├── pandas_version.py      # Reference implementation
│   └── njit_version.py        # Numba JIT implementation
├── benchmarks/
│   ├── run_benchmark.py       # Benchmark runner
│   └── test_correctness.py    # Pandas vs Numba validation
├── config.yaml                # Runtime config + grid parameters
├── pyproject.toml
└── results/                   # Benchmark output (generated)
```

---

## License

Strategy logic adapted from [Super Trend Daily 2.0 Alerts BF](https://www.tradingview.com/script/1aNKOSH3-Super-Trend-Daily-2-0-Alerts-BF/) by bennef (open-source, TradingView House Rules).