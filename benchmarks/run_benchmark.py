"""
run_benchmark.py
----------------
End-to-end benchmark runner comparing the Pandas and Numba JIT backends.

Usage
~~~~~
::

    python benchmarks/run_benchmark.py

The script:

1. Loads OHLCV data and a random sample of the parameter grid from
   ``config.yaml``.
2. Runs the **Pandas** implementation over every sampled parameter set while
   recording RAM, CPU, and wall-clock time via :class:`PerformanceMonitor`.
3. Repeats step 2 with the **Numba JIT** implementation.
4. Writes ``performance.csv`` to the working directory with one row per
   checkpoint.

Results (trade-level CSVs) are written to ``results/`` as configured in
``config.yaml``.
"""

import os
import time
import csv
import psutil
import numpy as np
import pandas as pd

from backtest_grid_scale.config import get_sample_grid, cfg
from backtest_grid_scale.data import load_ohlcv, StrategyParams
from backtest_grid_scale.indicators import calculate_indicators
from backtest_grid_scale.pandas_version import run_pandas_version
from backtest_grid_scale.njit_version import run_njit_version


# ─── Performance Monitor ──────────────────────────────────────────────────────

class PerformanceMonitor:
    """Lightweight resource-usage recorder.

    Writes a CSV file with one row per :meth:`record` call, capturing
    wall-clock elapsed time, process RSS memory, and CPU utilisation.

    Parameters
    ----------
    filename : str, optional
        Path to the output CSV file.  Defaults to ``"performance.csv"``.
    """

    def __init__(self, filename: str = "performance.csv") -> None:
        self.filename = filename
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        with open(self.filename, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'phase', 'ram_mb',
                'process_cpu_pct', 'system_cpu_pct', 'elapsed_sec'
            ])

    def record(self, phase_name: str) -> None:
        """Append a single resource-usage row to the CSV.

        Parameters
        ----------
        phase_name : str
            Human-readable label for this checkpoint (e.g. ``"Start Pandas"``).
        """
        ram_mb      = self.process.memory_info().rss / (1024 * 1024)
        process_cpu = self.process.cpu_percent(interval=0.1)  # my code only
        system_cpu  = psutil.cpu_percent(interval=0.1)        # whole machine
        elapsed     = time.time() - self.start_time

        with open(self.filename, 'a', newline='') as f:
            csv.writer(f).writerow([
                time.strftime("%H:%M:%S"), phase_name,
                round(ram_mb, 2),
                round(process_cpu, 1),
                round(system_cpu, 1),
                round(elapsed, 2)
            ])

        print(f"[{phase_name}] "
              f"RAM: {ram_mb:.0f}MB | "
              f"Process CPU: {process_cpu:.1f}% | "
              f"System CPU: {system_cpu:.1f}% | "
              f"{elapsed:.1f}s")


if __name__ == "__main__":
    symbol     = [cfg['symbol'], cfg['exchange'], 1]
    interval   = cfg['interval']
    data_dir   = cfg['data_dir']
    output_path = cfg['output_path']
    filename   = f'SuperDaily_{symbol[0]}_{interval}.csv'

    df_raw = load_ohlcv(symbol, interval, data_dir)

    len_grid, grid = get_sample_grid()
    monitor = PerformanceMonitor()

    # ── Pandas pass ──────────────────────────────────────────────────────────
    monitor.record("Start Pandas")
    for params in grid:
        df = calculate_indicators(df_raw.copy(), params)
        run_pandas_version(
            df=df,
            params=params,
            output_path=output_path,
            filename=filename,
        )
    monitor.record("End Pandas")

    # ── Numba JIT pass ───────────────────────────────────────────────────────
    monitor.record("Start Numba")
    for params in grid:
        df = calculate_indicators(df_raw.copy(), params)
        run_njit_version(
            df=df,
            params=params,
            output_path=output_path,
            filename=filename,
        )
    monitor.record("End Numba")
