import os
import time
import csv
import psutil

from backtest_grid_scale.config import get_sample_grid, cfg
from backtest_grid_scale.data import load_ohlcv
from backtest_grid_scale.indicators import calculate_indicators
from backtest_grid_scale.pandas_version import run_pandas_version
from backtest_grid_scale.njit_version import run_njit_version

# ─── Performance Monitor ──────────────────────────────────────────────────────

class PerformanceMonitor:
    def __init__(self, filename="performance.csv"):
        self.filename = filename
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        with open(self.filename, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'phase', 'ram_mb', 
                'process_cpu_pct', 'system_cpu_pct', 'elapsed_sec'
            ])

    def record(self, phase_name):
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
    symbol    = [cfg['symbol'], cfg['exchange'], 1]
    interval  = cfg['interval']
    data_dir  = cfg['data_dir']
    output_path= cfg['output_path']
    filename = f'SuperDaily_{symbol[0]}_{interval}.csv'

    df_raw = load_ohlcv(symbol, interval, data_dir)

    len_grid, grid = get_sample_grid()
    monitor = PerformanceMonitor()
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
