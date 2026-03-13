

# ─── Performance Monitor ──────────────────────────────────────────────────────

class PerformanceMonitor:
    def __init__(self, filename="fast_performance.csv"):
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
