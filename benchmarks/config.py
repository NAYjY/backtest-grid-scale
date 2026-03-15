# ─── Grid: benchmarks/config.py ───────────────────────────────────────────────────────

@dataclass
class OptimizationSpace:
    # ATR Ranges
    atr_len_range: tuple = (3, 10)
    atr_mult_range: tuple = (1.0, 3.0)
    
    # Momentum (ROC) Ranges
    roc_len_range: tuple = (20, 80)
    roc_thresh_range: tuple = (2, 8)
    
    # Risk Ranges
    sl_pct_range: tuple = (2, 10)

import itertools

def get_search_grid(space: OptimizationSpace):
    """Generates the full grid of parameters to test."""
    # Define steps (e.g., step of 1 for length, 0.5 for multiplier)
    atr_lens = range(space.atr_len_range[0], space.atr_len_range[1] + 1)
    atr_mults = np.arange(space.atr_mult_range[0], space.atr_mult_range[1] + 0.5, 0.5)
    
    # Use product to get every combination
    grid = list(itertools.product(atr_lens, atr_mults))
    return len(grid), grid
