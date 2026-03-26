from dataclasses import dataclass, field
import numpy as np
import itertools
from data import StrategyParams
import random
import yaml
from pathlib import Path

def load_config() -> dict:
    # try local config first, fall back to default
    root = Path(__file__).parent.parent  # benchmarks/ → root
    
    local   = root / "config.local.yaml"
    default = root / "config.yaml"
    
    path = local if local.exists() else default
    
    with open(path) as f:
        return yaml.safe_load(f)
cfg = load_config()

@dataclass
class OptimizationSpace:
    # ATR Long
    atr_len_l:   tuple = (2, 3, 1)    # (min, max, step)
    atr_mult_l:  tuple = (0.5, 1, 0.5)

    # ATR Short
    atr_len_s:   tuple = (2, 3, 1)
    atr_mult_s:  tuple = (0.5, 1, 0.5)

    # ROC Long
    roc_len_l:   tuple = (10, 11, 5)
    roc_thresh_l: tuple = (4, 5, 1)

    # ROC Short
    roc_len_s:   tuple = (60, 61, 5)
    roc_thresh_s: tuple = (4, 5, 1)

    # Stop Loss
    sl_pct_l:    tuple = (3, 4, 1)
    sl_pct_s:    tuple = (3, 4, 1)

def build_range(t: tuple):
    min_, max_, step = t
    if isinstance(step, float):
        return list(np.arange(min_, max_ + step, step))
    return list(range(int(min_), int(max_) + 1, int(step)))

def get_search_grid(space: OptimizationSpace = OptimizationSpace()) -> tuple[int, list[StrategyParams]]:
    grid = [
        StrategyParams(
            atr_len_l   = int(combo[0]),
            atr_mult_l  = float(combo[1]),
            atr_len_s   = int(combo[2]),
            atr_mult_s  = float(combo[3]),
            roc_len_l   = int(combo[4]),
            roc_thresh_l = float(combo[5]),
            roc_len_s   = int(combo[6]),
            roc_thresh_s = float(combo[7]),
            sl_pct_l    = float(combo[8]),
            sl_pct_s    = float(combo[9]),
        )
        for combo in itertools.product(
            build_range(space.atr_len_l),
            build_range(space.atr_mult_l),
            build_range(space.atr_len_s),
            build_range(space.atr_mult_s),
            build_range(space.roc_len_l),
            build_range(space.roc_thresh_l),
            build_range(space.roc_len_s),
            build_range(space.roc_thresh_s),
            build_range(space.sl_pct_l),
            build_range(space.sl_pct_s),
        )
    ]
    return len(grid), grid
    
def get_sample_grid(
    space: OptimizationSpace = OptimizationSpace(),
    n_samples: int = 100,
) -> tuple[int, list[StrategyParams]]:
    
    len_grid, full_grid = get_search_grid(space)
    sampled = random.sample(full_grid, min(n_samples, len_grid))
    return len(sampled), sampled
