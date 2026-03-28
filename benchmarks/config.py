from dataclasses import dataclass
import numpy as np
import itertools
from data import StrategyParams
import random
import yaml
from pathlib import Path


def load_config() -> dict:
    root = Path(__file__).parent.parent  # benchmarks/ → root
    local   = root / "config.local.yaml"
    default = root / "config.yaml"
    path = local if local.exists() else default
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config()


def _t(key: str, default: tuple) -> tuple:
    """Read a {min, max, step} block from cfg['grid'] into a tuple."""
    d = cfg.get("grid", {}).get(key, {})
    if not d:
        return default
    return (d["min"], d["max"], d["step"])


@dataclass
class OptimizationSpace:
    # ATR Long
    atr_len_l:    tuple = None
    atr_mult_l:   tuple = None
    # ATR Short
    atr_len_s:    tuple = None
    atr_mult_s:   tuple = None
    # ROC Long
    roc_len_l:    tuple = None
    roc_thresh_l: tuple = None
    # ROC Short
    roc_len_s:    tuple = None
    roc_thresh_s: tuple = None
    # Stop Loss
    sl_pct_l:     tuple = None
    sl_pct_s:     tuple = None

    def __post_init__(self):
        self.atr_len_l    = self.atr_len_l    or _t("atr_len_l",    (2,   3,  1  ))
        self.atr_mult_l   = self.atr_mult_l   or _t("atr_mult_l",   (0.5, 1.0, 0.5))
        self.atr_len_s    = self.atr_len_s    or _t("atr_len_s",    (2,   3,  1  ))
        self.atr_mult_s   = self.atr_mult_s   or _t("atr_mult_s",   (0.5, 1.0, 0.5))
        self.roc_len_l    = self.roc_len_l    or _t("roc_len_l",    (10, 11,  5  ))
        self.roc_thresh_l = self.roc_thresh_l or _t("roc_thresh_l", (4,   5,  1  ))
        self.roc_len_s    = self.roc_len_s    or _t("roc_len_s",    (60, 61,  5  ))
        self.roc_thresh_s = self.roc_thresh_s or _t("roc_thresh_s", (4,   5,  1  ))
        self.sl_pct_l     = self.sl_pct_l     or _t("sl_pct_l",     (3,   4,  1  ))
        self.sl_pct_s     = self.sl_pct_s     or _t("sl_pct_s",     (3,   4,  1  ))


def build_range(t: tuple) -> list:
    min_, max_, step = t
    if isinstance(step, float):
        return list(np.arange(min_, max_ + step, step))
    return list(range(int(min_), int(max_) + 1, int(step)))


def get_search_grid(space: OptimizationSpace = None) -> tuple[int, list[StrategyParams]]:
    if space is None:
        space = OptimizationSpace()
    grid = [
        StrategyParams(
            atr_len_l    = int(combo[0]),
            atr_mult_l   = float(combo[1]),
            atr_len_s    = int(combo[2]),
            atr_mult_s   = float(combo[3]),
            roc_len_l    = int(combo[4]),
            roc_thresh_l = float(combo[5]),
            roc_len_s    = int(combo[6]),
            roc_thresh_s = float(combo[7]),
            sl_pct_l     = float(combo[8]),
            sl_pct_s     = float(combo[9]),
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
    space: OptimizationSpace = None,
    n_samples: int = 100,
) -> tuple[int, list[StrategyParams]]:
    if space is None:
        space = OptimizationSpace()
    len_grid, full_grid = get_search_grid(space)
    sampled = random.sample(full_grid, min(n_samples, len_grid))
    return len(sampled), sampled