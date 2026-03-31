"""
config.py
---------
Configuration loading and grid-search space utilities.

Responsibilities
~~~~~~~~~~~~~~~~
* Load ``config.yaml`` / ``config.local.yaml`` from the repository root.
* Expose ``OptimizationSpace`` — a dataclass whose fields each describe the
  ``(min, max, step)`` range for one strategy parameter.
* Build the full Cartesian-product grid or a random sample from that grid.
"""

from dataclasses import dataclass
import numpy as np
import itertools
from .data import StrategyParams
import random
import yaml
from pathlib import Path


def load_config() -> dict:
    """Load runtime configuration from YAML.

    Looks for ``config.local.yaml`` in the repository root first (gitignored,
    intended for local overrides); falls back to ``config.yaml``.

    Returns
    -------
    dict
        Parsed YAML document as a plain Python dictionary.
    """
    root = Path(__file__).parent.parent  # benchmarks/ → root
    local   = root / "config.local.yaml"
    default = root / "config.yaml"
    path = local if local.exists() else default
    with open(path) as f:
        return yaml.safe_load(f)

cfg = load_config()


def _t(key: str, default: tuple) -> tuple:
    """Read a ``{min, max, step}`` block from ``cfg['grid']`` into a tuple.

    Parameters
    ----------
    key : str
        Parameter name as it appears under the ``grid:`` section of the config.
    default : tuple
        Fallback ``(min, max, step)`` returned when the key is absent.

    Returns
    -------
    tuple
        ``(min, max, step)`` taken from the config or *default*.
    """
    d = cfg.get("grid", {}).get(key, {})
    if not d:
        return default
    return (d["min"], d["max"], d["step"])


@dataclass
class OptimizationSpace:
    """Parameter search space for the strategy grid search.

    Each field holds a ``(min, max, step)`` tuple that defines the inclusive
    range for one strategy parameter.  ``None`` fields are populated from
    ``config.yaml`` by ``__post_init__``, falling back to hard-coded defaults
    when the key is absent from the config.

    Attributes
    ----------
    atr_len_l : tuple
        ATR look-back period for the *long* SuperTrend band.
    atr_mult_l : tuple
        ATR multiplier for the *long* SuperTrend band.
    atr_len_s : tuple
        ATR look-back period for the *short* SuperTrend band.
    atr_mult_s : tuple
        ATR multiplier for the *short* SuperTrend band.
    roc_len_l : tuple
        ROC look-back period used for the *long* momentum filter.
    roc_thresh_l : tuple
        ROC EMA threshold for the *long* momentum filter.
    roc_len_s : tuple
        ROC look-back period used for the *short* momentum filter.
    roc_thresh_s : tuple
        ROC EMA threshold for the *short* momentum filter.
    sl_pct_l : tuple
        Stop-loss percentage for *long* trades.
    sl_pct_s : tuple
        Stop-loss percentage for *short* trades.
    """

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
    """Convert a ``(min, max, step)`` tuple into a flat list of values.

    Uses ``numpy.arange`` for float steps and ``range`` for integer steps.
    The *max* value is **inclusive**.

    Parameters
    ----------
    t : tuple
        ``(min, max, step)`` — all three may be int or float.

    Returns
    -------
    list
        Ordered sequence of parameter values covering ``[min, max]``.
    """
    min_, max_, step = t
    if isinstance(step, float):
        return list(np.arange(min_, max_ + step, step))
    return list(range(int(min_), int(max_) + 1, int(step)))


def get_search_grid(space: OptimizationSpace = None) -> tuple[int, list[StrategyParams]]:
    """Build the full Cartesian-product parameter grid.

    Parameters
    ----------
    space : OptimizationSpace, optional
        Search space definition.  Defaults to ``OptimizationSpace()`` which
        reads ranges from ``config.yaml``.

    Returns
    -------
    tuple[int, list[StrategyParams]]
        ``(grid_size, list_of_params)`` where *grid_size* equals
        ``len(list_of_params)``.
    """
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
    """Draw a random sample from the full parameter grid.

    Useful for quick benchmarks or smoke tests without running every
    combination.  The sample is drawn without replacement using
    ``random.sample``.

    Parameters
    ----------
    space : OptimizationSpace, optional
        Search space definition.  Defaults to ``OptimizationSpace()``.
    n_samples : int, optional
        Maximum number of parameter sets to return.  Capped at the full grid
        size when the grid is smaller than *n_samples*.  Default is 100.

    Returns
    -------
    tuple[int, list[StrategyParams]]
        ``(sample_size, list_of_params)``.
    """
    if space is None:
        space = OptimizationSpace()
    len_grid, full_grid = get_search_grid(space)
    sampled = random.sample(full_grid, min(n_samples, len_grid))
    return len(sampled), sampled