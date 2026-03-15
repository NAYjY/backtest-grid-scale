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

SYMBOL       = ['S501!', 'TFEX', 1]
INTERVAL     = '1h'
DATA_DIR     = '/home/nayjy/Workplace/onRunMA/new_src'

scr_path     = '/home/nayjy/Workplace/onRunMA/new_src'
path = '/home/nayjy/Workplace/compare_claen/'
filename = f'SuperDaily_{SYMBOL[0]}_{INTERVAL}.csv'

ATR_LEN_L_MAX    = 10
ATR_MULT_L_MAX   = 4
ATR_LEN_S_MAX    = 10
ATR_MULT_S_MAX   = 4

ROC_LEN_L_MAX    = 50
ROC_THRESH_L_MAX = 8
ROC_LEN_S_MAX    = 90
ROC_THRESH_S_MAX = 8

SL_PCT_L_MAX     = 10   # stop loss % for longs
SL_PCT_S_MAX     = 10   # stop loss % for shorts

ATR_LEN_L_MIN    = 2
ATR_MULT_L_MIN   = 0.5
ATR_LEN_S_MIN    = 2
ATR_MULT_S_MIN   = 0.5

ROC_LEN_L_MIN    = 10
ROC_THRESH_L_MIN = 4
ROC_LEN_S_MIN    = 60
ROC_THRESH_S_MIN = 4

SL_PCT_L_MIN     = 3   # stop loss % for longs
SL_PCT_S_MIN     = 3   # stop loss % for shorts

ATR_LEN_L_STEP    = 1
ATR_MULT_L_STEP   = 0.5
ATR_LEN_S_STEP    = 1
ATR_MULT_S_STEP   = 0.5

ROC_LEN_L_STEP    = 5
ROC_THRESH_L_STEP = 1
ROC_LEN_S_STEP    = 5
ROC_THRESH_S_STEP = 1

SL_PCT_L_STEP     = 1   # stop loss % for longs
SL_PCT_S_STEP     = 1   # stop loss % for shorts