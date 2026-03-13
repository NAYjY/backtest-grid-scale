# benchmarks/test_correctness.py
import numpy as np
import pandas as pd
from talib import abstract
from numba import njit
import os
import csv
import time
import psutil
import warnings
def test_ci_is_working():
    assert 1 + 1 == 2