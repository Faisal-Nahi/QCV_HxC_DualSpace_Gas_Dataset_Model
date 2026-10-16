"""
windowing.py
============

Steps 2 and 3: chronological train/test split and causal sliding windows.

The split is sequential (never random) to prevent future information leakage.
Windows are causal: a window ending at absolute index i covers samples
[i-w+1, i], so a detection reported at time i uses only past/current data.
"""

from __future__ import annotations
import numpy as np


def chronological_split(T: int, train_ratio: float = 0.70) -> int:
    """Return the split index; samples [0, idx) are training, [idx, T) testing."""
    idx = int(round(train_ratio * T))
    return max(1, min(T - 1, idx))


def window_end_indices(T: int, w: int, step: int = 1) -> np.ndarray:
    """
    Absolute end indices of every causal window of length `w`.

    The first full window ends at index w-1 (covers [0, w-1]); the last ends at
    T-1. `step` controls the stride between consecutive windows (1 = every
    sample, fully online).
    """
    if T < w:
        return np.array([], dtype=int)
    return np.arange(w - 1, T, step, dtype=int)


def extract_window(X: np.ndarray, end_idx: int, w: int) -> np.ndarray:
    """Return the (w, p) window whose last row is at `end_idx`."""
    return X[end_idx - w + 1: end_idx + 1]
