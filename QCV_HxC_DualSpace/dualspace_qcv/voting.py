"""
voting.py
=========

Online consolidation of candidate alarms (from the QCV-HxC paper's voting
stage). Raw threshold crossings tend to arrive in bursts around a single drift;
the voter collapses each burst -- any run of crossings separated by <= `window`
samples -- into a single representative detection.

method:
    "first" -> earliest crossing in the burst (earliest possible alert)
    "last"  -> latest crossing
    "mean"  -> rounded mean position (smoother consolidation)
"""

from __future__ import annotations
import numpy as np


def consolidate(candidate_indices: np.ndarray, window: int = 500,
                method: str = "first") -> list[int]:
    """Collapse bursts of candidate indices into consolidated drift points."""
    idx = np.sort(np.asarray(candidate_indices, dtype=int))
    if idx.size == 0:
        return []

    drifts: list[int] = []
    burst = [int(idx[0])]
    for c in idx[1:]:
        if c - burst[-1] <= window:
            burst.append(int(c))
        else:
            drifts.append(_reduce(burst, method))
            burst = [int(c)]
    drifts.append(_reduce(burst, method))
    return drifts


def _reduce(burst: list[int], method: str) -> int:
    if method == "first":
        return burst[0]
    if method == "last":
        return burst[-1]
    if method == "mean":
        return int(round(float(np.mean(burst))))
    raise ValueError(f"unknown voting method '{method}'")
