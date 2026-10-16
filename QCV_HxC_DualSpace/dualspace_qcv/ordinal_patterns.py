"""
ordinal_patterns.py
===================

Bandt-Pompe ordinal-pattern transformation of a univariate signal.

Given a 1-D signal x, an embedding dimension d and a delay tau, each length-d
sub-sequence (x_i, x_{i+tau}, ..., x_{i+(d-1)tau}) is mapped to the permutation
that sorts it in ascending order. Ties are broken deterministically ("stable"
ordering: equal values keep their original time order), following the paper's
convention, so the histogram is reproducible.

The histogram has exactly d! bins (one per possible pattern), with zeros for
patterns that never occur -- this is required for an unbiased complexity
estimate on the full H x C manifold.
"""

from __future__ import annotations
import numpy as np
from itertools import permutations
from math import factorial


def _pattern_index_map(d: int) -> dict[tuple[int, ...], int]:
    """Map every permutation of {0,...,d-1} to a fixed histogram index."""
    return {perm: k for k, perm in enumerate(permutations(range(d)))}


def ordinal_histogram(x: np.ndarray, d: int = 3, tau: int = 1,
                      index_map: dict | None = None) -> np.ndarray:
    """
    Return the length-d! probability histogram of ordinal patterns in `x`.

    Deterministic tie handling: `np.argsort(..., kind='stable')` keeps the
    original index order for equal values, so the mapping is unique and
    reproducible (cf. Rey et al. 2023, used by the paper).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    N = factorial(d)
    counts = np.zeros(N, dtype=float)

    span = (d - 1) * tau
    if n <= span:
        # Not enough samples for a single pattern -> uniform-ish empty hist.
        return counts  # all zeros; caller handles degenerate windows

    if index_map is None:
        index_map = _pattern_index_map(d)

    # Build the (n-span) x d matrix of embedded vectors, then argsort each row.
    idx = np.arange(0, n - span)
    # columns: x[i], x[i+tau], ..., x[i+(d-1)tau]
    embedded = np.empty((idx.size, d), dtype=float)
    for j in range(d):
        embedded[:, j] = x[idx + j * tau]

    order = np.argsort(embedded, axis=1, kind="stable")  # (m, d) permutations
    for row in order:
        counts[index_map[tuple(row)]] += 1.0

    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


class OrdinalEncoder:
    """
    Convenience wrapper that caches the pattern index map for a fixed (d, tau).
    Encoding many windows with the same d is much faster with a cached map.
    """

    def __init__(self, d: int = 3, tau: int = 1):
        self.d = int(d)
        self.tau = int(tau)
        self.n_patterns = factorial(self.d)
        self._index_map = _pattern_index_map(self.d)

        # Vectorised encoding support: map a permutation row to its histogram
        # index through a dense mixed-radix lookup table. A length-d permutation
        # `perm` has key sum(perm[j] * d**(d-1-j)); `_key_to_code[key]` is the
        # same lexicographic index used by `_index_map`, so results are
        # identical to the per-row path -- just far faster.
        self._radix = (self.d ** np.arange(self.d - 1, -1, -1)).astype(np.int64)
        key_to_code = np.full(self.d ** self.d, -1, dtype=np.int64)
        for perm, k in self._index_map.items():
            key = int(np.dot(np.asarray(perm, dtype=np.int64), self._radix))
            key_to_code[key] = k
        self._key_to_code = key_to_code

    def histogram(self, x: np.ndarray) -> np.ndarray:
        return ordinal_histogram(x, self.d, self.tau, self._index_map)

    def codes(self, x: np.ndarray) -> np.ndarray:
        """
        Return the ordinal-pattern code (0..d!-1) for *every* embedding position
        in a 1-D signal, computed with a single vectorised stable argsort.

        The embedding starting at absolute index i covers samples
        (x[i], x[i+tau], ..., x[i+(d-1)tau]); there are len(x) - (d-1)*tau of
        them. Ties are broken with `kind="stable"`, exactly as in
        `ordinal_histogram`, so the two encoders agree bit-for-bit.
        """
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        span = (self.d - 1) * self.tau
        if n <= span:
            return np.empty(0, dtype=np.int64)
        m = n - span
        embedded = np.empty((m, self.d), dtype=float)
        for j in range(self.d):
            embedded[:, j] = x[j * self.tau: j * self.tau + m]
        order = np.argsort(embedded, axis=1, kind="stable").astype(np.int64)
        keys = order @ self._radix
        return self._key_to_code[keys]
