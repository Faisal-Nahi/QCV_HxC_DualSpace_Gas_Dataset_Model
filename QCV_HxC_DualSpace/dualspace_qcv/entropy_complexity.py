"""
entropy_complexity.py
=====================

Information-theoretic quantifiers on ordinal-pattern histograms, exactly as
used in the QCV-HxC paper (Bandt & Pompe permutation entropy and the
Lamberti / Martin / Rosso Jensen-Shannon statistical complexity).

This module *fixes* a simplification present in the original base model, whose
`statistical_complexity` omitted the normalisation constant Q0 and therefore
did not produce a complexity value bounded in [0, 1] on the H x C plane.
Here we use the full, properly normalised formulation so that both H and C lie
in [0, 1] and are comparable across embedding dimensions.

References
----------
Bandt, C. & Pompe, B. (2002). Permutation entropy. Phys. Rev. Lett. 88, 174102.
Lamberti, P. W., Martin, M. T., Plastino, A. & Rosso, O. A. (2004).
    Intensive entropic non-triviality measure. Physica A 334, 119-131.
Rosso, O. A. et al. (2007). Distinguishing noise from chaos. PRL 99, 154102.
"""

from __future__ import annotations
import numpy as np

_EPS = 1e-12


def _shannon_nat(p: np.ndarray) -> float:
    """Un-normalised Shannon entropy in *nats* (natural log). Zeros ignored."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def permutation_entropy(p: np.ndarray, n_patterns: int | None = None) -> float:
    """
    Normalised permutation entropy H(p) in [0, 1].

    H(p) = S(p) / ln(N),  where S is the natural-log Shannon entropy and
    N = n_patterns is the number of *possible* ordinal patterns (d!).
    """
    p = np.asarray(p, dtype=float)
    N = p.size if n_patterns is None else n_patterns
    if N <= 1:
        return 0.0
    return _shannon_nat(p) / np.log(N)


def jensen_shannon_complexity(p: np.ndarray, n_patterns: int | None = None) -> float:
    """
    Jensen-Shannon statistical complexity C(p) = Q_J(p) * H(p) in [0, 1].

    Q_J(p) = Q0 * JS(p, u),
        JS(p, u) = S((p + u) / 2) - S(p)/2 - S(u)/2   (natural-log Shannon)
        Q0       = -2 / [ ((N+1)/N) ln(N+1) - 2 ln(2N) + ln(N) ]
    with u the uniform distribution over N = d! possible patterns. Q0 is the
    inverse of the maximum attainable JS divergence, so Q_J in [0, 1].
    """
    p = np.asarray(p, dtype=float)
    N = p.size if n_patterns is None else n_patterns
    if N <= 1:
        return 0.0

    u = np.ones(N, dtype=float) / N
    # `p` is a length-N histogram (all d! patterns, zeros for the unobserved).
    if p.size != N:
        # Defensive: pad/truncate to length N.
        q = np.zeros(N, dtype=float)
        q[: min(N, p.size)] = p[: min(N, p.size)]
        p = q
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s

    m = 0.5 * (p + u)
    js = _shannon_nat(m) - 0.5 * _shannon_nat(p) - 0.5 * _shannon_nat(u)

    denom = ((N + 1.0) / N) * np.log(N + 1.0) - 2.0 * np.log(2.0 * N) + np.log(N)
    Q0 = -2.0 / denom
    Qj = Q0 * js

    H = permutation_entropy(p, N)
    C = Qj * H
    # Clip tiny numerical excursions outside [0, 1].
    return float(min(max(C, 0.0), 1.0))


def hxc(p: np.ndarray, n_patterns: int | None = None) -> tuple[float, float]:
    """Return (H, C) for an ordinal-pattern histogram `p`."""
    return permutation_entropy(p, n_patterns), jensen_shannon_complexity(p, n_patterns)


# --------------------------------------------------------------------------- #
#  Vectorised (batched) versions                                              #
# --------------------------------------------------------------------------- #
def _shannon_nat_rows(P: np.ndarray) -> np.ndarray:
    """Row-wise natural-log Shannon entropy of a (n, N) probability matrix."""
    P = np.asarray(P, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0.0, P * np.log(P), 0.0)
    return -terms.sum(axis=1)


def hxc_batch(counts: np.ndarray, n_patterns: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised (H, C) for a stack of ordinal-pattern histograms.

    Parameters
    ----------
    counts : (n_windows, N) array of *counts* (or probabilities) per pattern.
    n_patterns : N = d! (number of possible patterns).

    Returns
    -------
    H, C : (n_windows,) arrays, each identical to calling the scalar
           `permutation_entropy` / `jensen_shannon_complexity` per row.
    """
    counts = np.asarray(counts, dtype=float)
    n, N = counts.shape
    lnN = np.log(N) if N > 1 else 1.0

    totals = counts.sum(axis=1, keepdims=True)
    safe = totals[:, 0] > 0
    P = np.zeros_like(counts)
    np.divide(counts, totals, out=P, where=totals > 0)

    Sp = _shannon_nat_rows(P)                      # entropy of each window
    H = np.where(safe, Sp / lnN, 0.0)

    if N <= 1:
        return H, np.zeros(n)

    u = np.full(N, 1.0 / N)
    M = 0.5 * (P + u)
    Sm = _shannon_nat_rows(M)
    Su = _shannon_nat_rows(np.tile(u, (n, 1)))     # = ln(N) for every row
    js = Sm - 0.5 * Sp - 0.5 * Su

    denom = ((N + 1.0) / N) * np.log(N + 1.0) - 2.0 * np.log(2.0 * N) + np.log(N)
    Q0 = -2.0 / denom
    C = Q0 * js * H
    C = np.clip(np.where(safe, C, 0.0), 0.0, 1.0)
    return H, C
