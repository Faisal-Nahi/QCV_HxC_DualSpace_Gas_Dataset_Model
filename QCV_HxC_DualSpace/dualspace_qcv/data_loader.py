"""
data_loader.py
==============

Utilities to obtain multivariate streams together with ground-truth drift
locations. The framework is unsupervised (labels are never used for detection);
ground truth is used only for *evaluation*.

The thesis protocol is "learn normal behaviour on the first 70%, detect drift on
the last 30%". For that protocol to be meaningful the training portion must be a
single stationary (drift-free) regime and the drifts must occur inside the test
span. The generators below therefore keep the first regime long enough to cover
the whole training portion and place the drift points inside the test region by
default.

`load_csv` reads any real multivariate CSV so you can drop in electricity.csv or
your own data; pass known ground-truth drift indices separately.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Core generator                                                             #
# --------------------------------------------------------------------------- #
def make_multivariate_drift_stream(n=12000, p=6, drift_points=None,
                                   train_ratio=0.70, strength=1.0,
                                   temporal=False, seed=0):
    """
    Generate a p-variate stream with abrupt regime changes at `drift_points`.

    Each regime has a distinct mean, per-feature variance, AR(1) coefficient
    *vector* (temporal dynamics) and cross-correlation structure, so that BOTH
    the per-feature ordinal dynamics (raw space) and the joint distribution
    (latent space) shift at every drift.

    If `drift_points` is None, two drifts are placed at 1/3 and 2/3 of the test
    span (i.e. strictly inside the last 1-train_ratio fraction), so the training
    portion is a single normal regime.
    """
    rng = np.random.default_rng(seed)
    split = int(round(train_ratio * n))
    if drift_points is None:
        test_len = n - split
        drift_points = [split + test_len // 3, split + 2 * test_len // 3]
    drift_points = sorted(int(d) for d in drift_points)

    bounds = [0, *drift_points, n]
    n_regimes = len(bounds) - 1

    # Pre-draw clearly distinct regime parameter sets.
    def regime_params(r):
        rr = np.random.default_rng(seed * 100 + r + 1)
        mean = rr.uniform(-2, 2, size=p) * (1 + strength * 0.5 * r)
        scale = rr.uniform(0.6, 1.6, size=p) * (1 + strength * 0.3 * r)
        # AR vector: alternate sign / magnitude across regimes for dynamics shift
        base = rr.uniform(0.2, 0.9, size=p)
        phi = base * ((-1.0) ** r) * (0.6 + 0.4 * ((r + 1) % 2))
        phi = np.clip(phi, -0.95, 0.95)
        # correlation structure
        L = rr.standard_normal((p, p)) + (0.4 * strength * r) * np.eye(p)
        Cov = L @ L.T
        Dinv = np.diag(1.0 / np.sqrt(np.diag(Cov)))
        Corr = Dinv @ Cov @ Dinv
        chol = np.linalg.cholesky(Corr + 1e-6 * np.eye(p))
        return mean, scale, phi, chol

    X = np.zeros((n, p))
    for r in range(n_regimes):
        lo, hi = bounds[r], bounds[r + 1]
        m = hi - lo
        mean, scale, phi, chol = regime_params(r)
        seg = np.zeros((m, p))
        prev = rng.standard_normal(p)
        for t in range(m):
            innov = chol @ rng.standard_normal(p)
            prev = phi * prev + np.sqrt(np.clip(1 - phi ** 2, 1e-6, None)) * innov
            seg[t] = mean + scale * prev
        X[lo:hi] = seg

    cols = [f"f{j}" for j in range(p)]
    df = pd.DataFrame(X, columns=cols)
    return df, drift_points


# --------------------------------------------------------------------------- #
#  Named convenience wrappers                                                  #
# --------------------------------------------------------------------------- #
def make_multivariate_abrupt(n=12000, p=6, drift_points=None,
                             train_ratio=0.70, seed=0):
    """IID-style multivariate stream with abrupt drifts in the test span."""
    return make_multivariate_drift_stream(
        n=n, p=p, drift_points=drift_points, train_ratio=train_ratio,
        strength=1.2, temporal=False, seed=seed)


def make_multivariate_idt(n=15000, p=5, drift_points=None,
                          train_ratio=0.70, seed=1):
    """Strongly autocorrelated (IDT-style) multivariate stream with drifts."""
    return make_multivariate_drift_stream(
        n=n, p=p, drift_points=drift_points, train_ratio=train_ratio,
        strength=1.0, temporal=True, seed=seed)


# --------------------------------------------------------------------------- #
#  Real data                                                                   #
# --------------------------------------------------------------------------- #
def load_csv(path, feature_columns=None, drop_columns=None, max_rows=None):
    """
    Load a real multivariate CSV. Returns a DataFrame of numeric features.

    Example (Electricity):
        df = load_csv("electricity.csv",
                      feature_columns=["nswdemand", "vicdemand", "transfer"])
    """
    df = pd.read_csv(path, nrows=max_rows)
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    if feature_columns:
        df = df[feature_columns]
    else:
        df = df.select_dtypes(include=[np.number])
    return df
