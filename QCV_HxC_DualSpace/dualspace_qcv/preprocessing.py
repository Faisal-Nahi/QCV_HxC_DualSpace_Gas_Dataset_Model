"""
preprocessing.py
================

Step 1 of the proposed framework: prepare a multivariate stream for entropy /
complexity analysis.

Handles: missing-value imputation, per-feature normalisation (min-max as in the
thesis specification, or robust MAD scaling as in the QCV-HxC paper), optional
light noise reduction, and optional first-difference detrending.

Crucial causal detail: **scalers are fit on the training portion only** and then
applied to the whole stream, so no information leaks from the future/test part
into preprocessing. (Note: per-feature *monotonic* scaling does not change
ordinal patterns at all -- argsort is invariant -- so it affects only the
autoencoder branch; we still fit causally for correctness and for the AE.)
"""

from __future__ import annotations
import numpy as np


class MultivariatePreprocessor:
    def __init__(self,
                 normalization: str = "minmax",   # {"minmax", "robust", "zscore", "none"}
                 impute: str = "ffill",           # {"ffill", "mean", "median", "none"}
                 smooth_window: int = 0,           # >1 enables moving-average noise reduction
                 detrend: bool = False):           # first-difference detrending
        self.normalization = normalization
        self.impute = impute
        self.smooth_window = int(smooth_window)
        self.detrend = bool(detrend)
        self._stats: dict[str, np.ndarray] = {}
        self._fitted = False

    # ---------------- imputation ----------------
    @staticmethod
    def _impute_array(X: np.ndarray, method: str, fill: np.ndarray | None) -> np.ndarray:
        X = X.astype(float).copy()
        if method == "none":
            return X
        for j in range(X.shape[1]):
            col = X[:, j]
            mask = ~np.isfinite(col)
            if not mask.any():
                continue
            if method == "ffill":
                # forward fill, then back fill any leading NaNs
                last = np.nan
                for i in range(col.size):
                    if np.isfinite(col[i]):
                        last = col[i]
                    elif np.isfinite(last):
                        col[i] = last
                # back-fill leading
                nxt = np.nan
                for i in range(col.size - 1, -1, -1):
                    if np.isfinite(col[i]):
                        nxt = col[i]
                    elif np.isfinite(nxt):
                        col[i] = nxt
                col[~np.isfinite(col)] = 0.0 if fill is None else fill[j]
            else:  # mean / median use the training fill value
                col[mask] = 0.0 if fill is None else fill[j]
            X[:, j] = col
        return X

    # ---------------- fit ----------------
    def fit(self, X_train: np.ndarray) -> "MultivariatePreprocessor":
        X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
        if X_train.shape[0] == 1 and X_train.shape[1] > 1 and X_train.ndim == 2:
            pass  # already (T, p)
        p = X_train.shape[1]

        # imputation fill values from training
        with np.errstate(invalid="ignore"):
            self._stats["mean_fill"] = np.nanmean(X_train, axis=0)
            self._stats["median_fill"] = np.nanmedian(X_train, axis=0)
        fill = self._stats["median_fill"] if self.impute == "median" else self._stats["mean_fill"]
        fill = np.where(np.isfinite(fill), fill, 0.0)
        Xtr = self._impute_array(X_train, self.impute, fill)

        # normalisation stats
        if self.normalization == "minmax":
            self._stats["min"] = Xtr.min(axis=0)
            self._stats["max"] = Xtr.max(axis=0)
        elif self.normalization == "robust":
            med = np.median(Xtr, axis=0)
            mad = np.median(np.abs(Xtr - med), axis=0)
            self._stats["median"] = med
            self._stats["mad"] = mad
        elif self.normalization == "zscore":
            self._stats["mu"] = Xtr.mean(axis=0)
            self._stats["sigma"] = Xtr.std(axis=0)
        self._fitted = True
        return self

    # ---------------- transform ----------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        fill = self._stats["median_fill"] if self.impute == "median" else self._stats["mean_fill"]
        fill = np.where(np.isfinite(fill), fill, 0.0)
        X = self._impute_array(X, self.impute, fill)

        if self.normalization == "minmax":
            rng = self._stats["max"] - self._stats["min"]
            rng = np.where(rng == 0, 1.0, rng)
            X = (X - self._stats["min"]) / rng
        elif self.normalization == "robust":
            X = (X - self._stats["median"]) / (self._stats["mad"] + 1e-8)
        elif self.normalization == "zscore":
            X = (X - self._stats["mu"]) / (self._stats["sigma"] + 1e-8)

        if self.smooth_window and self.smooth_window > 1:
            X = self._moving_average(X, self.smooth_window)
        if self.detrend:
            X = np.vstack([np.zeros((1, X.shape[1])), np.diff(X, axis=0)])
        return X

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @staticmethod
    def _moving_average(X: np.ndarray, k: int) -> np.ndarray:
        """Causal moving average (no look-ahead)."""
        out = np.empty_like(X)
        csum = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
        for i in range(X.shape[0]):
            lo = max(0, i - k + 1)
            out[i] = (csum[i + 1] - csum[lo]) / (i + 1 - lo)
        return out
