"""
spaces.py
=========

Steps 4-8: turn a (possibly multivariate) signal matrix into a single drift
score for one representation space (raw feature space OR latent space).

For each channel j and each causal window ending at time i:
    1. compute the ordinal-pattern histogram of that channel over the window;
    2. derive permutation entropy H_j(i) and statistical complexity C_j(i);
    3. (optional) causal EMA smoothing of the H_j and C_j sequences;
    4. deviation from the *training* baseline:
           dH_j(i) = |H_j(i) - H_base_j|,   dC_j(i) = |C_j(i) - C_base_j|;
    5. robust-standardise dH_j and dC_j with training statistics so entropy and
       complexity deviations are comparable, then combine per channel:
           r_j(i) = lam_H * z(dH_j) + lam_C * z(dC_j);
    6. fuse the per-channel scores into one space score S(i) using one of the
       multivariate fusion strategies proposed as future work in the paper
       (mean, max/maximum aggregation, weighted summation, majority voting).

This module is used twice by the detector: once on the raw features (giving
R_t) and once on the autoencoder latent codes (giving L_t).
"""

from __future__ import annotations
import numpy as np

from .ordinal_patterns import OrdinalEncoder
from .entropy_complexity import hxc_batch
from .windowing import window_end_indices


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Causal exponential moving average (pandas 'adjust=False' semantics)."""
    if span is None or span <= 1:
        return x
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _robust_center_scale(v: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based scale (1.4826*MAD), with std fallback."""
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = float(np.std(v))
    if scale < 1e-9:
        scale = 1.0
    return med, scale


class SpaceAnalyzer:
    """
    Compute a per-space drift score sequence for a multichannel signal.

    Parameters
    ----------
    d, tau, w        : ordinal embedding dimension, delay, and window size.
    ema_span         : causal smoothing span for H and C (<=1 disables).
    lam_H, lam_C     : weights on entropy vs complexity deviations.
    fusion           : {"mean","max","weighted","vote"} multichannel fusion.
    vote_quantile    : per-channel training quantile used by "vote" fusion.
    step             : window stride (1 = fully online).
    """

    def __init__(self, d=3, tau=1, w=100, ema_span=5,
                 lam_H=1.0, lam_C=1.0, fusion="mean",
                 vote_quantile=0.90, step=1):
        self.enc = OrdinalEncoder(d=d, tau=tau)
        self.d, self.tau, self.w = d, tau, w
        self.ema_span = ema_span
        self.lam_H, self.lam_C = lam_H, lam_C
        self.fusion = fusion
        self.vote_quantile = vote_quantile
        self.step = step
        self._fitted = False

    # ---------- Steps 4/5/6/8: raw H,C sequences per channel ----------
    def _window_counts(self, codes: np.ndarray, ends: np.ndarray,
                       span: int, N: int) -> np.ndarray:
        """
        (n_windows, N) matrix of ordinal-pattern counts, one row per window.

        A window ending at absolute index e contains the codes whose embedding
        starts in [e-w+1, e-span], i.e. the fixed-length-L=(w-span) slice
        codes[e-w+1 : e-span+1]. For the fully-online case (step == 1) this is a
        sliding window of constant length over `codes`, so the histogram is
        maintained incrementally in O(1) per step; otherwise we fall back to a
        per-window bincount. Both paths give identical counts.
        """
        L = self.w - span                            # patterns per window
        n = ends.size
        counts = np.zeros((n, N), dtype=float)
        if n == 0:
            return counts

        if self.step == 1 and L > 0:
            a0 = ends[0] - self.w + 1                 # == 0
            cur = np.bincount(codes[a0: a0 + L], minlength=N).astype(float)
            counts[0] = cur
            for wi in range(1, n):
                a = ends[wi] - self.w + 1            # advances by 1
                cur = cur.copy()
                cur[codes[a - 1]] -= 1.0             # code leaving the window
                cur[codes[a + L - 1]] += 1.0         # code entering the window
                counts[wi] = cur
        else:
            for wi, e in enumerate(ends):
                a = e - self.w + 1
                b = e - span + 1
                seg = codes[a: b]
                if seg.size:
                    counts[wi] = np.bincount(seg, minlength=N).astype(float)
        return counts

    def _hc_sequences(self, X: np.ndarray):
        """Return end indices and (n_windows, n_channels) arrays of H and C."""
        X = np.atleast_2d(X)
        if X.shape[0] < X.shape[1] and X.ndim == 2 and X.shape[0] == 1:
            X = X.T
        T, C = X.shape
        ends = window_end_indices(T, self.w, self.step)
        N = self.enc.n_patterns
        span = (self.d - 1) * self.tau
        H = np.zeros((ends.size, C))
        Cx = np.zeros((ends.size, C))

        for j in range(C):
            codes = self.enc.codes(X[:, j])          # vectorised, once/channel
            counts = self._window_counts(codes, ends, span, N)
            Hj, Cj = hxc_batch(counts, N)            # vectorised over windows
            # causal EMA smoothing per channel
            H[:, j] = _ema(Hj, self.ema_span)
            Cx[:, j] = _ema(Cj, self.ema_span)
        return ends, H, Cx

    # ---------- fit baselines + scalers on training windows ----------
    def fit(self, X: np.ndarray, train_mask: np.ndarray):
        """
        `train_mask` is a boolean array over the *windows* (aligned to the end
        indices) marking which windows fall entirely inside the training split.
        """
        ends, H, Cx = self._hc_sequences(X)
        self.ends_ = ends
        self.H_, self.C_ = H, Cx
        self.n_channels_ = H.shape[1]
        tm = train_mask

        # Step 5/8 baselines: mean over training windows, per channel.
        self.H_base_ = H[tm].mean(axis=0)
        self.C_base_ = Cx[tm].mean(axis=0)

        # deviations
        dH = np.abs(H - self.H_base_)
        dC = np.abs(Cx - self.C_base_)

        # robust standardisation params from training deviations (per channel)
        self._dH_loc = np.zeros(self.n_channels_); self._dH_scl = np.ones(self.n_channels_)
        self._dC_loc = np.zeros(self.n_channels_); self._dC_scl = np.ones(self.n_channels_)
        for j in range(self.n_channels_):
            self._dH_loc[j], self._dH_scl[j] = _robust_center_scale(dH[tm, j])
            self._dC_loc[j], self._dC_scl[j] = _robust_center_scale(dC[tm, j])

        # per-channel combined score r_j
        r = self._combine_channels_scores(dH, dC)     # (n_windows, n_channels)
        self.r_ = r

        # thresholds for majority-vote fusion (per-channel training quantile)
        self._vote_thr = np.quantile(r[tm], self.vote_quantile, axis=0)

        # weights for weighted fusion: proportional to training-window spread
        spread = r[tm].std(axis=0)
        self._weights = spread / (spread.sum() + 1e-12) if spread.sum() > 0 else \
            np.ones(self.n_channels_) / self.n_channels_

        # fuse channels -> per-space raw score, then standardise on training
        S_raw = self._fuse(r)
        self._S_loc, self._S_scl = _robust_center_scale(S_raw[tm])
        self.score_ = (S_raw - self._S_loc) / self._S_scl
        self._fitted = True
        return self

    def _combine_channels_scores(self, dH, dC):
        zH = (dH - self._dH_loc) / self._dH_scl
        zC = (dC - self._dC_loc) / self._dC_scl
        return self.lam_H * zH + self.lam_C * zC

    def _fuse(self, r):
        if self.fusion == "mean":
            return r.mean(axis=1)
        if self.fusion == "max":
            return r.max(axis=1)
        if self.fusion == "weighted":
            return r @ self._weights
        if self.fusion == "vote":
            # fraction of channels exceeding their own training threshold
            return (r > self._vote_thr).mean(axis=1)
        raise ValueError(f"unknown fusion '{self.fusion}'")

    # convenience accessors
    @property
    def score(self) -> np.ndarray:
        """Standardised per-space drift score aligned to `ends_`."""
        return self.score_
