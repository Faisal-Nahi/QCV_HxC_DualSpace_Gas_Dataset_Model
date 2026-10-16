"""
fusion.py
=========

Step 9 (dual-space score fusion) and Step 10 (adaptive threshold).

Fusion
------
    D_t = alpha * R_t + (1 - alpha) * L_t + gamma * min(R_t, L_t)

R_t (raw space) and L_t (latent space) arrive already standardised on the
training windows (see SpaceAnalyzer), so alpha is a meaningful mixing weight and
the agreement term gamma * min(R_t, L_t) rewards *joint* evidence: when both
spaces flag a change the minimum is large and confidence is boosted; when the
spaces disagree the minimum is small (or negative) and the spurious peak is
damped.

Adaptive threshold
------------------
    Threshold = mu_D + k * sigma_D          (Gaussian form, per the spec)
or, more robustly,
    Threshold = median_D + k * (1.4826 * MAD_D)   (set robust=True)

computed from the *training-window* fused scores only.
"""

from __future__ import annotations
import numpy as np


def fuse_scores(R: np.ndarray, L: np.ndarray, alpha: float = 0.5,
                gamma: float = 0.25) -> np.ndarray:
    """Combine the raw-space and latent-space scores into the final D_t."""
    R = np.asarray(R, dtype=float)
    L = np.asarray(L, dtype=float)
    return alpha * R + (1.0 - alpha) * L + gamma * np.minimum(R, L)


def adaptive_threshold(D_train: np.ndarray, k: float = 3.0,
                       robust: bool = False) -> float:
    """Adaptive drift threshold from training fused scores."""
    D_train = np.asarray(D_train, dtype=float)
    if robust:
        med = float(np.median(D_train))
        mad = float(np.median(np.abs(D_train - med)))
        scale = 1.4826 * mad if mad > 0 else float(np.std(D_train))
        return med + k * scale
    return float(np.mean(D_train) + k * np.std(D_train))
