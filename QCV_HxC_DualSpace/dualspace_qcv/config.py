"""
config.py
=========

Central configuration for the Dual-Space Entropy-Complexity multivariate drift
detector. Every step of the proposed framework is controlled from here, with
defaults distilled from the QCV-HxC paper's recommended ranges
(d=3, tau in {1,2}, w in [80,120], q in [0.93,0.96], Wsc~400, Wq~1000).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DualSpaceConfig:
    # ---- Step 1: preprocessing ----
    normalization: str = "minmax"      # {"minmax","robust","zscore","none"}
    impute: str = "ffill"              # {"ffill","mean","median","none"}
    smooth_window: int = 0             # >1 => causal moving-average denoising
    detrend: bool = False              # first-difference detrending

    # ---- Step 2: chronological split ----
    train_ratio: float = 0.70

    # ---- Steps 3-6/8: ordinal + windowing ----
    d: int = 3                         # embedding dimension
    tau: int = 1                       # delay
    w: int = 100                       # entropy/complexity window
    step: int = 1                      # window stride (1 = fully online)
    ema_span: int = 5                  # causal smoothing of H,C (<=1 disables)
    lam_H: float = 1.0                 # weight on entropy deviation
    lam_C: float = 1.0                 # weight on complexity deviation

    # multivariate fusion across channels (paper's future-work strategies)
    raw_fusion: str = "mean"           # {"mean","max","weighted","vote"}
    latent_fusion: str = "mean"
    vote_quantile: float = 0.90

    # ---- Step 7: autoencoder (latent space) ----
    latent_dim: int | None = None      # default: max(2, p//2)
    ae_hidden: tuple[int, ...] | None = None
    ae_lr: float = 1e-2
    ae_epochs: int = 300
    ae_batch: int = 64
    ae_patience: int = 25
    ae_seed: int = 42

    # ---- Step 9: dual-space fusion ----
    alpha: float = 0.5                 # raw vs latent mixing weight
    gamma: float = 0.25               # agreement (min) coefficient

    # ---- Step 10: adaptive threshold ----
    k: float = 3.0                     # sensitivity: Threshold = mu + k*sigma
    robust_threshold: bool = False     # use median + k*1.4826*MAD instead

    # ---- Step 11 post-processing: voting ----
    use_voting: bool = True
    vote_window: int = 500
    vote_method: str = "first"         # {"first","last","mean"}

    # ---- evaluation ----
    tolerance: int = 500
    verbose: bool = False
