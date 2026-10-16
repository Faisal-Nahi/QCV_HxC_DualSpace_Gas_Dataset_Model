"""
detector.py
===========

The Dual-Space Entropy-Complexity Framework for Unsupervised Multivariate
Concept Drift Detection.

This single class wires together the full pipeline described in the proposal:

    Multivariate stream
        -> (1) preprocessing
        -> (2) chronological 70/30 split
        -> (3) causal sliding windows
        -> two parallel branches:
             RAW    : X            -> per-feature ordinal H,C -> R_t
             LATENT : X -> AE -> Z -> per-dim   ordinal H,C -> L_t
        -> (9)  dual-space fusion  D_t = a*R + (1-a)*L + g*min(R,L)
        -> (10) adaptive threshold  mu_D + k*sigma_D   (training only)
        -> (11) drift / no-drift decision  (+ optional online voting)

Everything is causal and leakage-free: autoencoder training, baselines,
standardisation and the threshold are all estimated on the training windows
only. The framework is unsupervised -- ground-truth labels are used solely for
evaluation, never for detection.

Per the specification the implementation ends at the drift decision; it does
*not* include a separate explainability module. However, because the pipeline
naturally produces per-channel and per-space contributions, these are exposed
in the result object as lightweight attribution diagnostics (they do not affect
the decision).
"""

from __future__ import annotations
import numpy as np

from .config import DualSpaceConfig
from .preprocessing import MultivariatePreprocessor
from .windowing import chronological_split
from .autoencoder import Autoencoder
from .spaces import SpaceAnalyzer
from .fusion import fuse_scores, adaptive_threshold
from .voting import consolidate
from .metrics import evaluate_detections


class DualSpaceResult:
    """Container for the outputs of a detection run."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def summary(self) -> str:
        lines = [
            "Dual-Space Entropy-Complexity Drift Detection",
            "=" * 46,
            f"stream length T           : {self.T}",
            f"features p                : {self.p}",
            f"latent dim k              : {self.latent_dim}",
            f"train/test split index    : {self.split_index} "
            f"({self.train_ratio:.0%} / {1 - self.train_ratio:.0%})",
            f"windows (train/test)      : {self.n_train_windows} / {self.n_test_windows}",
            f"adaptive threshold        : {self.threshold:.4f}",
            f"candidate alarms (test)   : {self.n_candidates}",
            f"consolidated drift points : {self.drift_points}",
        ]
        return "\n".join(lines)


class DualSpaceDriftDetector:
    def __init__(self, config: DualSpaceConfig | None = None, **overrides):
        self.cfg = config or DualSpaceConfig()
        for k, v in overrides.items():
            setattr(self.cfg, k, v)

    # ------------------------------------------------------------------ #
    def fit_detect(self, X, ground_truth=None) -> DualSpaceResult:
        cfg = self.cfg
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        T, p = X.shape

        # -------- Step 2: chronological split --------
        split = chronological_split(T, cfg.train_ratio)
        if cfg.verbose:
            print(f"[1-2] preprocessing + split @ {split} (T={T}, p={p})")

        # -------- Step 1: preprocessing (fit on training only) --------
        pre = MultivariatePreprocessor(
            normalization=cfg.normalization, impute=cfg.impute,
            smooth_window=cfg.smooth_window, detrend=cfg.detrend,
        )
        pre.fit(X[:split])
        Xp = pre.transform(X)

        # -------- Step 7: autoencoder trained on training portion --------
        latent_dim = cfg.latent_dim or max(2, p // 2)
        if cfg.verbose:
            print(f"[7]   training autoencoder  p={p} -> k={latent_dim}")
        ae = Autoencoder(
            n_features=p, latent_dim=latent_dim, hidden=cfg.ae_hidden,
            lr=cfg.ae_lr, epochs=cfg.ae_epochs, batch_size=cfg.ae_batch,
            patience=cfg.ae_patience, seed=cfg.ae_seed, verbose=cfg.verbose,
        )
        ae.fit(Xp[:split])
        Z = ae.encode(Xp)                      # latent codes for the whole stream

        # -------- Step 3: window end indices + training mask --------
        # (SpaceAnalyzer computes its own end indices; we mark which are training)
        raw = SpaceAnalyzer(
            d=cfg.d, tau=cfg.tau, w=cfg.w, ema_span=cfg.ema_span,
            lam_H=cfg.lam_H, lam_C=cfg.lam_C, fusion=cfg.raw_fusion,
            vote_quantile=cfg.vote_quantile, step=cfg.step,
        )
        lat = SpaceAnalyzer(
            d=cfg.d, tau=cfg.tau, w=cfg.w, ema_span=cfg.ema_span,
            lam_H=cfg.lam_H, lam_C=cfg.lam_C, fusion=cfg.latent_fusion,
            vote_quantile=cfg.vote_quantile, step=cfg.step,
        )

        # Determine window ends first (raw & latent share window geometry).
        from .windowing import window_end_indices
        ends = window_end_indices(T, cfg.w, cfg.step)
        train_mask = ends < split              # window fully inside training
        if train_mask.sum() < 5:
            raise ValueError(
                "Too few training windows; reduce w or increase train_ratio/T.")

        # -------- Steps 4-6: raw-space branch -> R_t --------
        if cfg.verbose:
            print(f"[4-6] raw-space H/C over {ends.size} windows "
                  f"(fusion={cfg.raw_fusion})")
        raw.fit(Xp, train_mask)
        R = raw.score                          # standardised on training

        # -------- Step 8: latent-space branch -> L_t --------
        if cfg.verbose:
            print(f"[8]   latent-space H/C (fusion={cfg.latent_fusion})")
        lat.fit(Z, train_mask)
        L = lat.score

        # -------- Step 9: dual-space fusion --------
        D = fuse_scores(R, L, alpha=cfg.alpha, gamma=cfg.gamma)

        # -------- Step 10: adaptive threshold (training windows only) --------
        thr = adaptive_threshold(D[train_mask], k=cfg.k,
                                 robust=cfg.robust_threshold)

        # -------- Step 11: drift decision on test windows --------
        test_mask = ~train_mask
        test_ends = ends[test_mask]
        test_D = D[test_mask]
        candidate_ends = test_ends[test_D > thr]

        if cfg.use_voting:
            drift_points = consolidate(candidate_ends, window=cfg.vote_window,
                                       method=cfg.vote_method)
        else:
            drift_points = [int(c) for c in candidate_ends]

        # -------- assemble result --------
        res = DualSpaceResult(
            T=T, p=p, latent_dim=latent_dim,
            split_index=split, train_ratio=cfg.train_ratio,
            ends=ends, train_mask=train_mask, test_mask=test_mask,
            R=R, L=L, D=D, threshold=thr,
            n_train_windows=int(train_mask.sum()),
            n_test_windows=int(test_mask.sum()),
            n_candidates=int(candidate_ends.size),
            candidate_ends=candidate_ends,
            drift_points=drift_points,
            ae_val_mse=getattr(ae, "best_val_mse_", None),
            # attribution diagnostics (do not affect the decision)
            raw_analyzer=raw, latent_analyzer=lat,
            preprocessor=pre, autoencoder=ae,
            config=cfg,
        )

        # -------- optional evaluation --------
        if ground_truth is not None:
            gt_test = [g for g in ground_truth if g >= split]  # evaluate on test span
            res.metrics = evaluate_detections(
                drift_points, gt_test, tolerance=cfg.tolerance,
                n_eval_points=int(test_mask.sum()),
            )
            res.ground_truth = list(ground_truth)
        return res
