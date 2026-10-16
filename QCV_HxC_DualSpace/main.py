"""
main.py
=======

Minimal end-to-end demo of the Dual-Space Entropy-Complexity multivariate drift
detector on a synthetic multivariate stream with known abrupt drifts.

Run:
    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dualspace_qcv import DualSpaceDriftDetector, DualSpaceConfig, data_loader


def main():
    # --- synthetic multivariate stream with ground-truth drifts ---
    # drift_points=None => two drifts placed inside the test span (last 30%),
    # so the 70% training portion is a single stationary "normal" regime.
    df, ground_truth = data_loader.make_multivariate_abrupt(
        n=12000, p=6, drift_points=None, train_ratio=0.70, seed=0)
    X = df.values
    print(f"Loaded stream: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Ground-truth drift points: {ground_truth}\n")

    # --- configure & run ---
    cfg = DualSpaceConfig(
        d=3, tau=1, w=100,
        alpha=0.5, gamma=0.25, k=3.0,
        raw_fusion="mean", latent_fusion="mean",
        use_voting=True, vote_window=500, vote_method="first",
        tolerance=500, verbose=True,
    )
    detector = DualSpaceDriftDetector(cfg)
    result = detector.fit_detect(X, ground_truth=ground_truth)

    print("\n" + result.summary())
    print("\nEvaluation (test span):")
    for kk, vv in result.metrics.items():
        if kk not in ("delays", "detections", "ground_truth"):
            print(f"  {kk:12s}: {vv}")
    print(f"  detections  : {result.metrics['detections']}")
    print(f"  ground truth: {result.metrics['ground_truth']}")


if __name__ == "__main__":
    main()
