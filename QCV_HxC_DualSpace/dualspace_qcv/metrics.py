"""
metrics.py
==========

Evaluation protocol matching the QCV-HxC paper: a detection counts as a true
positive if it falls within `tolerance` samples of a ground-truth drift point.
Each ground-truth point can be matched at most once (greedy nearest match).

Reported metrics: TP, FP, FN, precision, recall, F1, mean detection delay over
matched drifts, and a false-positive rate scaled by the number of evaluated
windows (comparable to the paper's FPR).
"""

from __future__ import annotations
import numpy as np


def evaluate_detections(detections, ground_truth, tolerance=500,
                        n_eval_points=None):
    det = sorted(int(d) for d in detections)
    gt = sorted(int(g) for g in ground_truth)

    matched_gt = set()
    tp = 0
    delays = []
    used_det = [False] * len(det)

    # Greedy: for each GT point, take the nearest unused detection within tol.
    for g in gt:
        best_j, best_dist = -1, None
        for j, d in enumerate(det):
            if used_det[j]:
                continue
            dist = abs(d - g)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_dist, best_j = dist, j
        if best_j >= 0:
            tp += 1
            used_det[best_j] = True
            matched_gt.add(g)
            delays.append(det[best_j] - g)   # signed: +ve = detected after drift

    fp = used_det.count(False)
    fn = len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    mean_delay = float(np.mean([abs(x) for x in delays])) if delays else float("nan")

    fpr = None
    if n_eval_points:
        fpr = fp / float(n_eval_points)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_delay": (round(mean_delay, 2) if mean_delay == mean_delay else None),
        "fpr": (round(fpr, 8) if fpr is not None else None),
        "delays": delays,
        "detections": det,
        "ground_truth": gt,
    }
