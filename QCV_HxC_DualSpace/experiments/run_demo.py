"""
run_demo.py
===========

End-to-end demonstration and evaluation of the Dual-Space Entropy-Complexity
framework for unsupervised multivariate concept-drift detection.

For each synthetic dataset (an IID-style abrupt stream and an IDT-style
autocorrelated stream) the script:

    1. runs the full dual-space detector and prints its summary + metrics;
    2. runs an ABLATION (raw-space only, latent-space only, dual space) to show
       that the two spaces are complementary and that fusing them beats either
       one alone -- this is the central claim of the thesis extension;
    3. compares the four multivariate channel-fusion strategies proposed as
       future work in the paper (mean, max, weighted, vote);
    4. sweeps the threshold sensitivity k;
    5. writes publication-style figures to ``results/`` (a Figure-7 style
       detection panel, an R/L/D score panel, an ablation bar chart, and an
       entropy-complexity plane), plus a JSON + text report.

Run from the package root:

    python experiments/run_demo.py

Everything is unsupervised: ground-truth drift points are used only to score
detections, never inside the detector.
"""

from __future__ import annotations
import json
import os
import sys

import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# make the package importable when run as a plain script from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dualspace_qcv import (  # noqa: E402
    DualSpaceConfig, DualSpaceDriftDetector, data_loader,
)
from dualspace_qcv.fusion import adaptive_threshold  # noqa: E402
from dualspace_qcv.voting import consolidate          # noqa: E402
from dualspace_qcv.metrics import evaluate_detections  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Ablation helper: turn a single-space score into evaluated detections        #
# --------------------------------------------------------------------------- #
def score_to_detections(score, ends, train_mask, cfg, ground_truth, split):
    """
    Apply the SAME Step 10-11 machinery the detector uses for the fused score
    (adaptive threshold on training windows -> decision on test windows ->
    online voting) to an arbitrary per-window score. Used to evaluate the
    raw-only and latent-only branches on an identical footing with the dual
    score.
    """
    thr = adaptive_threshold(score[train_mask], k=cfg.k,
                             robust=cfg.robust_threshold)
    test_mask = ~train_mask
    test_ends = ends[test_mask]
    cand = test_ends[score[test_mask] > thr]
    if cfg.use_voting:
        pts = consolidate(cand, window=cfg.vote_window, method=cfg.vote_method)
    else:
        pts = [int(c) for c in cand]
    gt_test = [g for g in ground_truth if g >= split]
    metrics = evaluate_detections(pts, gt_test, tolerance=cfg.tolerance,
                                  n_eval_points=int(test_mask.sum()))
    return thr, pts, metrics


def _fmt(m):
    return (f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"delay={m['mean_delay']}  FP={m['FP']}  det={m['detections']}")


# --------------------------------------------------------------------------- #
#  Plots                                                                        #
# --------------------------------------------------------------------------- #
def plot_detection_panel(res, name, title):
    """Figure-7 style: fused score D_t, threshold, GT lines, TP/FP markers."""
    ends, D = res.ends, res.D
    thr = res.threshold
    gt = res.ground_truth
    dets = res.drift_points
    tol = res.config.tolerance

    tp = [d for d in dets if any(abs(d - g) <= tol for g in gt)]
    fp = [d for d in dets if d not in tp]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(ends, D, lw=0.8, color="#1f77b4",
            label="Dual-space score $D_t$")
    ax.axhline(thr, ls="--", color="#ff7f0e", lw=1.3,
               label=f"Adaptive threshold ($\\mu+{res.config.k:g}\\sigma$)")
    ax.axvspan(ends.min(), res.split_index, color="0.9", zorder=0,
               label="Training span (70%)")
    for i, g in enumerate(gt):
        ax.axvline(g, ls=(0, (4, 3)), color="green", lw=1.4,
                   label="Ground-truth drift" if i == 0 else None)
    for i, d in enumerate(tp):
        ax.axvline(d, color="blue", lw=1.6,
                   label="True positive" if i == 0 else None)
    for i, d in enumerate(fp):
        ax.axvline(d, ls=":", color="red", lw=1.6,
                   label="False positive" if i == 0 else None)
    ax.set_xlabel("Instance $t$")
    ax.set_ylabel("Dual-space drift score")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_detection.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_score_panel(res, name, title):
    """Show R_t, L_t and D_t together to visualise dual-space complementarity."""
    ends = res.ends
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax, sig, lab, col in zip(
            axes, [res.R, res.L, res.D],
            ["Raw-space score $R_t$", "Latent-space score $L_t$",
             "Fused score $D_t$"],
            ["#2ca02c", "#9467bd", "#1f77b4"]):
        ax.plot(ends, sig, lw=0.8, color=col)
        ax.axvspan(ends.min(), res.split_index, color="0.92", zorder=0)
        for g in res.ground_truth:
            ax.axvline(g, ls=(0, (4, 3)), color="green", lw=1.1)
        ax.set_ylabel(lab, fontsize=9)
    axes[-1].axhline(res.threshold, ls="--", color="#ff7f0e", lw=1.2)
    axes[-1].set_xlabel("Instance $t$")
    axes[0].set_title(title)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_scores.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_ablation(ablation, name, title):
    """Grouped bar chart of F1/precision/recall for raw / latent / dual."""
    branches = ["raw", "latent", "dual"]
    metrics = ["f1", "precision", "recall"]
    vals = np.array([[ablation[b][m] for m in metrics] for b in branches])
    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#2ca02c", "#9467bd", "#1f77b4"]
    for i, b in enumerate(branches):
        ax.bar(x + (i - 1) * w, vals[i], w, label=b, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["F1", "Precision", "Recall"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(title="Space")
    for i, b in enumerate(branches):
        for j, v in enumerate(vals[i]):
            ax.text(x[j] + (i - 1) * w, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_ablation.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_hxc_plane(res, name, title):
    """
    Entropy-complexity plane for the raw space: each point is one window
    (mean H, mean C across channels), coloured by whether it is a training
    window or a post-drift test window. Visualises the core QCV-HxC measure.
    """
    an = res.raw_analyzer
    H = an.H_.mean(axis=1)
    C = an.C_.mean(axis=1)
    ends = res.ends
    gt = res.ground_truth
    first_drift = min(gt) if gt else res.split_index

    is_train = res.train_mask
    is_post = (~is_train) & (ends >= first_drift)
    is_pre_test = (~is_train) & (ends < first_drift)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(H[is_train], C[is_train], s=10, alpha=0.5,
               color="#1f77b4", label="training windows")
    ax.scatter(H[is_pre_test], C[is_pre_test], s=10, alpha=0.5,
               color="#7f7f7f", label="test, pre-drift")
    ax.scatter(H[is_post], C[is_post], s=12, alpha=0.6,
               color="#d62728", label="test, post-drift")
    ax.set_xlabel("Permutation entropy $H$")
    ax.set_ylabel("Statistical complexity $C$")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_hxc_plane.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
#  Per-dataset experiment                                                       #
# --------------------------------------------------------------------------- #
def run_dataset(name, title, df, gt, base_cfg):
    X = df.values
    p = X.shape[1]
    print("\n" + "=" * 70)
    print(f"DATASET: {title}   (T={X.shape[0]}, p={p})   GT drifts={gt}")
    print("=" * 70)

    # ---- 1. full dual-space detector ----
    det = DualSpaceDriftDetector(base_cfg)
    res = det.fit_detect(X, ground_truth=gt)
    print(res.summary())
    print("\n[dual-space]  " + _fmt(res.metrics))

    split = res.split_index
    ends = res.ends
    tmask = res.train_mask

    # ---- 2. ablation: raw-only, latent-only, dual ----
    _, raw_pts, raw_m = score_to_detections(res.R, ends, tmask, base_cfg,
                                            gt, split)
    _, lat_pts, lat_m = score_to_detections(res.L, ends, tmask, base_cfg,
                                            gt, split)
    print("\nAblation (identical threshold + voting on each score):")
    print(f"  raw-only    : " + _fmt(raw_m))
    print(f"  latent-only : " + _fmt(lat_m))
    print(f"  dual        : " + _fmt(res.metrics))
    ablation = {"raw": raw_m, "latent": lat_m, "dual": res.metrics}

    # ---- 3. channel-fusion strategy comparison ----
    print("\nChannel-fusion strategy comparison (dual space):")
    fusion_results = {}
    for strat in ("mean", "max", "weighted", "vote"):
        cfg = DualSpaceConfig(**{**base_cfg.__dict__,
                                 "raw_fusion": strat, "latent_fusion": strat})
        r = DualSpaceDriftDetector(cfg).fit_detect(X, ground_truth=gt)
        fusion_results[strat] = r.metrics
        print(f"  {strat:8s}: " + _fmt(r.metrics))

    # ---- 4. k-sensitivity sweep (reuses the fused score; only the
    #         threshold depends on k, so no need to re-run the pipeline) ----
    print("\nThreshold-sensitivity sweep (k):")
    k_results = {}
    for k in (2.5, 3.0, 3.5, 4.0):
        cfg_k = DualSpaceConfig(**{**base_cfg.__dict__, "k": k})
        _, _, mk = score_to_detections(res.D, ends, tmask, cfg_k, gt, split)
        k_results[f"{k:g}"] = mk
        print(f"  k={k:<4g}: " + _fmt(mk))

    # ---- 5. figures ----
    figs = {
        "detection": plot_detection_panel(
            res, name, f"Drift detection - {title}"),
        "scores": plot_score_panel(
            res, name, f"Dual-space scores - {title}"),
        "ablation": plot_ablation(
            ablation, name, f"Space ablation - {title}"),
        "hxc_plane": plot_hxc_plane(
            res, name, f"Entropy-complexity plane (raw) - {title}"),
    }
    for kind, path in figs.items():
        print(f"  figure[{kind}] -> {os.path.relpath(path, _ROOT)}")

    return {
        "name": name,
        "title": title,
        "T": int(X.shape[0]),
        "p": int(p),
        "latent_dim": int(res.latent_dim),
        "split_index": int(split),
        "ground_truth": [int(g) for g in gt],
        "threshold": float(res.threshold),
        "dual_metrics": res.metrics,
        "ablation": ablation,
        "fusion_comparison": fusion_results,
        "k_sensitivity": k_results,
        "figures": {k: os.path.relpath(v, _ROOT) for k, v in figs.items()},
    }


# --------------------------------------------------------------------------- #
#  Report writers                                                              #
# --------------------------------------------------------------------------- #
def _clean(obj):
    """Make metrics JSON-serialisable (drop bulky arrays / numpy types)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()
                if k not in ("delays",)}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_report(all_results):
    json_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(_clean(all_results), fh, indent=2)

    txt_path = os.path.join(RESULTS_DIR, "report.txt")
    lines = ["Dual-Space Entropy-Complexity Framework - experiment report",
             "=" * 62, ""]
    for r in all_results:
        m = r["dual_metrics"]
        lines += [
            f"## {r['title']}",
            f"   stream            : T={r['T']}, p={r['p']}, "
            f"latent_dim={r['latent_dim']}",
            f"   ground-truth      : {r['ground_truth']}",
            f"   dual-space        : F1={m['f1']:.3f}  P={m['precision']:.3f}  "
            f"R={m['recall']:.3f}  delay={m['mean_delay']}  FP={m['FP']}",
            f"   ablation F1       : raw={r['ablation']['raw']['f1']:.3f}  "
            f"latent={r['ablation']['latent']['f1']:.3f}  "
            f"dual={r['ablation']['dual']['f1']:.3f}",
            "   fusion F1         : " + "  ".join(
                f"{k}={v['f1']:.3f}" for k, v in r["fusion_comparison"].items()),
            "   k-sensitivity F1  : " + "  ".join(
                f"k{k}={v['f1']:.3f}" for k, v in r["k_sensitivity"].items()),
            "",
        ]
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport written: {os.path.relpath(json_path, _ROOT)}, "
          f"{os.path.relpath(txt_path, _ROOT)}")
    return json_path, txt_path


# --------------------------------------------------------------------------- #
def main():
    base_cfg = DualSpaceConfig(
        normalization="minmax", d=3, tau=1, w=100, ema_span=5,
        alpha=0.5, gamma=0.25, k=3.0, train_ratio=0.70,
        vote_window=500, vote_method="first", tolerance=500,
        verbose=False,
    )

    all_results = []

    # IID-style abrupt stream (RTG_2abrupt analogue)
    df_a, gt_a = data_loader.make_multivariate_abrupt(
        n=12000, p=6, train_ratio=0.70, seed=0)
    all_results.append(run_dataset(
        "synthetic_abrupt", "Synthetic abrupt (IID-style)", df_a, gt_a,
        base_cfg))

    # IDT-style autocorrelated stream (Electricity analogue)
    df_i, gt_i = data_loader.make_multivariate_idt(
        n=15000, p=5, train_ratio=0.70, seed=1)
    all_results.append(run_dataset(
        "synthetic_idt", "Synthetic IDT (autocorrelated)", df_i, gt_i,
        base_cfg))

    write_report(all_results)
    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
