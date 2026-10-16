"""
generate_synthetic.py
======================

Generate the two synthetic multivariate streams used in the demo and save them
to ``data/`` as CSVs, together with a JSON sidecar recording the ground-truth
drift indices and the train/test split.

These streams are analogues of the two regimes studied in the QCV-HxC paper:

    * ``synthetic_abrupt``  -- an IID-style stream with abrupt regime changes
      (analogue of the paper's RTG_2abrupt benchmark);
    * ``synthetic_idt``     -- a strongly autocorrelated (IDT-style) stream with
      abrupt regime changes (analogue of the temporally dependent Electricity
      stream).

In both, the first 70% is a single stationary "normal" regime and every drift
falls inside the last 30% (the test span), which is exactly the protocol the
thesis model is evaluated under.

Run from the package root:

    python experiments/generate_synthetic.py
"""

from __future__ import annotations
import json
import os
import sys

# make the package importable when run as a plain script from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dualspace_qcv import data_loader  # noqa: E402


DATA_DIR = os.path.join(_ROOT, "data")


def _save(name, df, drift_points, train_ratio):
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")
    meta_path = os.path.join(DATA_DIR, f"{name}.meta.json")
    df.to_csv(csv_path, index=False)
    meta = {
        "name": name,
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1]),
        "columns": list(df.columns),
        "train_ratio": train_ratio,
        "split_index": int(round(train_ratio * len(df))),
        "ground_truth_drifts": [int(d) for d in drift_points],
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  wrote {csv_path}  ({df.shape[0]}x{df.shape[1]}), "
          f"drifts={drift_points}")
    return csv_path, meta_path


def build_all():
    print("Generating synthetic multivariate streams ...")

    df_a, gt_a = data_loader.make_multivariate_abrupt(
        n=12000, p=6, train_ratio=0.70, seed=0)
    _save("synthetic_abrupt", df_a, gt_a, 0.70)

    df_i, gt_i = data_loader.make_multivariate_idt(
        n=15000, p=5, train_ratio=0.70, seed=1)
    _save("synthetic_idt", df_i, gt_i, 0.70)

    print("Done.")


if __name__ == "__main__":
    build_all()
