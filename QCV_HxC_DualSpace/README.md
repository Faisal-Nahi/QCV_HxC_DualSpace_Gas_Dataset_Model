# Dual-Space Entropy-Complexity Framework for Unsupervised Multivariate Concept-Drift Detection

A thesis-grade extension of **QCV-HxC** (Barbosa Costa, Frery, Aquino & Cavalcanti, *IEEE Access*, 2026) from **univariate** to **multivariate** data streams. The framework keeps the paper's information-theoretic core — permutation entropy `H` and Jensen–Shannon statistical complexity `C` computed from **ordinal patterns** — and adds:

1. a **second representation space** (an autoencoder latent space) analysed with the same entropy–complexity machinery;
2. **per-channel multivariate analysis** with configurable **cross-channel fusion** (mean / max / weighted / majority-vote) — the exact "indicator fusion" strategies the paper lists as future work;
3. a **dual-space score fusion** with an agreement term; and
4. an **adaptive baseline threshold** learned from the training window scores.

Everything is **unsupervised** (labels are used only for evaluation), **causal / leakage-free** (all baselines, scalers, the autoencoder, and the threshold are estimated on the first 70% only), and runs with **no deep-learning framework** — the autoencoder is implemented in pure NumPy.

---

## 1. What this upgrades relative to the base model

| Aspect              | Base QCV-HxC (paper / base model)           | This dual-space framework                                              |
| ------------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| Input               | Univariate stream                           | **Multivariate** stream (`p` features)                         |
| Feature interaction | Per-feature only                            | Per-channel**+ cross-channel fusion** (mean/max/weighted/vote)   |
| Representation      | Raw ordinal space                           | **Raw space + autoencoder latent space** (dual space)            |
| Decision signal     | Single derivative indicator                 | **Fused** `D_t = αR_t + (1−α)L_t + γ·min(R_t,L_t)`        |
| Thresholding        | Rolling quantile + voting                   | **Adaptive baseline** `μ_D + kσ_D` (+ optional voting)       |
| Complexity`C`     | Base model omitted the`Q₀` normalisation | **Fixed**: full Lamberti/Martin/Rosso `Q₀`, so `C ∈ [0,1]` |
| Protocol            | Online causal                               | Same, under an explicit**70% train / 30% test** split            |

The complexity fix matters: the base model's `statistical_complexity` left out the normalisation constant `Q₀`, so its `C` was not bounded on the `H × C` manifold. This package restores the proper formulation (see `entropy_complexity.py`).

Per the specification, the pipeline **ends at the drift decision** — there is no separate explainability module. (Per-channel and per-space contributions are still exposed on the result object as lightweight, decision-neutral diagnostics.)

---

## 2. Pipeline and the 11 development steps → code

```
Multivariate stream X (T × p)
      │  (1) preprocessing        preprocessing.py   MultivariatePreprocessor
      │  (2) chronological split  windowing.py       chronological_split (70/30)
      ▼
   causal sliding windows         windowing.py       window_end_indices
      │
      ├── RAW space ──────────────────────────────────────────────┐
      │   (3-6) per-channel ordinal H,C → deviations → r_j → fuse  │  spaces.py
      │        → standardised raw score R_t                        │  SpaceAnalyzer
      │                                                            │
      └── LATENT space ───────────────────────────────────────────┤
          (7) autoencoder  X → Z         autoencoder.py  Autoencoder│
          (8) per-dim ordinal H,C → … → standardised latent L_t     │  spaces.py
                                                                    ▼
      (9)  dual-space fusion   D_t = αR_t + (1−α)L_t + γ·min(R_t,L_t)   fusion.py fuse_scores
      (10) adaptive threshold  θ = μ_D + kσ_D   (training windows only)  fusion.py adaptive_threshold
      (11) decision  D_t > θ  → drift        detector.py DualSpaceDriftDetector
           (+ online voting)                 voting.py consolidate
                                             metrics.py evaluate_detections
```

| Step | Description                                                               | Module / symbol                            |
| ---- | ------------------------------------------------------------------------- | ------------------------------------------ |
| 1    | Missing-value handling, min-max normalisation, optional smoothing/detrend | `preprocessing.MultivariatePreprocessor` |
| 2    | Sequential 70/30 split (no shuffling)                                     | `windowing.chronological_split`          |
| 3    | Causal sliding windows of size`w`                                       | `windowing.window_end_indices`           |
| 4–6 | Raw-space per-channel`H`,`C` → baseline deviation → `R_t`         | `spaces.SpaceAnalyzer` (raw)             |
| 7    | Autoencoder`Z = f_θ(X)`, trained on the 70%                            | `autoencoder.Autoencoder`                |
| 8    | Latent-space per-dim`H`,`C` → `L_t`                                | `spaces.SpaceAnalyzer` (latent)          |
| 9    | Dual-space fusion`D_t`                                                  | `fusion.fuse_scores`                     |
| 10   | Adaptive threshold`μ_D + kσ_D`                                        | `fusion.adaptive_threshold`              |
| 11   | Drift decision + online voting                                            | `detector`, `voting`, `metrics`      |

---

## 3. The mathematics

**Ordinal patterns (Bandt–Pompe).** For a channel `x`, embedding dimension `d` and delay `τ`, each window position maps to the permutation that sorts `(x_i, x_{i+τ}, …, x_{i+(d−1)τ})`. Ties are broken with a deterministic **stable** argsort (equal values keep their time order), exactly as in the paper. The histogram has all `d!` bins (zeros for unobserved patterns) so complexity is unbiased.

**Permutation entropy** (Shannon entropy of the pattern histogram `p`, scaled to `[0,1]`):

```
H(p) = −(1 / ln N) · Σ_i p_i ln p_i ,      N = d!
```

**Jensen–Shannon statistical complexity** (Lamberti–Martin–Rosso):

```
C(p) = Q_J(p) · H(p) ∈ [0,1] ,   Q_J(p) = Q₀ · JS(p, u)
JS(p,u) = S((p+u)/2) − ½S(p) − ½S(u)          (natural-log Shannon S)
Q₀ = −2 / [ ((N+1)/N) ln(N+1) − 2 ln(2N) + ln(N) ]
```

with `u` the uniform distribution over the `d!` patterns. `Q₀` is the inverse of the maximum attainable JS divergence, so `Q_J ∈ [0,1]`. Both `H` and `C` are computed in a **vectorised, batched** form over all windows (`entropy_complexity.hxc_batch`) for speed.

**Per-channel drift score (each space).** For channel `j`, with EMA-smoothed `H_j`, `C_j` and training-baseline means `H_base,j`, `C_base,j`:

```
ΔH_j = |H_j − H_base,j| ,   ΔC_j = |C_j − C_base,j|
r_j  = λ_H · z(ΔH_j) + λ_C · z(ΔC_j)
```

where `z(·)` is a **robust** (median / 1.4826·MAD) standardisation whose statistics come from the **training windows only**. Standardising `ΔH` and `ΔC` before combining makes entropy and complexity deviations comparable.

**Cross-channel fusion → space score** (the paper's future-work strategies):

- `mean` — average of `r_j` (default);
- `max` — maximum aggregation (most sensitive channel drives the alarm);
- `weighted` — weighted sum, weights ∝ each channel's training-window spread;
- `vote` — fraction of channels whose `r_j` exceeds their own training quantile (majority voting).

The fused space score is then robustly standardised on training, yielding `R_t` (raw) and `L_t` (latent) on a common scale.

**Dual-space fusion (Step 9).**

```
D_t = α · R_t + (1 − α) · L_t + γ · min(R_t, L_t)
```

Because `R_t` and `L_t` are already standardised on training, `α` is a meaningful mixing weight and the **agreement term** `γ·min(R_t,L_t)` boosts confidence when *both* spaces flag a change and damps peaks where they disagree.

**Adaptive threshold (Step 10)** from the training-window fused scores:

```
θ = μ_D + k·σ_D            (Gaussian form, default)
θ = median_D + k·1.4826·MAD_D   (set robust_threshold=True)
```

**Decision (Step 11):** `D_t > θ ⇒ drift`, with candidate crossings consolidated by the online voter.

---

## 4. Key design decisions

- **Ordinal core, not generic Shannon.** The specification sketches Shannon entropy on raw values; we keep the paper's **ordinal-pattern** `H`/`C`. This preserves QCV-HxC's defining strengths — invariance to per-feature monotonic scaling and robustness to outliers — and keeps the extension faithful to its lineage.
- **Why the dual space is complementary.** Ordinal patterns are invariant to any monotonic per-feature transform, so the **raw** branch is tuned to shifts in *temporal dynamics* (autocorrelation, ordering, local structure). The **latent** branch runs the same `H`/`C` on autoencoder codes, which mix features, so it reacts to changes in the *joint distribution and cross-feature correlation* that are invisible to per-feature ordinal analysis. The ablation (below) shows each space alone misses a drift the other catches.
- **Causal, no leakage — everywhere.** The split is sequential; the autoencoder, all baselines, every robust scaler, and the threshold are estimated **only** on windows that end inside the training span. A window is "training" iff its end index `< split`.
- **Pure-NumPy autoencoder.** `p → [hidden] → k` (tanh hidden, linear bottleneck/output), mini-batch **Adam**, Glorot init, internal standardisation on training stats, early stopping on a causal validation tail. No PyTorch/TensorFlow needed; the backend can be swapped behind `encode`/`reconstruct`/`recon_error`.
- **Standardise before fusing.** Raw and latent scores live in different geometries; standardising both on training makes `α`, `γ`, and `min(·)` meaningful.
- **Multivariate fusion is configurable** precisely because the paper asks which strategy is best — and, as the experiments show, the answer is **regime-dependent** (no single strategy dominates both streams), which is itself a finding.

---

## 5. Package layout

```
QCV_HxC_DualSpace/
├── dualspace_qcv/
│   ├── __init__.py            # public API: DualSpaceConfig, DualSpaceDriftDetector, data_loader
│   ├── config.py              # DualSpaceConfig dataclass (all hyper-parameters)
│   ├── preprocessing.py       # Step 1  MultivariatePreprocessor
│   ├── windowing.py           # Steps 2-3  split + causal windows
│   ├── ordinal_patterns.py    # Bandt-Pompe ordinal encoding (vectorised codes())
│   ├── entropy_complexity.py  # H, C (with Q0) + batched hxc_batch()
│   ├── autoencoder.py         # Step 7  pure-NumPy autoencoder
│   ├── spaces.py              # Steps 4-6/8  per-space drift score (raw & latent)
│   ├── fusion.py              # Steps 9-10  fuse_scores + adaptive_threshold
│   ├── voting.py              # online consolidation of candidate alarms
│   ├── metrics.py             # tolerance-based P/R/F1/FPR/delay
│   ├── data_loader.py         # synthetic generators + load_csv for real data
│   └── detector.py            # DualSpaceDriftDetector.fit_detect (wires 1-11)
├── experiments/
│   ├── generate_synthetic.py  # write the two synthetic streams to data/
│   └── run_demo.py            # full demo: detect + ablation + fusion + k-sweep + figures
├── data/                      # (generated) synthetic CSVs + metadata; drop electricity.csv here
├── results/                   # (generated) figures + summary.json + report.txt
├── main.py                    # minimal end-to-end example
├── requirements.txt
└── README.md
```

---

## 6. Installation

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

Requires only NumPy, pandas, scikit-learn, SciPy, matplotlib, seaborn (Python ≥ 3.9).

---

## 7. Quickstart

Minimal example:

```bash
python main.py
```

Or from Python:

```python
from dualspace_qcv import DualSpaceConfig, DualSpaceDriftDetector, data_loader

df, gt = data_loader.make_multivariate_abrupt(n=12000, p=6, seed=0)

cfg = DualSpaceConfig(d=3, tau=1, w=100, alpha=0.5, gamma=0.25, k=3.0,
                      train_ratio=0.70, raw_fusion="mean", latent_fusion="mean")
res = DualSpaceDriftDetector(cfg).fit_detect(df.values, ground_truth=gt)

print(res.summary())
print(res.metrics)          # {'f1': 0.8, 'precision': 0.667, 'recall': 1.0, ...}
print(res.drift_points)     # consolidated detections in the test span
```

Reproduce all experiments and figures:

```bash
python experiments/generate_synthetic.py   # optional: save the streams to data/
python experiments/run_demo.py             # prints tables, writes results/
```

---

## 8. Using the real Electricity data

The framework reads any numeric multivariate CSV. To reproduce the paper's real-world setting, drop `electricity.csv` into `data/` and select the drift-sensitive channels the paper monitors (`nswdemand`, `vicdemand`, `transfer`):

```python
from dualspace_qcv import DualSpaceConfig, DualSpaceDriftDetector, data_loader

df = data_loader.load_csv("data/electricity.csv",
                          feature_columns=["nswdemand", "vicdemand", "transfer"])

# paper's pseudo-ground-truth reference regions (approx. centres)
gt = [28200, 29600, 32700]

cfg = DualSpaceConfig(d=3, tau=1, w=120, k=3.0, tolerance=500,
                      raw_fusion="mean", latent_fusion="mean")
res = DualSpaceDriftDetector(cfg).fit_detect(df.values, ground_truth=gt)
print(res.summary(), res.metrics)
```

The Electricity dataset is available from the MOA project (Waikato). Adjust `w`, `k`, and the fusion strategy following the recommended ranges in the paper (`w ∈ [80,120]`, `q/k` calibrated to the desired false-alarm rate).

---

## 9. Results (synthetic demo)

Two synthetic multivariate streams mirror the paper's two regimes: an **IID-style abrupt** stream (RTG_2abrupt analogue) and a **strongly autocorrelated IDT-style** stream (Electricity analogue). In both, the first 70% is a single stationary regime and drifts occur inside the test span. Metrics use a ±500-sample tolerance, matching the paper.

**Headline performance**

| Stream               | `T` | `p` | latent`k` | F1             | Precision | Recall | mean delay | FP |
| -------------------- | ----- | ----- | ----------- | -------------- | --------- | ------ | ---------- | -- |
| Abrupt (IID-style)   | 12000 | 6     | 3           | **0.80** | 0.67      | 1.00   | 264        | 1  |
| IDT (autocorrelated) | 15000 | 5     | 2           | **1.00** | 1.00      | 1.00   | 265        | 0  |

The abrupt-stream operating point (F1 = 0.80, P = 0.67, R = 1.00) reproduces the paper's headline result.

**Ablation — dual space vs each space alone** (identical threshold + voting on each score)

| Stream | raw-only F1  | latent-only F1 | **dual F1**      |
| ------ | ------------ | -------------- | ---------------------- |
| Abrupt | 0.67 (R=0.5) | 0.67 (R=0.5)   | **0.80 (R=1.0)** |
| IDT    | 1.00         | 1.00           | **1.00**         |

On the abrupt stream the raw branch catches one drift and the latent branch the other; **only the fused score recovers both** (recall 0.5 → 1.0). This is the central evidence that the two spaces are complementary.

**Cross-channel fusion strategy (F1)** — no single strategy dominates both regimes:

| Stream | mean           | max            | weighted       | vote |
| ------ | -------------- | -------------- | -------------- | ---- |
| Abrupt | 0.80           | **1.00** | 0.80           | 0.80 |
| IDT    | **1.00** | 0.50           | **1.00** | 0.67 |

`max` aggregation is best on the abrupt stream (a single strongly-shifting channel drives the alarm) but hurts the correlated IDT stream (it amplifies isolated noise); `mean`/`weighted` are best on IDT. This regime dependence is exactly the empirical question the paper flags for future work, and it argues for treating the fusion strategy as a tunable.

**Threshold sensitivity `k` (F1)**

| Stream | k=2.5 | k=3.0          | k=3.5          | k=4.0 |
| ------ | ----- | -------------- | -------------- | ----- |
| Abrupt | 0.80  | 0.80           | **1.00** | 0.00  |
| IDT    | 0.80  | **1.00** | 1.00           | 0.00  |

`k ≈ 3.0–3.5` is the sweet spot on both streams; `k = 4.0` is too strict and misses everything. The default is **`k = 3.0`**, which reproduces the paper's F1 = 0.80 operating point on the abrupt stream; raising it to `k = 3.5` removes the single boundary false positive and yields F1 = 1.0 on this demo.

**Figures written to `results/`** (per stream):

- `*_detection.png` — fused score `D_t`, adaptive threshold, ground-truth drifts, and TP/FP markers (Figure-7 analogue);
- `*_scores.png` — `R_t`, `L_t`, `D_t` stacked, showing dual-space complementarity;
- `*_ablation.png` — F1/precision/recall for raw vs latent vs dual;
- `*_hxc_plane.png` — the entropy-complexity plane (raw space), training vs post-drift windows.

Plus `results/summary.json` and `results/report.txt`.

---

## 10. Configuration reference (`DualSpaceConfig`)

Selected fields (see `config.py` for all, with paper-derived defaults):

| Field                                            | Default                        | Meaning                                                    |
| ------------------------------------------------ | ------------------------------ | ---------------------------------------------------------- |
| `normalization`                                | `"minmax"`                   | Step-1 scaling (`minmax`/`robust`/`zscore`/`none`) |
| `train_ratio`                                  | `0.70`                       | chronological split                                        |
| `d`, `tau`, `w`                            | `3`, `1`, `100`          | ordinal embedding dim, delay, window                       |
| `ema_span`                                     | `5`                          | causal EMA smoothing of`H`,`C`                         |
| `lam_H`, `lam_C`                             | `1.0`, `1.0`               | entropy vs complexity weights                              |
| `raw_fusion`, `latent_fusion`                | `"mean"`                     | cross-channel fusion strategy                              |
| `latent_dim`                                   | `None` → `max(2, p//2)`   | autoencoder bottleneck                                     |
| `alpha`, `gamma`                             | `0.5`, `0.25`              | dual-space mixing + agreement                              |
| `k`, `robust_threshold`                      | `3.0`, `False`             | adaptive threshold sensitivity                             |
| `use_voting`, `vote_window`, `vote_method` | `True`, `500`, `"first"` | online consolidation                                       |
| `tolerance`                                    | `500`                        | evaluation match tolerance                                 |

---

## 11. Limitations and scope

- Per the specification, there is **no explainability module**; the pipeline stops at the drift decision. Per-channel/per-space contributions are exposed only as diagnostics.
- The included datasets are **synthetic** (with a real-data path provided). They are constructed to exercise both branches; absolute numbers on real streams depend on preprocessing and reference construction.
- Computational cost scales **linearly in the number of channels** (each keeps its own fixed-size buffers), consistent with the paper's multivariate-extension analysis.

---

## 12. Reference

Barbosa Costa, K., Frery, A. C., Aquino, A., & Cavalcanti, G. D. C. (2026). *An Unsupervised Entropy-Complexity Framework for Real-Time Concept Drift Detection.* **IEEE Access**, 14. DOI: 10.1109/ACCESS.2026.3667170. Original univariate code: `github.com/keilabcs/Framework-QCV-HxC`.

This dual-space multivariate framework was built to realise, and empirically test, the multivariate extension and indicator-fusion strategies described in that paper's *Future Work*.
