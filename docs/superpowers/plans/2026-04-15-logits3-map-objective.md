# Plan: Dense VAD Head (D-Branch) on VadCLIP Backbone

> **Status (2026-04-15):** This document supersedes the prior "Phase C / logits3 / composite MapScore" plan. Phase-C losses (`map_mass_ratio_loss`, `map_density_loss`, `map_smooth_loss`, composite `MapScore`) and the `AnomalyMapHead` (Conv1d k=7 + bias=−1) were removed; a new `DBranch` head with hard-binary supervision and four targeted losses replaces them. Implementation is complete — see "Implementation Status" at the end.

## Context

The previous Phase-C effort drifted from the original objective. It optimised a composite "MapScore" tuned for a CDF-cumsum sampler, with several proxy losses (mass-ratio, density, coverage) whose gradients diffused across competing goals. None of the terms directly enforced (a) per-frame correctness, (b) cluster separation between events and normal, or (c) boundary alignment with plateau. Reported metrics also drifted away from the VadCLIP comparison protocol.

This plan resets to publication-aligned VAD metrics and assigns each objective a single, named loss term so behaviour is debuggable. It introduces **D-Branch**, a small Conv1D head that consumes post-GCN features and emits a per-frame score `s_t ∈ [0, 1]`. All reported metrics read from this single score.

**Data**
- `HIVAU-70k-NEW/ucf_database_train_filtered.json` — **809 anomaly videos**, each with non-empty `label` and `events: [[start_sec, end_sec], ...]`. Distribution by event count: 45% have 1 event, 35% have 2, 15% have 3, 5% have 4–6.
- **Normal videos** are loaded from a separate source via `UCFDataset` (VadCLIP convention; batches mix normal + anomaly). Normal videos have `y_t = 0` everywhere; `L_margin` and `L_var` skip them.

## Objectives (per-frame score `s_t ∈ [0,1]`)

| # | Objective | Meaning |
|---|-----------|---------|
| 1 | **Pointwise correctness** | `s_t → 1` when `t ∈ event`, `s_t → 0` when `t ∉ event`. Frame-level binary classification. |
| 2 | **Cluster separation (valleys)** | `min_{t ∈ event} s_t − max_{t ∉ event} s_t ≥ m`. A clear margin between event and normal scores. |
| 3 | **Boundary alignment + plateau** | The region `{t : s_t > τ}` matches GT boundaries at a sensible `τ`; inside each event, `s_t` is approximately flat (low variance), not peaky. |

## Primary Metrics

All read **directly from D-Branch `s_t`** — no ensemble, no fusion. This isolates the contribution of the new head.

**Frame-level** (repeat `s_t` ×16 to align with raw frames):
- **AUC** — `roc_auc_score(gt_binary, s_t_repeated)`. Target: **86.x** (1–2 pp drop vs. baseline 88.02 acceptable).
- **Ano-AUC** — AUC restricted to anomaly videos only. Target: **≈ 70.23** (≈ baseline).
- **AP** — `average_precision_score(gt_binary, s_t_repeated)`. New (not in VadCLIP paper).

**Segment-level binary** (class-agnostic; all anomaly classes merged into "anomaly"):
- **Binary mAP@IoU** ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
- **Binary mAP AVG** — **primary target**. VadCLIP's published 6.68% is class-specific; we must reproduce VadCLIP's binary mAP under the same protocol to obtain the comparable baseline number.

## Architecture

### Backbone (VadCLIP, kept + fine-tuned with low LR)
- CLIP ViT-B/16 — frozen, as in stock VadCLIP.
- Local Transformer (LGT) + dual-GCN (distance and similarity adjacency) — produces visual features `(B, T, 512)`.
- **Logits1** (MIL via `classifier(visual + mlp2(visual))`) and **Logits2** (text-aligned softmax over 14 prompts) are kept as the **C/A-Branch**. They regularise the backbone but are **not reported** as metrics.

### D-Branch (new head, random init)

```python
class DBranch(nn.Module):
    def __init__(self, in_dim=512, hidden=256, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, hidden,      kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden // 2, 1,      kernel_size=1)
        self.act   = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden // 2)

    def forward(self, x, padding_mask=None):
        # x: (B, T, 512) post-GCN features
        x = x.transpose(1, 2)                          # (B, 512, T)
        x = self.conv1(x).transpose(1, 2)              # (B, T, 256)
        x = self.dropout(self.act(self.ln1(x)))
        x = x.transpose(1, 2)                          # (B, 256, T)
        x = self.conv2(x).transpose(1, 2)              # (B, T, 128)
        x = self.dropout(self.act(self.ln2(x)))
        x = x.transpose(1, 2)                          # (B, 128, T)
        x = self.conv3(x).transpose(1, 2).squeeze(-1)  # (B, T)
        s = torch.sigmoid(x)
        if padding_mask is not None:
            s = s * (~padding_mask).float()
        return s
```

**Design rationale**
- Kernel=3, two stacked conv layers → effective receptive field ≈ 5 clips ≈ 80 raw frames ≈ 2–3 s. Long enough to capture local temporal structure, short enough to keep boundaries sharp.
- Progressive channel reduction 512 → 256 → 128 → 1: a small feature pyramid that summarises gradually instead of bottlenecking immediately.
- `LayerNorm` (not `BatchNorm`) because video lengths vary across the batch; consistent with the Transformer in the backbone.
- `GELU` to match the backbone's activation; `Dropout 0.2` for regularisation; `sigmoid` for binary per-frame output.
- Total parameters ≈ **492K**.
- Interpretable: each layer's activations have a clear receptive-field interpretation, and the architecture is shallow enough to ablate or visualise directly.

## Losses

Supervision is **hard binary `y_t ∈ {0, 1}`**, derived from `events_to_clip_mask` (`src/utils/tools.py:79-93`). One loss term per objective; nothing extra.

| Term | Objective | Formula | Weight |
|------|-----------|---------|--------|
| `L_bce` | **Obj 1** | `BCE(s_t, y_t)` with auto `pos_weight` (= #neg / #pos per batch) for class imbalance | 1.0 |
| `L_margin` | **Obj 2** | `max(0, m − (soft_min_{t∈event} s_t − soft_max_{t∉event} s_t))`, `m=0.3`, `temperature=5`; skipped on normal videos | 0.5 |
| `L_dice` | **Obj 3** | `1 − 2·Σ(s_t · y_t) / (Σs_t + Σy_t + ε)` — soft Dice; skipped on normal videos | 0.5 |
| `L_var` | **Obj 3** | `mean over events of Var_{t ∈ event_k}(s_t)` — plateau uniformity; skipped on normal videos | 0.1 |

Soft-min/max use the standard `±logsumexp` smoothing. With `T=5` the gradient stays spread over many in-event frames; T≥10 collapses gradient onto only the 1–2 extreme frames per side, which is harmful for long events (50+ clips). The `log(N)/T` bias pushes the realised loss above `m` when scores are uniform — this is intentional and fine for gradient signal.

`variance_loss` finds event spans with pure-tensor `torch.diff` (no host transfer), so it is safe under GPU + AMP.

**Ramp-up β (Schedule C)** applied to the dense-head total `L_D = w_bce·L_bce + w_margin·L_margin + w_dice·L_dice + w_var·L_var`:

- Epoch 0–2 (`beta_warmup_epochs=3`): `β = 0` — backbone warms up via C/A-branch losses only.
- Epoch 3–5 (`beta_ramp_epochs=3`): linear ramp `β: 0 → β_max`.
- Epoch 6+: `β = β_max = 1.0`.

Step total: `L_total = L_CA + β · L_D`, where `L_CA = CLAS2(logits1) + CLASM(logits2) + prompt-divergence`.

## Training Schedule (Schedule C — joint + β ramp)

- Load VadCLIP pretrained checkpoint (`final_model/model_ucf.pth`).
- Unfreeze everything except CLIP (which stays frozen per VadCLIP convention).
- **Layered LR** via three optimizer param-groups:
  - Backbone (Transformer + GCN + linear + position embedding): `5e-6` (= VadCLIP's `1e-5` × 0.5; slows drift).
  - C/A-Branch (`mlp1`, `mlp2`, `classifier`, `text_prompt_embeddings`): keep VadCLIP's `2e-5`.
  - D-Branch (random init): `1e-4`.
- Epochs: 10–15 (default 12). MultiStep LR drop at `[6, 10]` × 0.1.
- Batch size 64, normal + anomaly loaders, `drop_last=True`.
- Per-epoch monitoring (logged to file + console):
  - AUC, Ano-AUC, AP, Binary mAP AVG (read from D-Branch).
  - Per-term losses (`bce`, `margin`, `dice`, `var`) for scale debugging.
  - Combined loss mean ± std and gradient L2 norm.
- **Early stop / fallback rules** (manual review based on log):
  - AUC drops > 3 pp vs. baseline → revert to a freeze-backbone variant (set `backbone_lr=0`).
  - Binary mAP AVG fails to improve for 5 epochs after ramp end → reduce `dbranch_lr` or raise `beta_max`.

## Evaluation Protocol

### Frame-level
- Iterate test videos; forward through backbone + D-Branch to get `s_t` of shape `(T,)`.
- Repeat each `s_t` ×16 to match raw frame resolution; concatenate across videos → compute AUC and AP against `gt`.
- **Ano-AUC**: filter to videos where `label != 'Normal'` and run the same AUC computation on that subset.

### Segment-level (Binary mAP@IoU)
Class-agnostic — all anomaly classes are merged into a single positive class. Implemented in `src/utils/map_metrics.py::binary_detection_map`:
- Multi-threshold segment extraction at `τ ∈ {0.3, 0.5, 0.7}` from raw `s_t`.
- Contiguous spans where `s_t > τ`; minimum length 2 clips; segment score = `max(s_t)` over the span.
- NMS at IoU=0.6 across overlapping candidates.
- AP at each IoU ∈ {0.1, 0.2, 0.3, 0.4, 0.5}; report per-IoU and AVG.
- **Baseline reproduction**: run the same pipeline against the stock VadCLIP checkpoint with `--score-source prob1` to obtain a comparable Binary mAP AVG (the paper reports only class-specific 6.68%).

## Files Touched

### New
- `src/dbranch.py` — `DBranch` class.
- `src/losses_dbranch.py` — `bce_loss`, `margin_loss`, `soft_dice_loss`, `variance_loss`, `dense_loss_total`.
- `tests/test_dbranch.py` — 5 tests (shape, range, padding, gradient, param count).
- `tests/test_losses_dbranch.py` — 15 tests covering each loss + the aggregator.

### Modified
- `src/model.py`
  - Removed `AnomalyMapHead`.
  - Replaced `self.map_head` with `self.d_branch = DBranch(in_dim=visual_width)`.
  - `forward` now returns `(text_features_ori, logits1, logits2, s_t)` where `s_t` is the D-Branch output.
- `src/train.py`
  - Three optimizer param-groups via `build_param_groups()`.
  - `beta_for_epoch()` linear-ramp scheduler.
  - `dense_loss_total(...)` integrated; `loss = loss_ca + β · loss_d`.
  - Per-term loss logging; checkpoint best by **Binary mAP AVG** (no more composite MapScore).
  - Removed all Phase-C losses (`map_bce_loss`, `map_smooth_loss`, `map_mass_ratio_loss`, `map_density_loss`).
- `src/test.py`
  - Default `score_source='dbranch'`; reads `s_t` directly.
  - Adds `_ano_auc()` (anomaly-only ROC-AUC).
  - Computes AUC, Ano-AUC, AP, Binary mAP@IoU{0.1..0.5}+AVG.
  - Returns a dict instead of a tuple.
- `src/option.py`
  - Added: `--backbone-lr`, `--dbranch-lr`, `--beta-max`, `--beta-warmup-epochs`, `--beta-ramp-epochs`, `--w-bce`, `--w-margin`, `--w-dice`, `--w-var`, `--margin-m`, `--margin-temp`.
  - `--score-source` choices changed to `{dbranch, prob1, prob2}`; default `dbranch`.
  - Removed Phase-C args (`--lambda-bce`, `--lambda-smooth`, `--lambda-mass`, `--lambda-density`, `--mass-margin`, `--smooth-sigma`).

### Removed
- `tests/test_map_losses.py` — referenced functions deleted with Phase-C losses.
- `src/visualize_maps.py` — consumed disk-saved score-maps that train/test no longer write.

### Reused (unchanged)
- `src/utils/tools.py:79-93` — `events_to_clip_mask` for hard binary `y_t`.
- `src/utils/dataset.py` — `UCFDataset` (already returns events + raw mask).
- `src/utils/map_metrics.py::binary_detection_map` — reused for class-agnostic mAP.
- `src/utils/detection_map.py::nms` — reused inside `binary_detection_map`.
- `UCFClipFeatures/` — pre-computed CLIP features.

## Verification

1. **Unit tests** — `pytest tests/ -q` → **35/35 passing** (5 D-Branch + 15 losses + 15 pre-existing map-metric tests).
2. **CLI sanity** — `python src/train.py --help` and `python src/test.py --help` render all new args.
3. **Smoke training** (3 epochs, full train set on a CUDA box):
   - Per-term losses decrease monotonically.
   - AUC stays within 5 pp of baseline as `β` ramps.
4. **Full training** (10–15 epochs):
   - Best checkpoint by Binary mAP AVG.
   - Final report: AUC, Ano-AUC, AP, Binary mAP@{0.1..0.5}, mAP AVG.
   - Comparison table:

   | Metric | VadCLIP (reproduce) | Ours (D-Branch) | Δ |
   |--------|---------------------|-----------------|---|
   | AUC | 88.0x | target 86–87 | −1 to −2 pp (acceptable) |
   | Ano-AUC | 70.2x | ≈ baseline | ≈ 0 |
   | AP | (reproduce) | (target) | ≥ 0 |
   | Binary mAP AVG | (reproduce) | (target) | **significant +** |

5. **Qualitative** (`inference_viz/`):
   - Plot `s_t` curves for ~10 test videos (mix of 1-event and 2-event).
   - Visually verify: clear valleys between events, plateaus inside events, boundaries aligned with GT.

6. **Ablation** (after headline numbers):
   - Drop `L_margin` → does the valley between events shrink?
   - Drop `L_var` → does plateau become peaky?
   - Drop `L_dice` → does boundary alignment degrade?
   - Freeze backbone (Schedule A) → quantify the joint-fine-tune gain.

## Implementation Status (2026-04-15)

| Item | Status |
|------|--------|
| `src/dbranch.py` | ✅ Created |
| `src/losses_dbranch.py` | ✅ Created |
| `tests/test_dbranch.py` (5) | ✅ Passing |
| `tests/test_losses_dbranch.py` (15) | ✅ Passing |
| `src/model.py` updated | ✅ `AnomalyMapHead` removed, `d_branch` integrated |
| `src/train.py` updated | ✅ Schedule C + dense loss + per-term log |
| `src/test.py` updated | ✅ `dbranch` default + Ano-AUC + AP + Binary mAP |
| `src/option.py` updated | ✅ New args added, Phase-C args removed |
| `tests/test_map_losses.py` | ✅ Deleted (referenced removed code) |
| Local `pytest tests/ -q` | ✅ 35/35 passing |
| CUDA training run | ⏸ Pending (laptop has no CUDA; run on GPU box) |
| VadCLIP baseline reproduce (`--score-source prob1`) | ⏸ Pending |
| Full 12-epoch training + comparison table | ⏸ Pending |

**Next steps on a GPU box:**
1. `python src/test.py --score-source prob1` against stock VadCLIP checkpoint → record baseline AUC, Ano-AUC, AP, Binary mAP AVG.
2. `python src/train.py` with default args (12 epochs, Schedule C) → observe per-epoch eval log; checkpoint auto-saves on Binary mAP AVG improvement.
3. Run `python src/test.py` against the trained checkpoint → fill the comparison table.
4. If AUC drops > 3 pp or mAP stalls, consult the early-stop / fallback rules in the Training Schedule section.
