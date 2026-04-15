# Logits3 Map-Quality Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-frame logits3 training from "beat AUC" to "produce a dense, calibrated anomaly distribution s_t that feeds the CDF-cumsum keyframe sampler."

**Architecture:** Keep the existing AnomalyMapHead (Conv1d 512→256→1, kernel=7). Replace max-based coverage/ranking losses (which reward spikes) with mass-ratio + density losses. Add a binary localization mAP@IoU plus 6 map-quality metrics as the new primary eval suite; save checkpoints by composite MapScore instead of AUC1. Keep logits1 (MIL) and logits2 (text-alignment) unchanged — AUC1 becomes a regression guard, logits2 remains the semantic branch.

**Tech Stack:** PyTorch, NumPy, scikit-learn (AUC/AP), existing `src/utils/detection_map.py` NMS+IoU code (reused for binary mAP).

---

## Context

### The four objectives for s_t

| # | Objective | What it means for the map |
|---|---|---|
| 1 | **Separation** | s_t high inside events, low outside |
| 2 | **Mass concentration** | Σ s_t mass sits mostly inside events (so cumsum zooms in) |
| 3 | **Density (no spike)** | Inside events, s_t is evenly elevated — not one peak |
| 4 | **Semantic retained** | Logits2 still identifies class (needed for downstream description) |

### Objective ↔ Metric ↔ Loss mapping

| Objective | Primary metric(s) | Loss driving it |
|---|---|---|
| 1 Separation | Gap, Normal-mean | `map_bce_loss` (existing) |
| 2 Mass | EMR, MCL | `map_mass_ratio_loss` (**new**) |
| 3 Density | mAP@IoU binary, Peak Concentration, In-Event Coverage@τ | `map_smooth_loss` (existing) + `map_density_loss` (**new**) |
| 4 Semantic | AUC2, logits2 avgMAP | `loss2` / `loss3_text_align` (untouched) |

### What gets removed

`map_coverage_loss` and `map_ranking_loss` (train.py:98–139) use `max(s)` of anomaly/normal regions — they actively reward spikes, which contradicts Objective 3. They are deleted outright (this is a dev branch; no rollback flag).

---

## File Structure

| File | Role | Action |
|---|---|---|
| `src/utils/map_metrics.py` | Pure-function metric suite + composite score | **Create** |
| `tests/test_map_metrics.py` | Unit tests for metric suite | **Create** |
| `src/model.py` | `AnomalyMapHead` bias init | Edit (1 line + init) |
| `src/train.py` | Add 2 losses, remove 2 losses, new checkpoint criterion, new logging | Edit |
| `src/option.py` | Remove 4 args, add 3 args | Edit |
| `src/test.py` | Wire metric suite into eval, return `map_score` | Edit |
| `docs/experiments.md` | Document Phase C rationale + metric suite | Edit (append) |

One new utility file, focused on the metric suite. The loss functions live in train.py alongside the existing BCE/smoothness helpers (follow existing convention, not a separate losses module — only two small functions).

---

## Task 1: Separation metrics (Gap, Normal-mean)

**Files:**
- Create: `src/utils/map_metrics.py`
- Create: `tests/test_map_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_map_metrics.py`:

```python
import numpy as np
import pytest

from src.utils.map_metrics import separation_stats


def test_separation_perfect_map():
    # Event on frames 40..59, s_t = 1 inside, 0 outside
    scores = np.zeros(100)
    scores[40:60] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    s = separation_stats(scores, mask)
    assert s['gap'] == pytest.approx(1.0)
    assert s['in_mean'] == pytest.approx(1.0)
    assert s['out_mean'] == pytest.approx(0.0)


def test_separation_uniform_half():
    scores = np.full(100, 0.5)
    mask = np.zeros(100)
    mask[40:60] = 1.0
    s = separation_stats(scores, mask)
    assert s['gap'] == pytest.approx(0.0)


def test_separation_no_event_returns_nan_gap():
    scores = np.full(100, 0.1)
    mask = np.zeros(100)  # normal video
    s = separation_stats(scores, mask)
    assert np.isnan(s['gap'])
    assert s['in_mean'] is None or np.isnan(s['in_mean'])
    assert s['out_mean'] == pytest.approx(0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/hunghehe2205/Projects/VHung && python -m pytest tests/test_map_metrics.py -v`
Expected: FAIL with `ImportError: No module named 'src.utils.map_metrics'`

- [ ] **Step 3: Create `src/utils/map_metrics.py` with minimal implementation**

```python
"""Map-quality metrics for logits3 anomaly distribution.

Designed to evaluate whether s_t ∈ [0,1] is suitable for the
CDF-cumsum sampler (see inference_viz/sampling.py).

All functions operate on NumPy arrays (CPU) and are side-effect free.
"""
from __future__ import annotations

import math
import numpy as np


def separation_stats(scores: np.ndarray, mask: np.ndarray) -> dict:
    """Gap and per-side means.

    Args:
        scores: (T,) float array, s_t ∈ [0,1].
        mask:   (T,) float/bool array, 1 inside events.

    Returns:
        dict with keys: in_mean, out_mean, gap.
        If no in-event frames exist (normal video), in_mean is NaN and gap is NaN.
        If no out-of-event frames exist (fully anomalous), out_mean is NaN.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    in_frames = scores[mask]
    out_frames = scores[~mask]
    in_mean = float(in_frames.mean()) if in_frames.size > 0 else float('nan')
    out_mean = float(out_frames.mean()) if out_frames.size > 0 else float('nan')
    gap = in_mean - out_mean
    return {'in_mean': in_mean, 'out_mean': out_mean, 'gap': gap}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_map_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/map_metrics.py tests/test_map_metrics.py
git commit -m "feat(map_metrics): add separation_stats (Gap, in/out mean)"
```

---

## Task 2: Mass metrics (EMR, MCL)

**Files:**
- Modify: `src/utils/map_metrics.py`
- Modify: `tests/test_map_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_map_metrics.py`:

```python
from src.utils.map_metrics import mass_stats


def test_mass_perfect_map():
    scores = np.zeros(100)
    scores[40:60] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    m = mass_stats(scores, mask)
    assert m['emr'] == pytest.approx(1.0)
    assert m['etr'] == pytest.approx(0.2)
    assert m['mcl'] == pytest.approx(5.0)


def test_mass_uniform_map():
    scores = np.full(100, 0.5)
    mask = np.zeros(100)
    mask[40:60] = 1.0
    m = mass_stats(scores, mask)
    assert m['emr'] == pytest.approx(0.2)
    assert m['etr'] == pytest.approx(0.2)
    assert m['mcl'] == pytest.approx(1.0)


def test_mass_normal_video_returns_nan_mcl():
    scores = np.full(100, 0.1)
    mask = np.zeros(100)
    m = mass_stats(scores, mask)
    assert m['etr'] == pytest.approx(0.0)
    assert np.isnan(m['mcl'])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_metrics.py::test_mass_perfect_map -v`
Expected: FAIL with `ImportError` on `mass_stats`.

- [ ] **Step 3: Implement `mass_stats`**

Append to `src/utils/map_metrics.py`:

```python
def mass_stats(scores: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> dict:
    """Event-Mass Ratio (EMR), Event-Time Ratio (ETR), and Mass Concentration Lift (MCL).

    EMR = Σ s_t(in-event) / Σ s_t(all)
    ETR = |in-event| / |all|
    MCL = EMR / ETR  (NaN if no events)
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    total_mass = scores.sum()
    in_mass = scores[mask].sum()
    emr = float(in_mass / (total_mass + eps)) if total_mass > 0 else 0.0
    etr = float(mask.sum() / mask.size) if mask.size > 0 else 0.0
    mcl = float(emr / etr) if etr > 0 else float('nan')
    return {'emr': emr, 'etr': etr, 'mcl': mcl, 'mass_lift': emr - etr}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_map_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/map_metrics.py tests/test_map_metrics.py
git commit -m "feat(map_metrics): add mass_stats (EMR, ETR, MCL)"
```

---

## Task 3: Density metrics (Peak Concentration, In-Event Coverage, Entropy)

**Files:**
- Modify: `src/utils/map_metrics.py`
- Modify: `tests/test_map_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from src.utils.map_metrics import density_stats


def test_density_perfect_uniform_event():
    # Uniform 1.0 inside 20-frame event; PC is top-10% mass / event mass
    scores = np.zeros(100)
    scores[40:60] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    d = density_stats(scores, mask)
    assert d['in_event_cov_05'] == pytest.approx(1.0)
    assert d['in_event_cov_03'] == pytest.approx(1.0)
    # Top 10% of 20 frames = 2 frames out of total mass 20 → PC=0.1
    assert d['peak_concentration'] == pytest.approx(0.1)
    # Uniform distribution → normalized entropy = 1
    assert d['in_event_entropy'] == pytest.approx(1.0)


def test_density_single_spike():
    scores = np.zeros(100)
    scores[50] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    d = density_stats(scores, mask)
    assert d['peak_concentration'] == pytest.approx(1.0)
    assert d['in_event_cov_05'] == pytest.approx(1.0 / 20)
    assert d['in_event_entropy'] == pytest.approx(0.0)


def test_density_normal_video():
    scores = np.full(100, 0.1)
    mask = np.zeros(100)
    d = density_stats(scores, mask)
    assert np.isnan(d['peak_concentration'])
    assert np.isnan(d['in_event_entropy'])
    assert np.isnan(d['in_event_cov_05'])
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_metrics.py::test_density_perfect_uniform_event -v`
Expected: FAIL on import of `density_stats`.

- [ ] **Step 3: Implement `density_stats`**

Append to `src/utils/map_metrics.py`:

```python
def density_stats(scores: np.ndarray, mask: np.ndarray,
                  cov_thresholds: tuple = (0.3, 0.5),
                  top_fraction: float = 0.1,
                  eps: float = 1e-8) -> dict:
    """Metrics for in-event density and anti-spike.

    in_event_cov@τ: fraction of event frames with s_t > τ.
    peak_concentration: mass of top X% event frames / total event mass.
    in_event_entropy: normalized entropy H(p̂) / log(L_event), where
        p̂[t] = s_t / Σ_event s_t.  1 = uniform (dense), 0 = single spike.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    event_scores = scores[mask]
    L = event_scores.size
    out = {}
    for tau in cov_thresholds:
        key = f'in_event_cov_{int(round(tau * 10)):02d}'
        out[key] = float((event_scores > tau).mean()) if L > 0 else float('nan')

    if L == 0:
        out['peak_concentration'] = float('nan')
        out['in_event_entropy'] = float('nan')
        return out

    total_event_mass = event_scores.sum()
    if total_event_mass <= eps:
        out['peak_concentration'] = float('nan')
        out['in_event_entropy'] = 0.0
        return out

    k = max(1, int(math.ceil(L * top_fraction)))
    topk = np.sort(event_scores)[-k:]
    out['peak_concentration'] = float(topk.sum() / total_event_mass)

    p = event_scores / total_event_mass
    p_safe = np.clip(p, eps, 1.0)
    H = float(-(p * np.log(p_safe)).sum())
    out['in_event_entropy'] = float(H / math.log(L)) if L > 1 else 0.0
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_map_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/map_metrics.py tests/test_map_metrics.py
git commit -m "feat(map_metrics): add density_stats (PC, in-event coverage, entropy)"
```

---

## Task 4: Binary temporal mAP@IoU

**Files:**
- Modify: `src/utils/map_metrics.py`
- Modify: `tests/test_map_metrics.py`

Reuses `src/utils/detection_map.py:nms` for segment NMS, but implements matching directly because `getLocMAP` is hardcoded to 14 classes.

- [ ] **Step 1: Write the failing test**

Append:

```python
from src.utils.map_metrics import binary_detection_map


def test_binary_detection_map_perfect():
    # 2 videos; each has one event. Predictions match exactly.
    predictions = [
        np.concatenate([np.zeros(40), np.ones(20), np.zeros(40)]),
        np.concatenate([np.zeros(10), np.ones(30), np.zeros(60)]),
    ]
    gt_segments = [
        [[40, 60]],
        [[10, 40]],
    ]
    result = binary_detection_map(predictions, gt_segments)
    assert result['map_avg'] >= 0.99
    assert result['map_at_iou_05'] >= 0.99


def test_binary_detection_map_all_wrong():
    predictions = [np.zeros(100), np.zeros(100)]
    gt_segments = [[[40, 60]], [[10, 40]]]
    result = binary_detection_map(predictions, gt_segments)
    assert result['map_avg'] == pytest.approx(0.0)


def test_binary_detection_map_single_spike():
    # Spike at frame 50 inside GT [40,60] → IoU ≈ 1/20 = 0.05
    pred = np.zeros(100); pred[50] = 1.0
    result = binary_detection_map([pred], [[[40, 60]]])
    # At IoU >= 0.1, prediction matches nothing → mAP=0
    assert result['map_at_iou_01'] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_metrics.py::test_binary_detection_map_perfect -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `binary_detection_map`**

Append to `src/utils/map_metrics.py`:

```python
from utils.detection_map import nms as _nms  # reuse existing NMS


def _extract_segments(score: np.ndarray, thresholds=(0.3, 0.5, 0.7), min_length: int = 2):
    """Threshold a 1-D score array at several levels; produce candidate segments.

    Returns list of [start, end, score].
    """
    segments = []
    for thr in thresholds:
        mask = (score > thr).astype(np.int8)
        padded = np.concatenate([[0], mask, [0]])
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_length:
                seg_score = float(score[s:e].max())
                segments.append([int(s), int(e), seg_score])
    return segments


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    precision = tp_c / (tp_c + fp_c + 1e-12)
    # VOC-style: Σ (precision_at_tp) / n_gt
    ap = float((precision * tp).sum() / n_gt)
    return ap


def binary_detection_map(predictions: list,
                         gt_segments: list,
                         iou_thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5),
                         nms_thresh: float = 0.6) -> dict:
    """Class-agnostic temporal detection mAP over a dataset of videos.

    Args:
        predictions: list length V of 1-D numpy arrays (frame-level or snippet-level scores).
        gt_segments: list length V. gt_segments[i] is iterable of [start, end] pairs
                     (same frame domain as predictions[i]). Empty list means no events.
        iou_thresholds: IoU levels at which to compute AP.
        nms_thresh: IoU threshold for NMS between multi-threshold candidates.

    Returns:
        dict with 'map_at_iou_{XX}' for each threshold plus 'map_avg'.
    """
    # Collect all candidate segments across videos.
    all_preds = []  # [video_idx, start, end, score]
    for i, score in enumerate(predictions):
        score = np.asarray(score, dtype=np.float64)
        segs = _extract_segments(score)
        if len(segs) == 0:
            continue
        segs_arr = np.array(segs)
        segs_arr = segs_arr[np.argsort(-segs_arr[:, -1])]
        _, keep = _nms(segs_arr[:, :2], thresh=nms_thresh)
        for k in keep:
            all_preds.append([i, int(segs_arr[k, 0]), int(segs_arr[k, 1]), float(segs_arr[k, 2])])

    n_gt_total = sum(len(g) for g in gt_segments)
    result = {}
    if n_gt_total == 0:
        for thr in iou_thresholds:
            result[f'map_at_iou_{int(thr*10):02d}'] = float('nan')
        result['map_avg'] = float('nan')
        return result

    if len(all_preds) == 0:
        for thr in iou_thresholds:
            result[f'map_at_iou_{int(thr*10):02d}'] = 0.0
        result['map_avg'] = 0.0
        return result

    all_preds.sort(key=lambda x: -x[3])

    aps = []
    for iou_thr in iou_thresholds:
        # Track matched GT per video
        remaining_gt = [list(map(list, g)) for g in gt_segments]
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        for idx, (vi, ps, pe, _score) in enumerate(all_preds):
            best_iou = 0.0
            best_gi = -1
            for gi, (gs, ge) in enumerate(remaining_gt[vi]):
                inter = max(0, min(pe, ge) - max(ps, gs))
                union = max(pe, ge) - min(ps, gs)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thr and best_gi >= 0:
                tp[idx] = 1.0
                remaining_gt[vi].pop(best_gi)
            else:
                fp[idx] = 1.0
        ap = _average_precision(tp, fp, n_gt_total)
        aps.append(ap)
        result[f'map_at_iou_{int(iou_thr*10):02d}'] = ap

    result['map_avg'] = float(np.mean(aps))
    return result
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_map_metrics.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/map_metrics.py tests/test_map_metrics.py
git commit -m "feat(map_metrics): add binary temporal detection mAP@IoU"
```

---

## Task 5: Aggregator and composite score

**Files:**
- Modify: `src/utils/map_metrics.py`
- Modify: `tests/test_map_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from src.utils.map_metrics import compute_per_video_metrics, composite_map_score


def test_per_video_metrics_perfect():
    scores = np.zeros(100); scores[40:60] = 1.0
    mask = np.zeros(100); mask[40:60] = 1.0
    m = compute_per_video_metrics(scores, mask)
    assert m['gap'] == pytest.approx(1.0)
    assert m['mcl'] == pytest.approx(5.0)
    assert m['peak_concentration'] == pytest.approx(0.1)


def test_composite_map_score_uniform_zero():
    m = {'gap': 0.0, 'mcl': 1.0, 'map_avg': 0.0, 'peak_concentration': 0.1}
    score = composite_map_score(m)
    # gap=0, mcl/3=0.33, map=0, (1-pc)=0.9 → 0.25*0 + 0.25*0.33 + 0.25*0 + 0.25*0.9 = 0.308
    assert 0.3 < score < 0.32


def test_composite_map_score_perfect():
    m = {'gap': 1.0, 'mcl': 5.0, 'map_avg': 1.0, 'peak_concentration': 0.1}
    score = composite_map_score(m)
    # gap=1, clip(mcl/3,0,1)=1, map=1, (1-pc)=0.9 → 0.25*(1+1+1+0.9) = 0.975
    assert 0.97 < score < 0.98
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_metrics.py::test_per_video_metrics_perfect -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement aggregator and composite**

Append to `src/utils/map_metrics.py`:

```python
def compute_per_video_metrics(scores: np.ndarray, mask: np.ndarray) -> dict:
    """Combine separation + mass + density for one video."""
    out = {}
    out.update(separation_stats(scores, mask))
    out.update(mass_stats(scores, mask))
    out.update(density_stats(scores, mask))
    return out


def composite_map_score(metrics: dict,
                        w_gap: float = 0.25,
                        w_mcl: float = 0.25,
                        w_map: float = 0.25,
                        w_pc: float = 0.25,
                        mcl_ref: float = 3.0) -> float:
    """Scalar score ∈ ≈[0,1] for checkpoint selection.

    Components:
      - Gap (clipped to [0,1])
      - MCL normalized to [0,1] by dividing by mcl_ref then clipping
      - mAP@IoU average
      - 1 − PeakConcentration
    Missing keys default to 0 (conservative: a missing metric does not help the score).
    """
    gap = float(np.nan_to_num(metrics.get('gap', 0.0), nan=0.0))
    mcl = float(np.nan_to_num(metrics.get('mcl', 0.0), nan=0.0))
    map_avg = float(np.nan_to_num(metrics.get('map_avg', 0.0), nan=0.0))
    pc = float(np.nan_to_num(metrics.get('peak_concentration', 1.0), nan=1.0))

    gap_c = max(0.0, min(1.0, gap))
    mcl_c = max(0.0, min(1.0, mcl / mcl_ref))
    map_c = max(0.0, min(1.0, map_avg))
    anti_spike = max(0.0, min(1.0, 1.0 - pc))

    return w_gap * gap_c + w_mcl * mcl_c + w_map * map_c + w_pc * anti_spike
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_map_metrics.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add src/utils/map_metrics.py tests/test_map_metrics.py
git commit -m "feat(map_metrics): add per-video aggregator and composite score"
```

---

## Task 6: Mass-ratio loss

**Files:**
- Modify: `src/train.py` (after `map_smooth_loss`, before the removed `map_coverage_loss`)
- Create: `tests/test_map_losses.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_map_losses.py`:

```python
import torch
import pytest

# Will be imported after train.py is refactored; for now add to sys.path.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import map_mass_ratio_loss


def _make(logits_vals, mask_vals):
    logits = torch.tensor(logits_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    mask = torch.tensor(mask_vals, dtype=torch.float32).unsqueeze(0)  # (1, T)
    lengths = torch.tensor([len(logits_vals)])
    return logits, mask, lengths


def test_mass_ratio_loss_zero_when_mass_concentrated():
    # All mass inside event, margin=0.1 → EMR=1.0 ≥ ETR+0.1 ⇒ loss = 0
    T = 20
    logits = [-10.0] * T
    for i in range(5, 15):
        logits[i] = 10.0  # sigmoid ≈ 1.0
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_mass_ratio_loss(L, M, Lens, margin=0.1)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_mass_ratio_loss_positive_when_uniform():
    # Uniform mass → EMR ≈ ETR = 0.5 → loss = margin = 0.3
    T = 20
    logits = [0.0] * T  # sigmoid=0.5
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_mass_ratio_loss(L, M, Lens, margin=0.3)
    assert 0.25 < loss.item() < 0.35
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_losses.py -v`
Expected: FAIL (ImportError on `map_mass_ratio_loss`).

- [ ] **Step 3: Add `map_mass_ratio_loss` to `src/train.py`**

Insert **after** `map_smooth_loss` (currently line 95) and **before** the existing `map_coverage_loss` (which will be removed in Task 8):

```python
def map_mass_ratio_loss(logits3, raw_mask, lengths, margin=0.3, eps=1e-8):
    """Event-Mass Ratio loss: push Σs(in-event) / Σs(all) above ETR + margin.

    Directly drives Objective 2 (mass concentrated in event) so cumsum
    sampler zooms into event regions.
    """
    scores = torch.sigmoid(logits3.squeeze(-1))
    total_loss = 0.0
    total_count = 0
    for i in range(scores.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        s = scores[i, :L]
        m = raw_mask[i, :L]
        in_event = m > 0.5
        if not in_event.any() or in_event.all():
            continue  # normal video or fully anomalous — nothing to drive
        etr = in_event.float().mean()
        pred_ratio = s[in_event].sum() / (s.sum() + eps)
        target = etr + margin
        total_loss = total_loss + F.relu(target - pred_ratio)
        total_count += 1
    if total_count == 0:
        return torch.tensor(0.0, device=logits3.device, requires_grad=True)
    return total_loss / total_count
```

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_map_losses.py::test_mass_ratio_loss_zero_when_mass_concentrated -v`
Run: `python -m pytest tests/test_map_losses.py::test_mass_ratio_loss_positive_when_uniform -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_map_losses.py
git commit -m "feat(losses): add map_mass_ratio_loss (Objective 2)"
```

---

## Task 7: Density loss (negative normalized entropy)

**Files:**
- Modify: `src/train.py`
- Modify: `tests/test_map_losses.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_map_losses.py`:

```python
from train import map_density_loss


def test_density_loss_zero_on_uniform():
    # Uniform inside event → normalized entropy = 1 → loss = 0
    T = 20
    logits = [0.0] * T  # sigmoid=0.5 uniformly
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_density_loss(L, M, Lens)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_density_loss_high_on_spike():
    # One frame high, rest zero inside event → entropy → 0 → loss → 1
    T = 20
    logits = [-10.0] * T
    logits[10] = 10.0
    mask = [0.0] * T
    for i in range(5, 15):
        mask[i] = 1.0
    L, M, Lens = _make(logits, mask)
    loss = map_density_loss(L, M, Lens)
    assert loss.item() > 0.9
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_map_losses.py::test_density_loss_zero_on_uniform -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Add `map_density_loss` to `src/train.py`**

Insert **immediately after** `map_mass_ratio_loss`:

```python
def map_density_loss(logits3, raw_mask, lengths, eps=1e-8):
    """Drive inside-event distribution toward uniform (max normalized entropy).

    p̂[t] = s_t / Σ_event s_t; H = -Σ p̂ log p̂; H_norm = H / log(L_event).
    Loss = 1 - H_norm. Zero when in-event mass is uniform; 1 when fully spiky.
    """
    import math
    scores = torch.sigmoid(logits3.squeeze(-1))
    total_loss = 0.0
    total_count = 0
    for i in range(scores.shape[0]):
        L = int(lengths[i].item())
        if L == 0:
            continue
        s = scores[i, :L]
        m = raw_mask[i, :L]
        in_event = m > 0.5
        L_event = int(in_event.sum().item())
        if L_event <= 1:
            continue
        event_scores = s[in_event]
        total_event = event_scores.sum() + eps
        p = event_scores / total_event
        p_safe = torch.clamp(p, min=eps)
        H = -(p * torch.log(p_safe)).sum()
        H_norm = H / math.log(L_event)
        total_loss = total_loss + (1.0 - H_norm)
        total_count += 1
    if total_count == 0:
        return torch.tensor(0.0, device=logits3.device, requires_grad=True)
    return total_loss / total_count
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_map_losses.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_map_losses.py
git commit -m "feat(losses): add map_density_loss (Objective 3 global)"
```

---

## Task 8: Remove coverage/ranking losses and their args

**Files:**
- Modify: `src/train.py:98-139` (delete `map_coverage_loss` and `map_ranking_loss`)
- Modify: `src/option.py:38-41` (delete 4 args)

- [ ] **Step 1: Delete the two loss functions**

In `src/train.py`, delete these two full blocks:

```python
def map_coverage_loss(logits3, raw_mask, lengths, threshold=0.5):
    """Push anomaly frame scores above threshold."""
    scores = torch.sigmoid(logits3.squeeze(-1))
    ...  # DELETE all of it

def map_ranking_loss(logits3, raw_mask, lengths, margin=0.3):
    """Ensure max(anomaly) - max(normal) > margin per sample."""
    scores = torch.sigmoid(logits3.squeeze(-1))
    ...  # DELETE all of it
```

Verify with grep:

```bash
grep -n "map_coverage_loss\|map_ranking_loss" /Users/hunghehe2205/Projects/VHung/src/train.py
```
Expected: only remaining mentions are in the training-loop call sites (next sub-step).

- [ ] **Step 2: Remove call sites in the training loop**

In `src/train.py`, inside the `for i in pbar:` loop, delete:

```python
            l_cov = map_coverage_loss(logits3, raw_mask_batch, feat_lengths, threshold=args.coverage_threshold)
            l_rank = map_ranking_loss(logits3, raw_mask_batch, feat_lengths, margin=args.ranking_margin)
            ...
            loss_total_cov += l_cov.item()
            loss_total_rank += l_rank.item()
```

Also delete the accumulator variables (`loss_total_cov = 0`, `loss_total_rank = 0` at epoch start) and their appearance in `pbar.set_postfix(...)` and the `log_metrics(...)` calls that include `L_cov` / `L_rank`.

Verify:

```bash
grep -n "l_cov\|l_rank\|loss_total_cov\|loss_total_rank\|L_cov\|L_rank" /Users/hunghehe2205/Projects/VHung/src/train.py
```
Expected: no matches.

- [ ] **Step 3: Remove args from `src/option.py`**

Delete these 4 lines in `src/option.py`:

```python
parser.add_argument('--lambda-coverage', default=1.0, type=float)
parser.add_argument('--lambda-ranking', default=1.0, type=float)
parser.add_argument('--coverage-threshold', default=0.5, type=float)
parser.add_argument('--ranking-margin', default=0.3, type=float)
```

Verify:

```bash
grep -n "lambda-coverage\|lambda-ranking\|coverage-threshold\|ranking-margin\|args.coverage_threshold\|args.ranking_margin\|args.lambda_coverage\|args.lambda_ranking" /Users/hunghehe2205/Projects/VHung/src/train.py /Users/hunghehe2205/Projects/VHung/src/option.py
```
Expected: no matches.

- [ ] **Step 4: Smoke-import test**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import train; import option"`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/train.py src/option.py
git commit -m "refactor: remove coverage/ranking losses (reward spikes, obsolete)"
```

---

## Task 9: Add new args for mass/density + bias init tweak

**Files:**
- Modify: `src/option.py`
- Modify: `src/model.py` (lines 59-77, `AnomalyMapHead`)

- [ ] **Step 1: Add new args to `src/option.py`**

In `src/option.py`, in the "logits3 anomaly map head" section, add these three lines:

```python
parser.add_argument('--lambda-mass', default=1.0, type=float)
parser.add_argument('--lambda-density', default=0.5, type=float)
parser.add_argument('--mass-margin', default=0.3, type=float)
```

- [ ] **Step 2: Tweak `AnomalyMapHead` bias init**

In `src/model.py`, modify `AnomalyMapHead.__init__`:

```python
class AnomalyMapHead(nn.Module):
    """Anomaly map head. Produces frame-level anomaly scores from visual features."""

    def __init__(self, in_dim, hidden_dim=256, kernel_size=7, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size, padding=padding)
        # Init output bias so sigmoid(bias) ≈ prior P(anomaly) ≈ 0.27 (matches HIVAU ETR stats).
        nn.init.constant_(self.conv2.bias, -1.0)
```

- [ ] **Step 3: Verify init**

Run:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import torch
from model import AnomalyMapHead
h = AnomalyMapHead(512)
print('conv2.bias =', h.conv2.bias.data.item())
print('sigmoid(bias) =', torch.sigmoid(h.conv2.bias).item())
"
```
Expected output: `conv2.bias = -1.0` and `sigmoid(bias) ≈ 0.2689`.

- [ ] **Step 4: Commit**

```bash
git add src/option.py src/model.py
git commit -m "feat(model): add mass/density args; init map-head output bias to -1"
```

---

## Task 10: Wire new losses into training loop

**Files:**
- Modify: `src/train.py` training loop

- [ ] **Step 1: Initialize accumulators at epoch start**

Inside the `for e in range(args.max_epoch):` loop, in the block that initializes accumulators, replace the old cov/rank lines with mass/density:

```python
        loss_total1 = 0
        loss_total2 = 0
        loss_total3 = 0
        loss_total_bce = 0
        loss_total_smooth = 0
        loss_total_mass = 0
        loss_total_density = 0
        epoch_map_stats = {'anomaly_mean': 0, 'normal_mean': 0, 'gap': 0, 'coverage': 0}
        stat_count = 0
```

- [ ] **Step 2: Compute losses and update total loss**

Replace the per-batch loss block (was lines ~248–259) with:

```python
            l_bce = map_bce_loss(logits3, soft_mask_batch, feat_lengths)
            l_smooth = map_smooth_loss(logits3, raw_mask_batch, feat_lengths)
            l_mass = map_mass_ratio_loss(logits3, raw_mask_batch, feat_lengths, margin=args.mass_margin)
            l_density = map_density_loss(logits3, raw_mask_batch, feat_lengths)

            loss_total_bce += l_bce.item()
            loss_total_smooth += l_smooth.item()
            loss_total_mass += l_mass.item()
            loss_total_density += l_density.item()

            loss = (loss1 + loss2 + loss3
                    + args.lambda_bce * l_bce
                    + args.lambda_smooth * l_smooth
                    + args.lambda_mass * l_mass
                    + args.lambda_density * l_density)
```

- [ ] **Step 3: Update pbar postfix**

Replace `pbar.set_postfix(...)` to:

```python
            pbar.set_postfix(loss1=loss_total1 / (i + 1),
                             loss2=loss_total2 / (i + 1),
                             L_bce=loss_total_bce / (i + 1),
                             L_mass=loss_total_mass / (i + 1),
                             L_den=loss_total_density / (i + 1))
```

- [ ] **Step 4: Update the step-level and end-of-epoch `log_metrics` strings**

Replace the `f"L_cov=... L_rank=..."` strings with `f"L_mass={loss_total_mass/n_cur:.4f} L_density={loss_total_density/n_cur:.4f}"` in both the step-level log (inside the `if step % 1280 == 0` block) and the end-of-epoch summary log. Same substitution for `n` vs `n_cur`.

- [ ] **Step 5: Smoke-import**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import train"`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/train.py
git commit -m "feat(train): wire mass/density losses into training loop"
```

---

## Task 11: Refactor checkpointing to use composite map_score

**Files:**
- Modify: `src/train.py` (checkpoint save block around lines 189, 288-302, 325-327)

- [ ] **Step 1: Change `ap_best` → `map_score_best`**

At the top of `train()`, replace:

```python
    ap_best = 0
```

with:

```python
    map_score_best = 0.0
```

And update the checkpoint load block (where `checkpoint['ap']` is read) to handle either key:

```python
    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        map_score_best = checkpoint.get('map_score', checkpoint.get('ap', 0.0))
        log_metrics(logger, f"Loaded checkpoint: epoch={epoch + 1} map_score={map_score_best}")
```

- [ ] **Step 2: Update in-epoch eval and save**

Find the block after `test(model, ...)` (currently saving on `auc1 > ap_best`). Replace:

```python
                auc1, _, auc3, _, score_maps = test(model, testloader, args.visual_length, prompt_text,
                                                    gt, gtsegments, gtlabels, device, logger)

                if auc1 > ap_best:
                    ap_best = auc1
                    log_metrics(logger, f"  >> New best AUC1={ap_best:.4f} (AUC3={auc3:.4f})")
                    ...
```

with:

```python
                auc1, _, auc3, _, score_maps, map_score = test(
                    model, testloader, args.visual_length, prompt_text,
                    gt, gtsegments, gtlabels, device, logger)

                if map_score > map_score_best:
                    map_score_best = map_score
                    log_metrics(logger, f"  >> New best MapScore={map_score_best:.4f} "
                                        f"(AUC1={auc1:.4f} AUC3={auc3:.4f})")
                    maps_dir = os.path.join(args.log_dir, 'score_maps')
                    os.makedirs(maps_dir, exist_ok=True)
                    for name, smap in score_maps.items():
                        np.save(os.path.join(maps_dir, f"{name}.npy"), smap)
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'map_score': map_score_best,
                    }
                    torch.save(checkpoint, args.checkpoint_path)
                else:
                    log_metrics(logger, f"  Best MapScore={map_score_best:.4f} (current={map_score:.4f})")
```

- [ ] **Step 3: Update final log line**

Replace:

```python
    log_metrics(logger, f"=== Training finished. Best AUC1: {ap_best:.4f} ===")
```

with:

```python
    log_metrics(logger, f"=== Training finished. Best MapScore: {map_score_best:.4f} ===")
```

- [ ] **Step 4: Smoke-import**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import train"`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/train.py
git commit -m "refactor(train): checkpoint on composite MapScore instead of AUC1"
```

---

## Task 12: Integrate metric suite into `test.py`

**Files:**
- Modify: `src/test.py`

- [ ] **Step 1: Import new module**

At the top of `src/test.py`, add:

```python
from utils.map_metrics import (
    compute_per_video_metrics,
    binary_detection_map,
    composite_map_score,
)
```

- [ ] **Step 2: Collect per-video prob3 and GT segments in the test loop**

Inside the main loop over videos (after `prob3 = torch.sigmoid(logits3[0:len_cur].squeeze(-1))`), accumulate into new lists. Before the loop, initialize:

```python
    per_video_metrics = []
    binary_preds = []      # list of frame-level prob3 arrays
    binary_gt_segs = []    # list of [[start, end], ...]
    video_start = 0        # running index into `gt` array
```

Replace the score_map saving block with one that also captures per-video data:

```python
            score_map_np = np.repeat(prob3.cpu().numpy(), 16)
            score_maps[f"video_{i:04d}_{label}"] = score_map_np

            # Per-video map metrics (uses frame-level gt slice + gt_segments[i]).
            video_len_frames = score_map_np.shape[0]
            gt_slice = np.asarray(gt[video_start: video_start + video_len_frames], dtype=np.float32)
            video_start += video_len_frames

            metrics_i = compute_per_video_metrics(score_map_np, gt_slice)
            per_video_metrics.append(metrics_i)
            binary_preds.append(score_map_np)
            seg_list = gtsegments[i] if gtsegments[i] is not None else []
            binary_gt_segs.append([list(map(int, s)) for s in seg_list])
```

- [ ] **Step 3: Aggregate after loop**

Before `dmap, iou = dmAP(...)` add:

```python
    # Aggregate per-video metrics. Use nan-aware mean for metrics defined only on anomaly videos.
    def _nanmean_key(key):
        vals = [m[key] for m in per_video_metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float('nan')

    gap_mean = _nanmean_key('gap')
    mcl_mean = _nanmean_key('mcl')
    pc_mean = _nanmean_key('peak_concentration')
    cov05_mean = _nanmean_key('in_event_cov_05')
    cov03_mean = _nanmean_key('in_event_cov_03')
    entropy_mean = _nanmean_key('in_event_entropy')

    # Normal-mean: averaged on videos with no events (gtsegments[i] empty).
    normal_means = []
    for i, m in enumerate(per_video_metrics):
        if len(binary_gt_segs[i]) == 0:
            normal_means.append(m['out_mean'])
    normal_mean = float(np.mean(normal_means)) if normal_means else float('nan')

    binary_map = binary_detection_map(binary_preds, binary_gt_segs)

    aggregated = {
        'gap': gap_mean, 'mcl': mcl_mean, 'peak_concentration': pc_mean,
        'map_avg': binary_map['map_avg'],
    }
    map_score = composite_map_score(aggregated)
```

- [ ] **Step 4: Print and log the new metric suite**

Replace the existing print lines with a grouped summary after the mAP prints:

```python
    print("  --- Map Quality Metrics ---")
    print(f"  [Separation] Gap={gap_mean:.3f}  Normal-mean={normal_mean:.3f}")
    print(f"  [Mass]       MCL={mcl_mean:.3f}")
    print(f"  [Localize]   mAP@IoU avg={binary_map['map_avg']:.3f}  "
          f"@0.1={binary_map['map_at_iou_01']:.3f}  @0.5={binary_map['map_at_iou_05']:.3f}")
    print(f"  [Density]    InEventCov@0.5={cov05_mean:.3f}  InEventCov@0.3={cov03_mean:.3f}  "
          f"PeakConc={pc_mean:.3f}  Entropy={entropy_mean:.3f}")
    print(f"  [Composite]  MapScore={map_score:.4f}")

    if logger:
        logger.info(
            f"[MapEval] Gap={gap_mean:.3f} NormMean={normal_mean:.3f} MCL={mcl_mean:.3f} "
            f"mAPavg={binary_map['map_avg']:.3f} InCov05={cov05_mean:.3f} "
            f"PC={pc_mean:.3f} Entropy={entropy_mean:.3f} MapScore={map_score:.4f}"
        )
```

- [ ] **Step 5: Extend the function signature / return**

Change the last line of `test()` from:

```python
    return ROC1, AP1, ROC3, AP3, score_maps
```

to:

```python
    return ROC1, AP1, ROC3, AP3, score_maps, map_score
```

Also update the `__main__` block at bottom of `test.py` which unpacks the return:

```python
    _, _, _, _, score_maps, _ = test(model, testdataloader, args.visual_length, prompt_text,
                                     gt, gtsegments, gtlabels, device, logger)
```

- [ ] **Step 6: Smoke-import**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import test as t"`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/test.py
git commit -m "feat(test): integrate map-quality metric suite + composite MapScore"
```

---

## Task 13: End-to-end smoke test

**Files:**
- No code changes

- [ ] **Step 1: Run 1-epoch training smoke test**

Run:

```bash
cd /Users/hunghehe2205/Projects/VHung
python src/train.py --max-epoch 1 --lambda-mass 1.0 --lambda-density 0.5 \
    --log-dir logs/phase_c_smoke
```

Expected:
- No NaN losses (pbar shows finite numbers for L_bce, L_mass, L_den).
- End-of-epoch log includes `[MapEval]` line with Gap / MCL / mAPavg / MapScore.
- A checkpoint file `final_model/checkpoint.pth` saved if MapScore > 0.

- [ ] **Step 2: Run full unit-test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (15 from Task 1-5 + 4 from Task 6-7 = 19 total).

- [ ] **Step 3: Run standalone eval on existing Exp 4 checkpoint (if available)**

```bash
python src/test.py --model-path final_model/model_ucf.pth
```

Expected: prints all 5 metric groups, produces a finite MapScore for the baseline.

- [ ] **Step 4: Commit log file only if it has useful info**

```bash
# Optional: only commit the log if you want to record the baseline MapScore.
git add logs/phase_c_smoke/train.log
git commit -m "chore: smoke-test log for Phase-C map-quality objective"
```

---

## Task 14: Document Phase C in experiments.md

**Files:**
- Modify: `docs/experiments.md`

- [ ] **Step 1: Append Phase C section**

Append to `docs/experiments.md`:

```markdown
## Phase C — Map-First Objective (starting 2026-04-15)

### Motivation

Phases A–B and Exp 1–5 established that pushing logits3 toward AUC parity is
a dead end: the p1 MIL head already covers AUC, and ensembling p3 with p1 shows
p3 contributes no ranking signal (Exp 5 α*=1.0). Gradient interference from
logits3's BCE/smoothness onto shared features costs ~1 pp of AUC1 (Exp 4).

The real consumer of s_t is the CDF-cumsum keyframe sampler
(`inference_viz/sampling.py:density_aware_sample`), which requires the map to
be a *distribution over time*, not a ranking. That distribution has four
properties we care about:

1. **Separation** — s high inside events, low outside.
2. **Mass concentration** — Σ s_t mass sits inside events (so cumsum zooms in).
3. **Density** — mass is spread across the event, not concentrated on one spike.
4. **Semantic retention** — logits2 still identifies the class (for description pipeline).

### Changes

- Removed `map_coverage_loss` and `map_ranking_loss` (max-based; rewarded spikes).
- Added `map_mass_ratio_loss` (hinge on EMR ≥ ETR + margin) for Objective 2.
- Added `map_density_loss` (1 − normalized in-event entropy) for Objective 3.
- Initialized `AnomalyMapHead.conv2.bias = -1.0` so sigmoid starts near prior ETR.
- Replaced AUC1-based checkpointing with a composite `MapScore`:
  `0.25·Gap + 0.25·clip(MCL/3, 0, 1) + 0.25·mAP@IoU_avg + 0.25·(1 − PeakConc)`.

### New metric suite (primary)

| Group | Metric | Target |
|---|---|---|
| Separation | Gap, Normal-mean | ≥ 0.5, ≤ 0.2 |
| Mass | EMR, MCL | MCL ≥ 2.0 |
| Localization | mAP@IoU binary, avg over {0.1..0.5} | report |
| Density | In-Event Cov@0.5, Peak Concentration | ≥ 0.6, ≤ 0.3 |
| Semantic (reference only) | AUC2, logits2 avgMAP | no regression > 2 pp |
| Reference | AUC1, AUC3 | log only |

### Exp 6 (TBD)

- [ ] Run full 5-epoch training: `python src/train.py --max-epoch 5 --lambda-mass 1.0 --lambda-density 0.5 --log-dir logs/phase_c_v1`
- [ ] Record best MapScore + per-group metrics.
- [ ] Compare against Exp 4 checkpoint on the same metric suite.
```

- [ ] **Step 2: Commit**

```bash
git add docs/experiments.md
git commit -m "docs: document Phase C map-first objective and new metric suite"
```

---

## Verification Matrix

| Objective | Metric(s) checked | Loss(es) | Verified via |
|---|---|---|---|
| 1 Separation | Gap ≥ 0.5, Normal-mean ≤ 0.2 | `map_bce_loss` | Tasks 1, 13 |
| 2 Mass | MCL ≥ 2.0 | `map_mass_ratio_loss` | Tasks 2, 6, 13 |
| 3 Density | mAP@IoU, PC ≤ 0.3, InCov@0.5 ≥ 0.6 | `map_smooth_loss` + `map_density_loss` | Tasks 3, 4, 7, 13 |
| 4 Semantic | AUC2 & logits2 avgMAP no regression | `loss2` (unchanged) | Task 13 eval output |

## Non-goals

- No differentiable sampler / Gumbel-softmax in training.
- No Platt / post-hoc calibration.
- No per-class mAP on logits3 (binary only — per-class stays with logits2).
- No changes to logits1 pipeline or dataset.
- No backward-compat flags for removed losses (dev branch; clean cut).
