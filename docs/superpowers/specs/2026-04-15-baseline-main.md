# Baseline (main branch, model_ucf.pth) — 2026-04-15

## Environment

- Machine: macOS (Darwin 25.0.0), Apple Silicon, CPU-only inference (MPS available but
  `device = cuda if is_available() else cpu` → CPU fallback).
- Conda env: `vadclip` (torch 2.4.1, Python 3.8).
- Checkpoint: `model/model_ucf.pth` (tracked on `main`).
- Test set: `list/ucf_CLIP_rgbtest.csv` (290 videos).
- Eval code: `src/test.py` on commit `8edce95e` (dev_bce), with additional CPU/MPS
  portability fixes (see Notes).

## Command

```bash
conda activate vadclip
python src/test.py --model-path model/model_ucf.pth
```

## Results

### Video-level frame scores

- **AUC1** (sigmoid(logits1), binary C-Branch) = **0.8736**
- **AP1**  (sigmoid(logits1))                   = **0.3122**
- **AUC2** (1 - softmax(logits2)[:, Normal])    = **0.8570**
- **AP2**  (same score)                         = **0.2553**

### Per-class mAP (legacy VadCLIP metric, kept for reference)

| IoU | mAP  |
|----:|-----:|
| 0.1 | 13.44% |
| 0.2 | 10.10% |
| 0.3 | 7.01% |
| 0.4 | 6.66% |
| 0.5 | 3.88% |
| **AVG** | **8.22%** |

### Class-agnostic mAP (the metric dev_bce will be compared against)

| IoU | mAP  |
|----:|-----:|
| 0.1 | 21.98% |
| 0.2 | 12.93% |
| 0.3 | 6.80% |
| 0.4 | 4.01% |
| 0.5 | 1.65% |
| **AVG** | **9.47%** |

## Observations

- `AUC1 > AUC2` by ~1.6 points — C-Branch is already the stronger video-level
  discriminator; this aligns with the decision to evaluate class-agnostic mAP from
  `sigmoid(logits1)`.
- At `IoU=0.1` class-agnostic (21.98%) > per-class (13.44%) — confirms per-class is
  harder (correct class + correct segment), and class-agnostic surfaces more of the
  model's localization signal.
- At `IoU=0.5` class-agnostic (1.65%) < per-class (3.88%) — the binary actionness
  score from `sigmoid(logits1)` is less sharply peaked than per-class softmax, so
  boundary precision is worse at tight IoU thresholds. This is the gap Phase 2
  (frame BCE) and Phase 3 (soft IoU) should close most visibly.

## Success thresholds for dev_bce

From `docs/superpowers/specs/2026-04-15-dev-bce-supervised-design.md` §12:

- **Must:** `AVG_agnostic_mAP` (Phase 3, epochs 7+) ≥ **14.47%** (baseline 9.47 + 5).
- **Should:** Phase 3 > Phase 2 (confirms L_iou adds value).
- **May:** AUC1 drop ≤ 2 → final AUC1 ≥ **0.8536**.

## Notes — CPU/MPS portability fixes made during baseline measurement

The checkpoint was saved on CUDA and two locations hardcoded `.to('cuda')`/omitted
`map_location`. To run on a CUDA-less machine we applied minimal fixes (committed
separately):

- `src/test.py:122` — added `map_location=device` to `torch.load(...)`.
- `src/utils/layers.py:57-62` — `DistanceAdj.forward` now reads `device` from
  `self.sigma.device` instead of hardcoded `'cuda'`, so the module follows the
  model's device.

These fixes are device-agnostic and safe on CUDA machines.
