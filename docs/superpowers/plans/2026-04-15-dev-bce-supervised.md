# dev_bce — Supervised Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add frame-level supervised losses (binary BCE + Soft-IoU) on top of existing MIL losses in CLIPVAD to optimize class-agnostic temporal localization mAP@IoU on UCF-Crime.

**Architecture:** Keep CLIPVAD model unchanged. Load segment-level events from `HIVAU-70k-NEW/ucf_database_train_filtered.json`, build per-feature binary labels. Add `L_frame_bce` on `logits1` and `L_iou` (soft temporal IoU) via a 3-phase lambda schedule. Evaluate with class-agnostic `mAP@IoU {0.1..0.5, AVG}` computed from `sigmoid(logits1)`.

**Tech Stack:** PyTorch, numpy, pandas, CLIP (ViT-B/16 frozen), pytest (existing `tests/` folder).

**Spec reference:** `docs/superpowers/specs/2026-04-15-dev-bce-supervised-design.md`

**Branch:** `dev_bce` (already created from `main`, spec already committed).

---

## File Structure

| File | Responsibility | Change type |
|---|---|---|
| `list/ucf_CLIP_rgb.csv`, `list/ucf_CLIP_rgbtest.csv` | Feature paths → rewrite to local absolute | MODIFY |
| `src/utils/detection_map.py` | Add `getDetectionMAP_agnostic` | MODIFY |
| `src/test.py` | Log class-agnostic mAP; change return signature | MODIFY |
| `src/utils/tools.py` | Add `build_frame_labels` helper | MODIFY |
| `src/utils/dataset.py` | Load JSON, return `y_bin` | MODIFY |
| `src/utils/compute_pos_weight.py` | Precompute `pos_weight_bin` scalar | NEW |
| `list/pos_weight_bin.npy` | Cached scalar for BCE pos_weight | NEW (generated) |
| `src/train.py` | Add losses, phase schedule, model selection | MODIFY |
| `src/option.py` | New CLI args | MODIFY |
| `tests/test_detection_map_agnostic.py` | Unit test class-agnostic mAP | NEW |
| `tests/test_frame_labels.py` | Unit test label construction | NEW |
| `tests/test_losses.py` | Unit test soft_iou & frame_bce | NEW |
| `docs/superpowers/specs/2026-04-15-baseline-main.md` | Baseline numbers log | NEW (generated) |

`src/model.py` is **not modified**. `tests/` currently contains only `__pycache__` — add new test modules at top level.

---

## Phase 0 — Preparation & Baseline

Purpose: make `model/model_ucf.pth` (trained on `main`) runnable locally, implement class-agnostic mAP, record baseline numbers before any dev_bce change.

### Task 0.1: Rewrite CSV paths to local absolute

**Files:**
- Modify: `list/ucf_CLIP_rgb.csv`
- Modify: `list/ucf_CLIP_rgbtest.csv`

- [ ] **Step 1: Confirm current prefix & target prefix**

```bash
head -2 list/ucf_CLIP_rgb.csv
head -2 list/ucf_CLIP_rgbtest.csv
ls /Users/hunghehe2205/Projects/VHung/UCFClipFeatures/Abuse/Abuse001_x264__0.npy
```

Expected: CSV lines start with `/home/emogenai4e/emo/VHung/UCFClipFeatures/...`; the `ls` succeeds (file exists locally).

- [ ] **Step 2: Rewrite paths with sed (macOS-compatible)**

```bash
sed -i '' 's|/home/emogenai4e/emo/VHung|/Users/hunghehe2205/Projects/VHung|g' list/ucf_CLIP_rgb.csv
sed -i '' 's|/home/emogenai4e/emo/VHung|/Users/hunghehe2205/Projects/VHung|g' list/ucf_CLIP_rgbtest.csv
```

- [ ] **Step 3: Verify**

```bash
head -2 list/ucf_CLIP_rgb.csv
python3 -c "
import pandas as pd, os
df = pd.read_csv('list/ucf_CLIP_rgb.csv')
print(df.iloc[0]['path'])
print('exists:', os.path.exists(df.iloc[0]['path']))
df2 = pd.read_csv('list/ucf_CLIP_rgbtest.csv')
print(df2.iloc[0]['path'])
print('exists:', os.path.exists(df2.iloc[0]['path']))
"
```

Expected: both print `exists: True`.

- [ ] **Step 4: Commit**

```bash
git add list/ucf_CLIP_rgb.csv list/ucf_CLIP_rgbtest.csv
git commit -m "Rewrite feature CSV paths to local absolute paths"
```

### Task 0.2: Add `getDetectionMAP_agnostic` with unit test

**Files:**
- Modify: `src/utils/detection_map.py`
- Create: `tests/test_detection_map_agnostic.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detection_map_agnostic.py`:

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.detection_map import getDetectionMAP_agnostic


def test_agnostic_perfect_prediction():
    # 1 video, 100 frames, GT event frames [20, 50] (anomaly)
    pred = np.zeros(100, dtype=np.float32)
    pred[20:50] = 0.9   # high score inside event
    predictions = [pred]
    gtsegments = [[[20, 50]]]                 # one segment in video 0
    gtlabels = [['Abuse']]                    # class ignored by agnostic

    dmap, iou = getDetectionMAP_agnostic(predictions, gtsegments, gtlabels)

    assert iou == [0.1, 0.2, 0.3, 0.4, 0.5]
    # Perfect IoU=1.0 segment should yield AP=100% at all thresholds
    for v in dmap:
        assert v > 80.0, f"Expected >80 mAP, got {v}"


def test_agnostic_no_overlap():
    # prediction is far from GT
    pred = np.zeros(100, dtype=np.float32)
    pred[80:95] = 0.9
    gtsegments = [[[10, 30]]]
    gtlabels = [['Abuse']]
    dmap, _ = getDetectionMAP_agnostic([pred], gtsegments, gtlabels)
    # Zero true positives at every IoU threshold >= 0.1
    for v in dmap:
        assert v == 0.0


def test_agnostic_collapses_classes():
    # Same video has two GT segments with different class labels;
    # agnostic should treat them as a single "anomaly" class.
    pred = np.zeros(200, dtype=np.float32)
    pred[10:40] = 0.8
    pred[80:110] = 0.7
    gtsegments = [[[10, 40], [80, 110]]]
    gtlabels = [['Abuse', 'Fighting']]        # 2 classes, collapsed
    dmap, _ = getDetectionMAP_agnostic([pred], gtsegments, gtlabels)
    # Both predicted segments should match, so high mAP at every IoU
    for v in dmap:
        assert v > 50.0
```

- [ ] **Step 2: Run test — confirm it fails (import error)**

```bash
cd /Users/hunghehe2205/Projects/VHung && python -m pytest tests/test_detection_map_agnostic.py -v
```

Expected: ImportError because `getDetectionMAP_agnostic` doesn't exist yet.

- [ ] **Step 3: Implement `getDetectionMAP_agnostic`**

Append to `src/utils/detection_map.py`:

```python
def _loc_map_agnostic(predictions, th, gtsegments, gtlabels):
    """
    Class-agnostic temporal localization mAP at a single IoU threshold.
    predictions: list of 1-D np.ndarray (per-frame score)
    gtsegments: list of list of [start_frame, end_frame]
    gtlabels: list of list of class names (ignored; kept for interface parity)
    Returns: float AP % (0..100)
    """
    videos_num = len(predictions)

    predictions_mod = []
    c_score = []
    for p in predictions:
        # p: 1-D [n_frames]; keep top-k% frame scores as video-level "actionness"
        pp = np.sort(p)[::-1]  # descending
        c_s = np.mean(pp[:max(1, int(len(pp) / 16))])
        c_score.append(c_s)
        predictions_mod.append(p)
    predictions = predictions_mod

    segment_predict = []
    for i in range(videos_num):
        tmp = predictions[i]
        segment_predict_multithr = []
        thr_set = np.arange(0.6, 0.7, 0.1)
        for thr in thr_set:
            if tmp.max() == tmp.min():
                continue
            threshold = tmp.max() - (tmp.max() - tmp.min()) * thr
            vid_pred = np.concatenate([np.zeros(1),
                                       (tmp > threshold).astype('float32'),
                                       np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1]
                             for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
            for j in range(len(s)):
                if e[j] - s[j] >= 2:
                    segment_scores = float(np.max(tmp[s[j]:e[j]])) + 0.7 * c_score[i]
                    segment_predict_multithr.append([i, s[j], e[j], segment_scores])
        if len(segment_predict_multithr) != 0:
            arr = np.array(segment_predict_multithr)
            arr = arr[np.argsort(-arr[:, -1])]
            _, keep = nms(arr[:, 1:-1], 0.6)
            segment_predict.extend(list(arr[keep]))

    segment_predict = np.array(segment_predict) if len(segment_predict) else np.zeros((0, 4))
    if len(segment_predict) == 0:
        return 0.0
    segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

    # Collapse ALL GT segments (any class) into a single "anomaly" class
    segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]]
                  for i in range(len(gtsegments))
                  for j in range(len(gtsegments[i]))]
    gtpos = len(segment_gt)
    if gtpos == 0:
        return 0.0

    tp, fp = [], []
    for i in range(len(segment_predict)):
        flag = 0.0
        best_iou = 0.0
        best_j = -1
        for j in range(len(segment_gt)):
            if segment_predict[i][0] == segment_gt[j][0]:
                gt = range(int(segment_gt[j][1]), int(segment_gt[j][2]))
                p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                inter = len(set(gt).intersection(set(p)))
                union = len(set(gt).union(set(p)))
                if union == 0:
                    continue
                IoU = float(inter) / float(union)
                if IoU >= th and IoU > best_iou:
                    flag = 1.0
                    best_iou = IoU
                    best_j = j
        if flag > 0 and best_j >= 0:
            del segment_gt[best_j]
        tp.append(flag)
        fp.append(1.0 - flag)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    if sum(tp) == 0:
        prc = 0.0
    else:
        prc = np.sum((tp_c / (fp_c + tp_c)) * np.array(tp)) / gtpos
    return 100.0 * prc


def getDetectionMAP_agnostic(predictions, gtsegments, gtlabels):
    """Class-agnostic version of getDetectionMAP.
    predictions: list of 1-D np.ndarray (frame-level scores).
    gtsegments, gtlabels: same structure as the class-aware variant.
    """
    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_list = [_loc_map_agnostic(predictions, iou, gtsegments, gtlabels)
                 for iou in iou_list]
    return dmap_list, iou_list
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_detection_map_agnostic.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/detection_map.py tests/test_detection_map_agnostic.py
git commit -m "Add class-agnostic getDetectionMAP + unit tests"
```

### Task 0.3: Add class-agnostic mAP logging to `test.py`

**Files:**
- Modify: `src/test.py`

- [ ] **Step 1: Edit `test()` to also compute class-agnostic mAP**

Replace the body of `test()` in `src/test.py` (keep signature; add a 3rd return value). Apply this edit:

Replace lines 82-90 (the block that computes per-class dmap and prints, and the `return ROC1, AP1`) with:

```python
    from utils.detection_map import getDetectionMAP_agnostic

    # Per-class (legacy)
    dmap_pc, iou = dmAP(element_logits2_stack, gtsegments, gtlabels,
                       excludeNormal=False)
    averageMAP_pc = 0.0
    for i in range(5):
        print('[per-class] mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap_pc[i]))
        averageMAP_pc += dmap_pc[i]
    averageMAP_pc = averageMAP_pc / 5
    print('[per-class] AVG mAP: {:.2f}'.format(averageMAP_pc))

    # Class-agnostic (new)
    # ap1 is list of video-level flattened per-feature sigmoid(logits1) scores.
    # We need per-video lists (same shape as element_logits2_stack each item),
    # upsampled ×16 to frame granularity.
    agnostic_stack = []
    idx = 0
    for feat_scores in ap1_per_video:
        agnostic_stack.append(np.repeat(feat_scores, 16))
    dmap_ag, _ = getDetectionMAP_agnostic(agnostic_stack, gtsegments, gtlabels)
    averageMAP_ag = float(np.mean(dmap_ag))
    for i in range(5):
        print('[agnostic ] mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap_ag[i]))
    print('[agnostic ] AVG mAP: {:.2f}'.format(averageMAP_ag))

    return ROC1, averageMAP_ag
```

And inside the main loop, build `ap1_per_video` as you already build `element_logits2_stack`. Replace the eval loop body. The full updated loop body (inside `with torch.no_grad(): for i, item in enumerate(...)`) should be:

```python
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            ap1_per_video.append(prob1.cpu().numpy())     # NEW: per-video list
            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)
```

And before the loop, add `ap1_per_video = []`.

- [ ] **Step 2: Run `test.py` standalone on `main` checkpoint**

```bash
cd /Users/hunghehe2205/Projects/VHung
python src/test.py --model-path model/model_ucf.pth 2>&1 | tee /tmp/baseline_eval.log
```

Expected: prints `AUC1`, `AP1`, `AUC2`, `AP2`, per-class mAP@{0.1..0.5}, [agnostic] mAP@{0.1..0.5}, [agnostic] AVG mAP, final `return ROC1, averageMAP_ag`.

- [ ] **Step 3: Verify output sanity**

```bash
grep -E "AUC1|AP1|per-class|agnostic" /tmp/baseline_eval.log
```

Expected: baseline AUC1 in typical VadCLIP range ~0.85-0.88, per-class AVG mAP ~20-28%, agnostic AVG mAP should exist and be a positive number.

- [ ] **Step 4: Record baseline results**

Create `docs/superpowers/specs/2026-04-15-baseline-main.md` with the actual numbers copied from `/tmp/baseline_eval.log`. Template (fill in actual numbers):

```markdown
# Baseline (main branch, model_ucf.pth) — 2026-04-15

Commands:
  python src/test.py --model-path model/model_ucf.pth

Results:
- AUC1 = <value>
- AP1  = <value>
- AUC2 = <value>
- AP2  = <value>
- Per-class mAP@0.1 = <>
- Per-class mAP@0.2 = <>
- Per-class mAP@0.3 = <>
- Per-class mAP@0.4 = <>
- Per-class mAP@0.5 = <>
- Per-class AVG mAP = <>
- Agnostic  mAP@0.1 = <>
- Agnostic  mAP@0.2 = <>
- Agnostic  mAP@0.3 = <>
- Agnostic  mAP@0.4 = <>
- Agnostic  mAP@0.5 = <>
- Agnostic  AVG mAP = <baseline number for dev_bce comparison>
```

- [ ] **Step 5: Commit**

```bash
git add src/test.py docs/superpowers/specs/2026-04-15-baseline-main.md
git commit -m "Log class-agnostic mAP in test.py; record main baseline numbers"
```

---

## Phase 1 — Label Construction

Build per-feature binary labels `y_bin[T]` from JSON events.

### Task 1.1: Add `build_frame_labels` helper with tests

**Files:**
- Modify: `src/utils/tools.py`
- Create: `tests/test_frame_labels.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_labels.py`:

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.tools import build_frame_labels


def test_no_event_all_zero():
    y = build_frame_labels(events_sec=[], fps=30.0, n_features=10,
                           clip_len=16, target_len=32)
    assert y.shape == (32,)
    assert y.dtype == np.float32
    assert y.sum() == 0.0


def test_event_marks_overlapping_features():
    # fps=30, event [0.0, 1.0]sec => frames [0, 30). clip_len=16 =>
    # feature 0 covers [0,16): overlap -> 1
    # feature 1 covers [16,32): overlap (16..30) -> 1
    # feature 2 covers [32,48): no overlap -> 0
    y = build_frame_labels(events_sec=[(0.0, 1.0)], fps=30.0,
                           n_features=5, clip_len=16, target_len=5)
    assert list(y) == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_multiple_events_union():
    y = build_frame_labels(events_sec=[(0.0, 0.5), (2.0, 3.0)], fps=30.0,
                           n_features=8, clip_len=16, target_len=8)
    # event1 frames [0,15): feature 0 covers [0,16) -> 1
    # event2 frames [60,90): feature 3 covers [48,64) overlaps at [60,64)-> 1
    #                        feature 4 covers [64,80): overlap -> 1
    #                        feature 5 covers [80,96): overlap at [80,90)-> 1
    assert list(y) == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_padding_when_short():
    # n_features=3 < target_len=10 -> last 7 slots are 0 (padding)
    y = build_frame_labels(events_sec=[(0.0, 1.0)], fps=30.0,
                           n_features=3, clip_len=16, target_len=10)
    assert y.shape == (10,)
    assert y[0] == 1.0
    assert y[3:].sum() == 0.0


def test_downsample_uses_max():
    # n_features=10, target_len=5 -> each target slot aggregates 2 source.
    # Use max (any-overlap survives).
    # Source y = [1,0, 0,0, 0,1, 0,0, 0,0] (events mapped to features 0 and 5)
    # Target should be [max(1,0)=1, max(0,0)=0, max(0,1)=1, max(0,0)=0, max(0,0)=0]
    # Achieve this by placing events at seconds matching features 0 and 5.
    # fps=30 clip_len=16: feature 0 -> frames [0,16)  => event at 0.0-0.5sec
    # feature 5 -> frames [80,96)                    => event at ~2.66-3.1 sec
    y = build_frame_labels(events_sec=[(0.0, 0.3), (2.66, 3.1)], fps=30.0,
                           n_features=10, clip_len=16, target_len=5)
    # After max-downsample, indices 0 and 2 (containing src 0 and src 5) should be 1.
    assert y[0] == 1.0
    assert y[2] == 1.0
    assert y[1] == 0.0
```

- [ ] **Step 2: Run the tests — confirm failure**

```bash
python -m pytest tests/test_frame_labels.py -v
```

Expected: ImportError (function doesn't exist).

- [ ] **Step 3: Implement `build_frame_labels` in `src/utils/tools.py`**

Append to `src/utils/tools.py`:

```python
def build_frame_labels(events_sec, fps, n_features, clip_len=16, target_len=256):
    """
    Build binary per-feature anomaly labels of shape [target_len].

    events_sec: iterable of (start_sec, end_sec) tuples (possibly empty).
    fps: frames-per-second of the source video.
    n_features: number of features actually present in the loaded file.
    clip_len: frames per feature (default 16, matching feature extractor stride).
    target_len: final length after pad/truncate (e.g. visual_length=256).

    Rule: any-overlap between feature window [i*clip_len, (i+1)*clip_len)
          and any event (in frame space) marks label 1.

    When n_features > target_len, downsamples via MAX (preserves any-overlap).
    When n_features < target_len, pads with zeros (normal).
    """
    events_frame = [(float(s) * fps, float(e) * fps) for s, e in events_sec]
    y_raw = np.zeros(n_features, dtype=np.float32)
    for i in range(n_features):
        ws = i * clip_len
        we = (i + 1) * clip_len
        for s, e in events_frame:
            if we > s and ws < e:  # overlap
                y_raw[i] = 1.0
                break

    if n_features >= target_len:
        # Uniform-MAX downsample to target_len (matches uniform_extract layout)
        r = np.linspace(0, n_features, target_len + 1, dtype=np.int32)
        y = np.zeros(target_len, dtype=np.float32)
        for i in range(target_len):
            lo, hi = r[i], r[i + 1]
            if lo == hi:
                y[i] = y_raw[lo] if lo < n_features else 0.0
            else:
                y[i] = y_raw[lo:hi].max()
        return y
    else:
        y = np.zeros(target_len, dtype=np.float32)
        y[:n_features] = y_raw
        return y
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
python -m pytest tests/test_frame_labels.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/utils/tools.py tests/test_frame_labels.py
git commit -m "Add build_frame_labels helper + unit tests"
```

### Task 1.2: Update `UCFDataset` to load JSON and return `y_bin`

**Files:**
- Modify: `src/utils/dataset.py`

- [ ] **Step 1: Rewrite `UCFDataset`**

Replace the entire contents of `src/utils/dataset.py` with:

```python
import os
import json
import re
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools


_VIDEO_NAME_RE = re.compile(r'(.*?)__\d+$')


def _parse_video_name(path: str) -> str:
    """Turn '.../Abuse001_x264__3.npy' -> 'Abuse001_x264'."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = _VIDEO_NAME_RE.match(base)
    return m.group(1) if m else base


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool,
                 label_map: dict, normal: bool = False,
                 json_path: str = None):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal

        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        self._events = {}
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self._events = json.load(f)

    def __len__(self):
        return self.df.shape[0]

    def _lookup_events(self, video_name):
        """Return (events_sec, fps) or ([], 30.0) if not present."""
        entry = self._events.get(video_name)
        if entry is None:
            return [], 30.0
        return list(entry.get('events', [])), float(entry.get('fps', 30.0))

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        clip_feature = np.load(path)
        n_features_raw = clip_feature.shape[0]

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        if not self.test_mode:
            video_name = _parse_video_name(path)
            events_sec, fps = self._lookup_events(video_name)
            y_bin = tools.build_frame_labels(
                events_sec=events_sec,
                fps=fps,
                n_features=n_features_raw,
                clip_len=16,
                target_len=self.clip_dim,
            )
            y_bin = torch.from_numpy(y_bin)  # [clip_dim] float32
            return clip_feature, clip_label, y_bin, clip_length

        # test mode — keep legacy 3-tuple for compatibility with test.py
        return clip_feature, clip_label, clip_length
```

- [ ] **Step 2: Quick smoke test — load one training sample**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.dataset import UCFDataset, _parse_video_name
assert _parse_video_name('/x/Abuse001_x264__3.npy') == 'Abuse001_x264'
assert _parse_video_name('/x/Normal_Videos001__0.npy') == 'Normal_Videos001'
ds = UCFDataset(
    clip_dim=256, file_path='list/ucf_CLIP_rgb.csv', test_mode=False,
    label_map={'Normal':'normal','Abuse':'abuse','Arrest':'arrest','Arson':'arson',
               'Assault':'assault','Burglary':'burglary','Explosion':'explosion',
               'Fighting':'fighting','RoadAccidents':'roadAccidents','Robbery':'robbery',
               'Shooting':'shooting','Shoplifting':'shoplifting','Stealing':'stealing',
               'Vandalism':'vandalism'},
    normal=False,
    json_path='HIVAU-70k-NEW/ucf_database_train_filtered.json')
feat, label, y_bin, length = ds[0]
print('feat', feat.shape, feat.dtype)
print('label', label)
print('y_bin shape', y_bin.shape, 'sum', y_bin.sum().item(), 'any>0', (y_bin>0).any().item())
print('length', length)
"
```

Expected: `feat torch.Size([256, 512])`, `label Abuse` (or some class), `y_bin shape torch.Size([256])`, `y_bin sum` > 0 for anomaly videos, `length` positive int.

- [ ] **Step 3: Commit**

```bash
git add src/utils/dataset.py
git commit -m "UCFDataset: load JSON events and return per-feature y_bin"
```

### Task 1.3: Create `compute_pos_weight.py` & generate `pos_weight_bin.npy`

**Files:**
- Create: `src/utils/compute_pos_weight.py`
- Create (generated): `list/pos_weight_bin.npy`

- [ ] **Step 1: Write the script**

Create `src/utils/compute_pos_weight.py`:

```python
"""Precompute scalar pos_weight for L_frame_bce on the training set.

Usage:
    python src/utils/compute_pos_weight.py \
        --train-list list/ucf_CLIP_rgb.csv \
        --train-json HIVAU-70k-NEW/ucf_database_train_filtered.json \
        --clip-dim 256 \
        --out list/pos_weight_bin.npy
"""
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import UCFDataset


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
    p.add_argument('--train-json',
                   default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
    p.add_argument('--clip-dim', type=int, default=256)
    p.add_argument('--out', default='list/pos_weight_bin.npy')
    args = p.parse_args()

    # Iterate over ALL training samples (both normal + anomaly partitions of CSV)
    # To reuse the dataset class, build one dataset with normal=False filter lifted:
    # simplest: load ALL rows (not filtered). We do that by building two datasets.
    all_ds = [
        UCFDataset(args.clip_dim, args.train_list, test_mode=False,
                   label_map=LABEL_MAP, normal=True,
                   json_path=args.train_json),
        UCFDataset(args.clip_dim, args.train_list, test_mode=False,
                   label_map=LABEL_MAP, normal=False,
                   json_path=args.train_json),
    ]

    n_pos = 0
    n_neg = 0
    for ds in all_ds:
        for i in tqdm(range(len(ds)), desc='scan'):
            _, _, y_bin, length = ds[i]
            # Only count valid frames (within length)
            valid = y_bin[:length]
            n_pos += int((valid > 0.5).sum().item())
            n_neg += int((valid <= 0.5).sum().item())

    pos_weight = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    print(f'n_pos={n_pos}  n_neg={n_neg}  pos_weight={pos_weight:.4f}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, np.array([pos_weight], dtype=np.float32))
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /Users/hunghehe2205/Projects/VHung
python src/utils/compute_pos_weight.py
```

Expected: prints `n_pos=...  n_neg=...  pos_weight=X.XXXX` with X in roughly `[3, 25]`. Writes `list/pos_weight_bin.npy`.

- [ ] **Step 3: Verify file**

```bash
python3 -c "import numpy as np; print(np.load('list/pos_weight_bin.npy'))"
```

Expected: array with 1 float value.

- [ ] **Step 4: Commit**

```bash
git add src/utils/compute_pos_weight.py list/pos_weight_bin.npy
git commit -m "Add compute_pos_weight script and cache pos_weight_bin.npy"
```

---

## Phase 2 — Loss Functions

### Task 2.1: Add `soft_iou_loss` and `frame_bce_loss` to `train.py` with tests

**Files:**
- Modify: `src/train.py`
- Create: `tests/test_losses.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_losses.py`:

```python
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train import soft_iou_loss, frame_bce_loss


def test_soft_iou_perfect_overlap_is_zero():
    probs = torch.tensor([[0.99, 0.99, 0.01, 0.01]])
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(probs).bool()
    loss = soft_iou_loss(probs, target, mask)
    assert loss.item() < 0.05


def test_soft_iou_zero_overlap_is_high():
    probs = torch.tensor([[0.99, 0.99, 0.01, 0.01]])
    target = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    mask = torch.ones_like(probs).bool()
    loss = soft_iou_loss(probs, target, mask)
    assert loss.item() > 0.9


def test_soft_iou_respects_mask():
    probs = torch.tensor([[0.9, 0.9, 0.9, 0.9]])
    target = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask_full = torch.ones_like(probs).bool()
    mask_first_two = torch.tensor([[True, True, False, False]])
    l_full = soft_iou_loss(probs, target, mask_full).item()
    l_masked = soft_iou_loss(probs, target, mask_first_two).item()
    # Masking out the irrelevant half should reduce the loss.
    assert l_masked < l_full


def test_frame_bce_uses_pos_weight():
    logits = torch.tensor([[-2.0, 2.0, -2.0, 2.0]])
    y = torch.tensor([[1.0, 0.0, 1.0, 0.0]])          # all predictions wrong
    mask = torch.ones_like(logits).bool()
    pw_low = torch.tensor([1.0])
    pw_high = torch.tensor([10.0])
    l_low = frame_bce_loss(logits, y, mask, pw_low).item()
    l_high = frame_bce_loss(logits, y, mask, pw_high).item()
    assert l_high > l_low
```

- [ ] **Step 2: Run test — confirm failure**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: ImportError (functions don't exist).

- [ ] **Step 3: Add functions to `src/train.py`**

Add right after the existing `CLAS2` function (before `train()`):

```python
def soft_iou_loss(probs, target, mask, eps=1e-6):
    """Sequence-level soft temporal IoU loss.
    probs: [B, T] in [0,1] — sigmoid(logits1).
    target: [B, T] in {0,1} — y_bin.
    mask: [B, T] bool — True for valid frames.
    Returns scalar mean across batch.
    """
    mask_f = mask.float()
    probs = probs * mask_f
    target = target * mask_f
    inter = (probs * target).sum(dim=-1)
    union = probs.sum(dim=-1) + target.sum(dim=-1) - inter
    iou = (inter + eps) / (union + eps)
    return (1.0 - iou).mean()


def frame_bce_loss(logits, target, mask, pos_weight):
    """Frame-level binary BCE on logits1 with scalar pos_weight.
    logits: [B, T] raw logits.
    target: [B, T] {0,1}.
    mask: [B, T] bool.
    pos_weight: 1-D tensor [1].
    """
    import torch.nn.functional as F  # local import tolerated for clarity
    logits_m = logits[mask]
    target_m = target[mask]
    return F.binary_cross_entropy_with_logits(
        logits_m, target_m, pos_weight=pos_weight.to(logits_m.device))
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_losses.py
git commit -m "Add soft_iou_loss and frame_bce_loss helpers with unit tests"
```

### Task 2.2: Add `get_lambda` schedule to `train.py`

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Add `get_lambda` function**

Right after `frame_bce_loss`, before `def train(...)`, add:

```python
def get_lambda(epoch, phase1_epochs, phase2_epochs, lambda1, lambda2):
    """3-phase schedule for extra losses.
    epoch < phase1_epochs:                return (0, 0)       # MIL-only warmup
    phase1_epochs <= epoch < phase2_epochs: return (lambda1, 0) # +frame BCE
    epoch >= phase2_epochs:                return (lambda1, lambda2) # +IoU
    """
    if epoch < phase1_epochs:
        return 0.0, 0.0
    elif epoch < phase2_epochs:
        return float(lambda1), 0.0
    else:
        return float(lambda1), float(lambda2)
```

- [ ] **Step 2: Add inline test** (extend `tests/test_losses.py`)

Append to `tests/test_losses.py`:

```python
from train import get_lambda


def test_get_lambda_phases():
    assert get_lambda(0, 3, 6, 0.1, 0.1) == (0.0, 0.0)
    assert get_lambda(2, 3, 6, 0.1, 0.1) == (0.0, 0.0)
    assert get_lambda(3, 3, 6, 0.1, 0.1) == (0.1, 0.0)
    assert get_lambda(5, 3, 6, 0.1, 0.1) == (0.1, 0.0)
    assert get_lambda(6, 3, 6, 0.1, 0.1) == (0.1, 0.1)
    assert get_lambda(9, 3, 6, 0.1, 0.1) == (0.1, 0.1)
```

- [ ] **Step 3: Run**

```bash
python -m pytest tests/test_losses.py -v
```

Expected: 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/train.py tests/test_losses.py
git commit -m "Add 3-phase get_lambda schedule"
```

---

## Phase 3 — Training Loop Integration

### Task 3.1: Extend `option.py` with new CLI args

**Files:**
- Modify: `src/option.py`

- [ ] **Step 1: Add args**

Append to `src/option.py` (before the final blank line):

```python
# dev_bce — supervised losses
parser.add_argument('--train-json',
                    default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
parser.add_argument('--pos-weight-path', default='list/pos_weight_bin.npy')
parser.add_argument('--phase1-epochs', default=3, type=int)
parser.add_argument('--phase2-epochs', default=6, type=int)
parser.add_argument('--lambda1', default=0.1, type=float)
parser.add_argument('--lambda2', default=0.1, type=float)
```

- [ ] **Step 2: Smoke check parse**

```bash
python -c "
import sys
sys.path.insert(0, 'src')
import option
args = option.parser.parse_args([])
print('train_json:', args.train_json)
print('pos_weight_path:', args.pos_weight_path)
print('phase1:', args.phase1_epochs, 'phase2:', args.phase2_epochs)
print('lambdas:', args.lambda1, args.lambda2)
"
```

Expected: prints the defaults.

- [ ] **Step 3: Commit**

```bash
git add src/option.py
git commit -m "option.py: add dev_bce CLI args"
```

### Task 3.2: Rewrite `train.py` main training loop

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Update dataset instantiation in `__main__`**

Replace the block at the bottom of `src/train.py` (the one starting with `normal_dataset = UCFDataset(...)`) with:

```python
    label_map = LABEL_MAP

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, True,
        json_path=args.train_json)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, False,
        json_path=args.train_json)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
```

- [ ] **Step 2: Rewrite `train()` function**

Replace the entire body of `def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):` with:

```python
def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    # Load pos_weight scalar (precomputed once by compute_pos_weight.py)
    pos_weight_bin = torch.tensor(
        np.load(args.pos_weight_path).astype(np.float32), device=device)

    ap_best = 0.0   # best avg_mAP (class-agnostic)
    epoch = 0

    if args.use_checkpoint:
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print('checkpoint info: epoch', epoch + 1, 'avg_mAP', ap_best)

    os.makedirs('final_model', exist_ok=True)

    for e in range(epoch, args.max_epoch):
        lam1, lam2 = get_lambda(e, args.phase1_epochs, args.phase2_epochs,
                                args.lambda1, args.lambda2)
        model.train()
        sum_bce_v = sum_nce = sum_cts = sum_fbce = sum_iou = 0.0
        n_iters = min(len(normal_loader), len(anomaly_loader))
        pbar = tqdm(range(n_iters), desc=f'Ep {e+1}/{args.max_epoch} λ1={lam1} λ2={lam2}')

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in pbar:
            n_feat, n_lab, n_ybin, n_len = next(normal_iter)
            a_feat, a_lab, a_ybin, a_len = next(anomaly_iter)

            visual = torch.cat([n_feat, a_feat], dim=0).to(device)
            y_bin = torch.cat([n_ybin, a_ybin], dim=0).to(device)        # [2B, T]
            text_labels = list(n_lab) + list(a_lab)
            lengths = torch.cat([n_len, a_len], dim=0).to(device)
            text_labels_t = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2 = model(visual, None, prompt_text, lengths)

            # Original losses
            loss_bce_v = CLAS2(logits1, text_labels_t, lengths, device)
            loss_nce   = CLASM(logits2, text_labels_t, lengths, device)

            # Text feature divergence (unchanged structure)
            loss_cts = torch.zeros(1).to(device)
            tf_n = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                tf_a = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss_cts += torch.abs(tf_n @ tf_a)
            loss_cts = loss_cts / 13 * 1e-1

            # New losses (phase-gated)
            if lam1 > 0:
                logits1_2d = logits1.squeeze(-1)                          # [2B, T]
                mask_T = (torch.arange(logits1_2d.shape[1], device=device)
                          .unsqueeze(0) < lengths.unsqueeze(1))            # [2B, T]
                loss_fbce = frame_bce_loss(logits1_2d, y_bin, mask_T, pos_weight_bin)
            else:
                loss_fbce = torch.zeros(1, device=device)

            if lam2 > 0:
                probs = torch.sigmoid(logits1.squeeze(-1))
                mask_T2 = (torch.arange(probs.shape[1], device=device)
                           .unsqueeze(0) < lengths.unsqueeze(1))
                loss_iou = soft_iou_loss(probs, y_bin, mask_T2)
            else:
                loss_iou = torch.zeros(1, device=device)

            loss = (loss_bce_v + loss_nce + loss_cts
                    + lam1 * loss_fbce + lam2 * loss_iou)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_bce_v += float(loss_bce_v)
            sum_nce   += float(loss_nce)
            sum_cts   += float(loss_cts)
            sum_fbce  += float(loss_fbce)
            sum_iou   += float(loss_iou)
            pbar.set_postfix(bce_v=sum_bce_v / (i + 1),
                             nce=sum_nce / (i + 1),
                             cts=sum_cts / (i + 1),
                             fbce=sum_fbce / (i + 1),
                             iou=sum_iou / (i + 1))

        # End-of-epoch eval — model selection by class-agnostic avg_mAP
        AUC, avg_mAP = test(model, testloader, args.visual_length, prompt_text,
                            gt, gtsegments, gtlabels, device)
        print(f'[epoch {e+1}] AUC={AUC:.4f}  avg_mAP_agnostic={avg_mAP:.2f}')
        if avg_mAP > ap_best:
            ap_best = avg_mAP
            torch.save({'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}, args.checkpoint_path)
            print(f'  -> new best ({avg_mAP:.2f}), checkpoint saved')

        scheduler.step()
        torch.save(model.state_dict(), 'final_model/model_cur.pth')

    best_ck = torch.load(args.checkpoint_path, weights_only=False)
    torch.save(best_ck['model_state_dict'], args.model_path)
    print(f'Final best avg_mAP_agnostic = {ap_best:.2f}')
```

- [ ] **Step 3: Sanity check imports & parse**

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import train  # noqa: F401
print('train.py imports clean')
"
```

Expected: prints line; no ImportError.

- [ ] **Step 4: Commit**

```bash
git add src/train.py
git commit -m "train.py: integrate frame BCE + Soft IoU with phase schedule; select by agnostic mAP"
```

### Task 3.3: Sanity run — 1 epoch dry-run

**Files:** none modified — verify wiring.

- [ ] **Step 1: Run 1 epoch with tiny batch to confirm loop runs end-to-end**

```bash
cd /Users/hunghehe2205/Projects/VHung
python src/train.py --max-epoch 1 --batch-size 8 --phase1-epochs 0 --phase2-epochs 0 \
    --lambda1 0.1 --lambda2 0.1 2>&1 | tee /tmp/sanity_ep1.log
```

Note: `phase1=phase2=0` forces Phase-3 (all losses active) from epoch 0, validating every code path.

Expected behavior:
- Progress bar shows `bce_v`, `nce`, `cts`, `fbce`, `iou` losses all nonzero.
- At end: `[epoch 1] AUC=<>  avg_mAP_agnostic=<>`.
- No CUDA/shape errors.

- [ ] **Step 2: Inspect log for red flags**

```bash
grep -iE "error|nan|inf" /tmp/sanity_ep1.log | head -20 || echo "clean"
```

Expected: "clean" (nothing found).

- [ ] **Step 3: (If green) commit log excerpt for reference**

```bash
cp /tmp/sanity_ep1.log docs/superpowers/specs/2026-04-15-dev-bce-sanity.log
git add docs/superpowers/specs/2026-04-15-dev-bce-sanity.log
git commit -m "Record 1-epoch sanity run log for dev_bce"
```

---

## Phase 4 — Full Training & Result Documentation

### Task 4.1: Full 10-epoch training with default schedule

- [ ] **Step 1: Kick off training**

```bash
cd /Users/hunghehe2205/Projects/VHung
python src/train.py 2>&1 | tee /tmp/dev_bce_full.log
```

Expected: 10 epochs. Phases:
- Epoch 1-3: λ1=0, λ2=0 (MIL-only; should reproduce ~main behavior)
- Epoch 4-6: λ1=0.1, λ2=0 (+frame BCE)
- Epoch 7-10: λ1=0.1, λ2=0.1 (+soft IoU)

Log should show best `avg_mAP_agnostic` updates across epochs.

- [ ] **Step 2: Extract phase-wise best numbers**

```bash
grep -E "\[epoch [0-9]+\] AUC" /tmp/dev_bce_full.log
```

Expected: 10 lines with per-epoch AUC and avg_mAP_agnostic.

### Task 4.2: Write results doc and compare to baseline

**Files:**
- Create: `docs/superpowers/specs/2026-04-15-dev-bce-results.md`

- [ ] **Step 1: Fill results doc**

Template:

```markdown
# dev_bce — Full training results — 2026-04-15

## Config
- max_epoch=10, batch=64, lr=2e-5
- phase1_epochs=3, phase2_epochs=6, lambda1=0.1, lambda2=0.1

## Per-epoch numbers (from /tmp/dev_bce_full.log)
| epoch | phase | AUC | avg_mAP_agnostic |
|------:|:-----:|----:|-----------------:|
| 1 | MIL-only | ... | ... |
| 2 | MIL-only | ... | ... |
| 3 | MIL-only | ... | ... |
| 4 | +frame_bce | ... | ... |
| 5 | +frame_bce | ... | ... |
| 6 | +frame_bce | ... | ... |
| 7 | +soft_iou | ... | ... |
| 8 | +soft_iou | ... | ... |
| 9 | +soft_iou | ... | ... |
| 10| +soft_iou | ... | ... |

## Comparison vs baseline (from 2026-04-15-baseline-main.md)
| metric | main (baseline) | dev_bce Phase 1 best | dev_bce Phase 2 best | dev_bce Phase 3 best | delta (P3 − baseline) |
|---|---:|---:|---:|---:|---:|
| agnostic AVG mAP | ... | ... | ... | ... | +X.XX |
| AUC              | ... | ... | ... | ... | −X.XX |

## Success criteria check
- [x/ ] Must: Phase 3 AVG mAP > baseline by ≥ 5 (target met / not met)
- [x/ ] Should: Phase 3 > Phase 2 (L_iou contributes)
- [x/ ] May: AUC drop ≤ 2
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-04-15-dev-bce-results.md
git commit -m "dev_bce full training results + baseline comparison"
```

---

## Self-Review Checklist (verification before execution)

- Spec coverage: every section of spec 2026-04-15-dev-bce-supervised-design.md is mapped to at least one task:
  - §3 Data Pipeline → Tasks 1.1, 1.2
  - §5 Loss → Tasks 2.1, 2.2
  - §6 Training Schedule → Tasks 3.1, 3.2, 3.3
  - §7 Evaluation → Task 0.2, 0.3 (and re-used in Task 3.2 via `test()`)
  - §8 Config → Task 3.1
  - §9 Files Impact → covered across tasks
  - §11 Baseline Measurement → Tasks 0.1–0.3 (baseline recorded)
  - §12 Success Criteria → Task 4.2 (comparison doc)
- Placeholders: no "TBD" / "fill in later" / abstract descriptions — every code step has concrete code.
- Type consistency: `UCFDataset.__getitem__` returns `(feat, label, y_bin, length)` in train mode and 3-tuple in test mode — downstream code (`compute_pos_weight.py`, `train.py` loop) unpacks 4-tuple accordingly; `test.py` still unpacks 3-tuple (unchanged).
- `build_frame_labels` signature matches usage in `UCFDataset`.
- `soft_iou_loss` / `frame_bce_loss` / `get_lambda` signatures match callers in `train()` and unit tests.
- `getDetectionMAP_agnostic` signature matches call in updated `test.py` body.

---

## Execution Options

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — executing-plans skill, batch through tasks with checkpoints.

Which approach?
