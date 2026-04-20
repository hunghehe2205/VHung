# TCN Head (Hướng A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Exp 18's boundary heads (start/end BCE) with a dilated TCN head that branches from `x_pre` (post-Transformer, pre-GCN) and produces the primary localization score. Drop boundary/BSN machinery entirely. Target: beat Exp 18 best (mAP_abn=24.06) on UCF-Crime.

**Architecture:** CLIP visual (512d) → position embed → Temporal Transformer (attn_window=8, 2 blocks) → `x_pre` branch point. One branch continues through Dual GCN → `logits1` (video-level) + text-aligned `logits2` (14-class). Second branch is a 4-layer dilated Conv1d TCN (512→128→128→128→1, dilations 1,2,4) producing `tcn_logits` — the primary frame-level anomaly score.

**Tech Stack:** PyTorch, CLIP ViT-B/16 (frozen), AdamW, MultiStepLR. Python 3.10 in conda env `vadclip`. UCF-Crime CLIP features. All paths relative to repo root `/Users/hunghehe2205/Projects/VHung`.

**Why hướng A1 (not A2 with parallel raw):** `x_pre` is already post-Transformer (sharper than post-GCN), so the TCN does not need a parallel raw-CLIP bypass. If it undershoots on boundary sharpness, add parallel raw in a follow-up iteration.

**What gets deleted:** `start_head`/`end_head`, `x_diff`, `boundary_cls_loss`, `build_boundary_offset_targets`, BSN proposal code, `infer_variants.py`, outdated diagnostics. See Task 8 for the full dead-code list.

---

## File Inventory

| File | Action | Responsibility |
| ---- | ------ | -------------- |
| `src/model.py`                | Modify | Add TCN head, remove start/end/x_diff |
| `src/train.py`                | Rewrite | 6 losses, curriculum, dual optimizer, new logging |
| `src/test.py`                 | Modify | Use `tcn_logits` as primary score; drop BSN |
| `src/option.py`               | Modify | Add `--tcn-*`, `--gauss-sigma`, etc.; drop boundary/BSN flags |
| `src/utils/dataset.py`        | Modify | Drop `bnd_targets`; 4-tuple return in train mode |
| `src/utils/tools.py`          | Modify | Drop `build_boundary_offset_targets`; add `build_gaussian_target` |
| `src/utils/detection_map.py`  | Modify | Drop BSN functions (`_peak_pick`, `_bsn_generate_proposals`, `getDetectionMAP_agnostic_bsn`) |
| `src/infer_variants.py`       | Delete  | Depends on logits1 BSN ablation — no longer relevant |
| `src/diagnostic.py`           | Delete  | Uses boundary heads — replaced by new standalone eval in test.py |
| `src/diag_compare.py`         | Delete  | Uses boundary heads |
| `src/diag_localization.py`    | Delete  | Uses boundary heads |
| `src/anomaly_diag.py`         | Delete  | Uses boundary heads |
| `src/normal_diag.py`          | Delete  | Uses boundary heads |
| `src/diagnostic_category.py`  | Audit   | Keep only if it does not reference start/end heads |
| `src/visualize.py`            | Audit   | Keep if it only uses logits1/logits2 |

> **Dead-code rule:** before deleting, `grep` each script for `start_head`, `end_head`, `bnd`, `s_logits`, `e_logits`, `build_boundary_offset`. If ANY match, delete. The code itself is the reference, not the filename.

---

## Phase 0: Branch & Worktree Setup

### Task 0: Confirm branch state

**Files:** none (git only)

- [ ] **Step 0.1: Confirm current branch is `dev_reframe` and working tree clean**

Run: `git -C /Users/hunghehe2205/Projects/VHung status && git -C /Users/hunghehe2205/Projects/VHung branch --show-current`
Expected: branch `dev_reframe`, "nothing to commit, working tree clean"

- [ ] **Step 0.2: If you want isolation, create a worktree — otherwise commit on `dev_reframe`**

Worktree (optional): `git -C /Users/hunghehe2205/Projects/VHung worktree add ../VHung-tcn-A1 -b exp19-tcn-A1`

For this plan we assume work happens on `dev_reframe` directly (user has been making exp commits on this branch). If the user asks for a worktree, create `exp19-tcn-A1`.

---

## Phase 1: Model Surgery

### Task 1: Add TCN head in `src/model.py`, drop start/end heads

**Files:**
- Modify: `src/model.py:60-212`

The current `CLIPVAD.__init__` defines `start_head`, `end_head`. The `forward` returns 5 items: `(text_features_ori, logits1, logits2, start_logits, end_logits)`. After Task 1 it returns 4: `(text_features_ori, logits1, logits2, tcn_logits)`.

- [ ] **Step 1.1: Write failing unit test for TCN head shape**

Create `src/tests/test_tcn_head.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from model import CLIPVAD


def _make_model():
    # small model; CLIP is frozen, we only test TCN head shape
    return CLIPVAD(num_class=14, embed_dim=512, visual_length=256,
                   visual_width=512, visual_head=1, visual_layers=2,
                   attn_window=8, prompt_prefix=10, prompt_postfix=10,
                   device='cpu')


def test_forward_returns_4_tensors_including_tcn_logits():
    model = _make_model().eval()
    visual = torch.randn(2, 256, 512)
    lengths = torch.tensor([128, 200])
    text = ['normal', 'abuse', 'arrest', 'arson', 'assault', 'burglary',
            'explosion', 'fighting', 'roadAccidents', 'robbery', 'shooting',
            'shoplifting', 'stealing', 'vandalism']
    with torch.no_grad():
        out = model(visual, None, text, lengths)
    assert len(out) == 4, f'forward must return 4 tensors, got {len(out)}'
    text_features_ori, logits1, logits2, tcn_logits = out
    assert logits1.shape == (2, 256, 1)
    assert logits2.shape == (2, 256, 14)
    assert tcn_logits.shape == (2, 256, 1), f'tcn_logits must be [B,T,1], got {tuple(tcn_logits.shape)}'


def test_tcn_head_parameters_exist():
    model = _make_model()
    names = {n for n, _ in model.named_parameters()}
    assert any(n.startswith('tcn.') for n in names), 'TCN head params not registered'
    assert not any('start_head' in n or 'end_head' in n for n in names), \
        'start_head / end_head must be removed'
```

- [ ] **Step 1.2: Run test to confirm it fails**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/test_tcn_head.py -x -q
```

Expected: `test_forward_returns_4_tensors_including_tcn_logits` fails (current forward returns 5). `test_tcn_head_parameters_exist` also fails (tcn submodule missing).

- [ ] **Step 1.3: Implement TCN head + drop start/end in `src/model.py`**

In `CLIPVAD.__init__`, REPLACE lines 100-103 (`self.classifier = ...`, `self.start_head = ...`, `self.end_head = ...`) with:

```python
        self.classifier = nn.Linear(visual_width, 1)

        # TCN head (hướng A1): branches from x_pre (post-Transformer, pre-GCN).
        # Dilations 1,2,4 on a 512→128 stem give a receptive field of 15 snippets
        # (≈240 frames at stride 16) — matches typical UCF anomaly duration.
        self.tcn = nn.Sequential(
            nn.Conv1d(visual_width, 128, kernel_size=3, dilation=1, padding=1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, dilation=2, padding=2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, dilation=4, padding=4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 1, kernel_size=1),
        )
```

In `forward` (lines 186-211), REPLACE the whole body with:

```python
    def forward(self, visual, padding_mask, text, lengths):
        visual_features, x_pre = self.encode_video(visual, padding_mask, lengths)
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        # TCN head: x_pre [B, T, D] -> [B, D, T] -> conv stack -> [B, T, 1]
        tcn_logits = self.tcn(x_pre.transpose(1, 2)).transpose(1, 2)

        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07

        return text_features_ori, logits1, logits2, tcn_logits
```

Also remove the comment `# Boundary cls heads — gradient flows to backbone (auxiliary regularizer)` from `__init__`, and remove `import numpy as np` from line 3 if no longer used (grep the file first — likely still unused after this change).

- [ ] **Step 1.4: Run test to confirm it passes**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/test_tcn_head.py -x -q
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add src/model.py src/tests/test_tcn_head.py
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: add TCN head branching from x_pre, drop start/end heads"
```

---

## Phase 2: Data Pipeline Simplification

### Task 2: Drop `bnd_targets` from dataset; add Gaussian target builder

**Files:**
- Modify: `src/utils/tools.py:121-162` (drop `build_boundary_offset_targets`)
- Add to: `src/utils/tools.py` (add `build_gaussian_target`)
- Modify: `src/utils/dataset.py:51-80` (drop `bnd_targets` from __getitem__)

- [ ] **Step 2.1: Write failing test for Gaussian target**

Create `src/tests/test_gaussian_target.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from utils.tools import build_gaussian_target


def test_gaussian_target_has_soft_peak_at_event_center():
    # One event spanning snippets 100..120 in a 256-length target.
    y_bin = np.zeros(256, dtype=np.float32)
    y_bin[100:120] = 1.0
    y_soft = build_gaussian_target(y_bin, sigma=2.0)
    assert y_soft.shape == (256,)
    assert y_soft[110] > 0.99, f'peak inside event must be ~1 (got {y_soft[110]})'
    # Outside event but within sigma*3 of the boundary gets nonzero mass.
    assert y_soft[121] > 0.0 and y_soft[121] < 1.0
    assert y_soft[99] > 0.0 and y_soft[99] < 1.0
    # Far from event is 0.
    assert y_soft[50] == 0.0
    assert y_soft[200] == 0.0


def test_gaussian_target_all_zero_when_no_events():
    y_bin = np.zeros(256, dtype=np.float32)
    y_soft = build_gaussian_target(y_bin, sigma=2.0)
    assert (y_soft == 0).all()
```

- [ ] **Step 2.2: Run test to confirm it fails**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/test_gaussian_target.py -x -q
```

Expected: ImportError for `build_gaussian_target`.

- [ ] **Step 2.3: Implement `build_gaussian_target` in `src/utils/tools.py`**

In `src/utils/tools.py`, DELETE `build_boundary_offset_targets` (lines 121-162) — it is no longer called. APPEND at the end of the file:

```python
def build_gaussian_target(y_bin, sigma=2.0):
    """Convolve a binary [T] label with a Gaussian (radius=3σ) to produce a
    soft regression target in [0, 1]. Inside-event positions saturate at 1.0;
    outside positions near a boundary get exponential falloff. Used as the
    TCN-head BCE target so the head sees a sharper gradient near GT edges
    than a plain step function would give (Exp 13 Gaussian lesson).
    """
    y_bin = np.asarray(y_bin, dtype=np.float32)
    T = y_bin.shape[0]
    if sigma <= 0 or y_bin.sum() == 0:
        return y_bin.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)  # peak=1 (not area=1)
    padded = np.pad(y_bin, radius, mode='constant')
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        out[t] = float((padded[t:t + 2 * radius + 1] * kernel).max())
    return np.clip(out, 0.0, 1.0)
```

Note on kernel choice: peak-normalized (max=1), not area-normalized — so inside-GT positions keep value 1. The max-over-kernel (not sum) gives a smooth falloff that matches the expected probability at a distance.

- [ ] **Step 2.4: Update `src/utils/dataset.py` to return `y_soft`, drop `bnd_targets`**

Replace lines 64-80 of `src/utils/dataset.py` with:

```python
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
            y_soft = tools.build_gaussian_target(y_bin, sigma=2.0)
            y_bin = torch.from_numpy(y_bin)     # [clip_dim] float32
            y_soft = torch.from_numpy(y_soft)   # [clip_dim] float32
            return clip_feature, clip_label, y_bin, y_soft, clip_length
```

Remove the `build_boundary_offset_targets` import (if any) — check top of file; currently `import utils.tools as tools` so no change needed there.

- [ ] **Step 2.5: Run Gaussian-target test**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/test_gaussian_target.py -x -q
```

Expected: 2 passed.

- [ ] **Step 2.6: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add src/utils/tools.py src/utils/dataset.py src/tests/test_gaussian_target.py
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: swap boundary targets for Gaussian-smoothed TCN target"
```

---

## Phase 3: Loss Replacement

### Task 3: Rewrite loss functions in `src/train.py`

**Files:**
- Modify: `src/train.py` — drop `focal_bce_loss`, `boundary_cls_loss`, `dice_loss_anomaly` on logits1. Keep `CLAS2`, `CLASM`. Add `tcn_bce`, `tcn_dice`, `tcn_ctr` (all on `tcn_logits`). Rewrite training loop.

Losses that remain on `logits1`/`logits2`/text:
- `CLAS2(logits1, labels, lengths, device, y_bin)` — unchanged (keeps top-k inside-GT Exp 18 trick).
- `CLASM(logits2, labels, lengths, device)` — unchanged.
- `loss_cts` — unchanged (inline text-feature divergence).

New losses on `tcn_logits` (all compute on `tcn_logits.squeeze(-1)` → [B, T]):
- `tcn_bce(tcn_logits_2d, y_soft, mask, pos_weight=6.0)` — `F.binary_cross_entropy_with_logits` with scalar pos_weight, all videos.
- `tcn_dice(torch.sigmoid(tcn_logits_2d), y_bin, mask)` — anomaly-only (has_pos gating), identical formula to current `dice_loss_anomaly`.
- `tcn_ctr(torch.sigmoid(tcn_logits_2d), y_bin, mask, margin=0.3)` — anomaly-only, identical to `within_video_contrast_loss`.

- [ ] **Step 3.1: Write failing test for new loss signatures**

Create `src/tests/test_losses.py`:

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from train import tcn_bce, tcn_dice, tcn_ctr, CLAS2, CLASM


def test_tcn_bce_scalar_positive():
    logits = torch.randn(2, 256)
    y_soft = torch.rand(2, 256)
    mask = torch.ones(2, 256, dtype=torch.bool)
    pw = torch.tensor([6.0])
    loss = tcn_bce(logits, y_soft, mask, pw)
    assert loss.dim() == 0 and loss.item() > 0


def test_tcn_dice_only_anomaly():
    # All-zero target = normal-only batch → loss should be 0 (no anomaly contribution).
    probs = torch.rand(2, 256)
    y = torch.zeros(2, 256)
    mask = torch.ones(2, 256, dtype=torch.bool)
    assert tcn_dice(probs, y, mask).item() == 0.0


def test_tcn_ctr_margin_violation_positive():
    # inside_mean < outside_mean → violation → positive loss.
    probs = torch.zeros(1, 256)
    probs[0, 0:100] = 0.1   # inside
    probs[0, 100:] = 0.9    # outside
    y = torch.zeros(1, 256)
    y[0, 0:100] = 1.0
    mask = torch.ones(1, 256, dtype=torch.bool)
    loss = tcn_ctr(probs, y, mask, margin=0.3)
    assert loss.item() > 0.3
```

- [ ] **Step 3.2: Run test to confirm it fails**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/test_losses.py -x -q
```

Expected: ImportError for `tcn_bce`/`tcn_dice`/`tcn_ctr`.

- [ ] **Step 3.3: Implement new loss functions + rewrite training loop**

REPLACE the entire `src/train.py` contents with (preserve seed + main block):

```python
import os
import sys
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from tqdm import tqdm

from model import CLIPVAD
from test import test, LABEL_MAP
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
import option


def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)


def CLAS2(logits, labels, lengths, device, y_bin=None):
    """Video-level MIL BCE on logits1. Top-k inside-GT for anomaly (Exp 18).
    Normal videos pool from whole video unchanged."""
    instance_logits = torch.zeros(0).to(device)
    vid_label = 1 - labels[:, 0].reshape(labels.shape[0])
    vid_label = vid_label.to(device)
    probs = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        L = int(lengths[i])
        v = probs[i, :L]
        if y_bin is not None and vid_label[i] > 0.5:
            inside = y_bin[i, :L] > 0.5
            n_inside = int(inside.sum())
            if n_inside > 0:
                pool, k = v[inside], max(1, n_inside // 16 + 1)
            else:
                pool, k = v, max(1, L // 16 + 1)
        else:
            pool, k = v, max(1, L // 16 + 1)
        tmp, _ = torch.topk(pool, k=min(k, pool.shape[0]), largest=True)
        instance_logits = torch.cat([instance_logits, tmp.mean().view(1)], dim=0)
    return F.binary_cross_entropy(instance_logits, vid_label)


def tcn_bce(logits, target, mask, pos_weight):
    """Frame-level BCE on TCN logits [B,T] with Gaussian-smoothed target.
    pos_weight: 1-D tensor [1] (default 6.0 from plan)."""
    logits_m = logits[mask]
    target_m = target[mask]
    return F.binary_cross_entropy_with_logits(
        logits_m, target_m, pos_weight=pos_weight.to(logits_m.device))


def tcn_dice(probs, target, mask, eps=1.0):
    """Anomaly-only soft Dice on sigmoid(tcn_logits). Normal videos skipped."""
    mask_f = mask.float()
    p = probs * mask_f
    y = target * mask_f
    inter = (p * y).sum(-1)
    denom = p.sum(-1) + y.sum(-1)
    dice = (2.0 * inter + eps) / (denom + eps)
    has_pos = (y.sum(-1) > 0).float()
    return ((1.0 - dice) * has_pos).sum() / has_pos.sum().clamp_min(1.0)


def tcn_ctr(probs, target, mask, margin=0.3):
    """Within-video contrast on sigmoid(tcn_logits). Anomaly-only."""
    valid = mask.float()
    inside = (target > 0.5).float() * valid
    outside = (1.0 - (target > 0.5).float()) * valid
    has_inside = (inside.sum(-1) > 0).float()
    inside_mean = (probs * inside).sum(-1) / inside.sum(-1).clamp_min(1.0)
    outside_mean = (probs * outside).sum(-1) / outside.sum(-1).clamp_min(1.0)
    gap_loss = F.relu(margin - (inside_mean - outside_mean))
    return (gap_loss * has_inside).sum() / has_inside.sum().clamp_min(1.0)


def get_lambdas(epoch, phase1_epochs, phase2_epochs):
    """3-phase curriculum for TCN losses.
    P1 (epoch < phase1_epochs)                : (0, 0, 0)  — CLAS2+CLASM warmup
    P2 (phase1 <= epoch < phase2_epochs)      : (1, 0, 0)  — add tcn_bce
    P3 (epoch >= phase2_epochs)               : (1, 1, 1)  — add dice + ctr
    """
    if epoch < phase1_epochs:
        return 0.0, 0.0, 0.0, 1  # last = phase label
    elif epoch < phase2_epochs:
        return 1.0, 0.0, 0.0, 2
    else:
        return 1.0, 1.0, 1.0, 3


def _split_param_groups(model, lr_backbone, lr_tcn):
    tcn_params = [p for n, p in model.named_parameters() if n.startswith('tcn.')]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('tcn.') and p.requires_grad]
    return [
        {'params': other_params, 'lr': lr_backbone, 'weight_decay': 0.0},
        {'params': tcn_params, 'lr': lr_tcn, 'weight_decay': 0.0},
    ]


def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(_split_param_groups(model, args.lr, args.lr_tcn))
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)

    pw_scalar = args.tcn_pos_weight if args.tcn_pos_weight is not None else 6.0
    pos_weight_tcn = torch.tensor([pw_scalar], dtype=torch.float32, device=device)

    ap_best = 0.0
    start_epoch = 0
    if args.use_checkpoint:
        ckpt = torch.load(args.checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        ap_best = ckpt['ap']

    os.makedirs('final_model', exist_ok=True)

    prev_total = None
    for e in range(start_epoch, args.max_epoch):
        lam_bce, lam_dice, lam_ctr, phase = get_lambdas(
            e, args.phase1_epochs, args.phase2_epochs)

        # Curriculum transition logs
        if e == args.phase1_epochs:
            print(f'[curriculum ep {e+1} P1->P2] activating: tcn_bce (λ=1.0)', flush=True)
        if e == args.phase2_epochs:
            print(f'[curriculum ep {e+1} P2->P3] activating: tcn_dice (λ=1.0), tcn_ctr (λ=1.0)', flush=True)

        model.train()
        sum_clas2 = sum_clasm = sum_cts = 0.0
        sum_tbce = sum_tdice = sum_tctr = 0.0
        n_iters = min(len(normal_loader), len(anomaly_loader))
        t_start = time.time()
        pbar = tqdm(range(n_iters), desc=f'Ep {e+1}/{args.max_epoch} P{phase}',
                    disable=not sys.stderr.isatty(), leave=False)
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in pbar:
            n_feat, n_lab, n_ybin, n_ysoft, n_len = next(normal_iter)
            a_feat, a_lab, a_ybin, a_ysoft, a_len = next(anomaly_iter)
            visual = torch.cat([n_feat, a_feat], dim=0).to(device)
            y_bin = torch.cat([n_ybin, a_ybin], dim=0).to(device)
            y_soft = torch.cat([n_ysoft, a_ysoft], dim=0).to(device)
            text_labels = list(n_lab) + list(a_lab)
            lengths = torch.cat([n_len, a_len], dim=0).to(device)
            text_labels_t = get_batch_label(text_labels, prompt_text, label_map).to(device)

            text_features, logits1, logits2, tcn_logits = model(visual, None, prompt_text, lengths)

            loss_clas2 = CLAS2(logits1, text_labels_t, lengths, device, y_bin=y_bin)
            loss_clasm = CLASM(logits2, text_labels_t, lengths, device)

            loss_cts = torch.zeros(1, device=device)
            tf_n = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                tf_a = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss_cts += torch.abs(tf_n @ tf_a)
            loss_cts = loss_cts / 13 * args.lambda_cts

            tcn_2d = tcn_logits.squeeze(-1)
            mask_T = (torch.arange(tcn_2d.shape[1], device=device)
                      .unsqueeze(0) < lengths.unsqueeze(1))

            if lam_bce > 0:
                loss_tbce = tcn_bce(tcn_2d, y_soft, mask_T, pos_weight_tcn)
            else:
                loss_tbce = torch.zeros(1, device=device)

            if lam_dice > 0 or lam_ctr > 0:
                probs = torch.sigmoid(tcn_2d)
            if lam_dice > 0:
                loss_tdice = tcn_dice(probs, y_bin, mask_T)
            else:
                loss_tdice = torch.zeros(1, device=device)
            if lam_ctr > 0:
                loss_tctr = tcn_ctr(probs, y_bin, mask_T, margin=args.contrast_margin)
            else:
                loss_tctr = torch.zeros(1, device=device)

            loss = (loss_clas2
                    + args.lambda_nce * loss_clasm
                    + loss_cts
                    + lam_bce * loss_tbce
                    + lam_dice * loss_tdice
                    + lam_ctr * loss_tctr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_clas2 += float(loss_clas2); sum_clasm += float(loss_clasm)
            sum_cts += float(loss_cts); sum_tbce += float(loss_tbce)
            sum_tdice += float(loss_tdice); sum_tctr += float(loss_tctr)
            pbar.set_postfix(CLAS2=sum_clas2 / (i + 1), CLASM=sum_clasm / (i + 1),
                             tbce=sum_tbce / (i + 1), tdice=sum_tdice / (i + 1),
                             tctr=sum_tctr / (i + 1))

        train_secs = time.time() - t_start
        avg_clas2 = sum_clas2 / n_iters
        avg_clasm = sum_clasm / n_iters
        avg_cts   = sum_cts   / n_iters
        avg_tbce  = sum_tbce  / n_iters
        avg_tdice = sum_tdice / n_iters
        avg_tctr  = sum_tctr  / n_iters
        total = avg_clas2 + args.lambda_nce * avg_clasm + avg_cts \
                + lam_bce * avg_tbce + lam_dice * avg_tdice + lam_ctr * avg_tctr
        lr_bb = optimizer.param_groups[0]['lr']
        lr_tcn = optimizer.param_groups[1]['lr']

        AUC, avg_mAP_abn, dmap_abn, AUC_tcn, diag = test(
            model, testloader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, device, quiet=True,
            return_diag=True, eval_head=args.eval_head)
        abn_str = '/'.join(f'{v:.2f}' for v in dmap_abn[:5])
        is_best = avg_mAP_abn > ap_best
        tag = ' *' if is_best else ''

        # Per-epoch 2-line training log + compact eval
        print(f'[ep {e+1:2d}/{args.max_epoch} P{phase} {train_secs:.0f}s] '
              f'lam=(1,{args.lambda_nce},{args.lambda_cts},{lam_bce},{lam_dice},{lam_ctr}) '
              f'lr_bb={lr_bb:.1e} lr_tcn={lr_tcn:.1e}', flush=True)
        print(f'  loss: CLAS2={avg_clas2:.3f} CLASM={avg_clasm:.3f} cts={avg_cts:.4f} '
              f'tcn_bce={avg_tbce:.3f} tcn_dice={avg_tdice:.3f} tcn_ctr={avg_tctr:.3f} '
              f'total={total:.3f}', flush=True)
        print(f'[ep {e+1:2d} eval] mAP_abn={avg_mAP_abn:.2f} AUC_tcn={AUC_tcn:.4f} '
              f'AUC_wsv={AUC:.4f} bsh_med={diag["bsh_med"]:.4f} '
              f"peak_in_gt={diag['peak_in_gt']:.3f} over_cov_med={diag['over_cov_med']:.2f}x"
              f' [{abn_str}]{tag}', flush=True)

        # Phase transition loss diagnostic
        if e == args.phase1_epochs and prev_total is not None:
            delta = total - prev_total
            print(f'[ep {e+1} P1->P2 loss trace] Δ_total={delta:+.3f} '
                  f'(expected from tcn_bce activation)', flush=True)
        if e == args.phase2_epochs and prev_total is not None:
            delta = total - prev_total
            print(f'[ep {e+1} P2->P3 loss trace] Δ_total={delta:+.3f} '
                  f'(expected from tcn_dice + tcn_ctr activation)', flush=True)

        # Watchdogs
        frac_high = diag.get('frac_tcn_high', 1.0)
        if frac_high < 0.02:
            print(f'[watchdog] WARN frac(tcn_prob>0.5)={frac_high:.4f} <2% — possible collapse', flush=True)
        if e >= 5 and AUC_tcn < 0.80:
            print(f'[watchdog] WARN AUC_tcn={AUC_tcn:.4f} <0.80 at ep {e+1}', flush=True)

        if is_best:
            ap_best = avg_mAP_abn
            torch.save({'epoch': e, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}, args.checkpoint_path)

        scheduler.step()
        prev_total = total

    # Save final-epoch checkpoint alongside best (for phase-transition analysis)
    torch.save(model.state_dict(), 'final_model/model_final.pth')
    best_ck = torch.load(args.checkpoint_path, weights_only=False, map_location=device)
    torch.save(best_ck['model_state_dict'], args.model_path)
    print(f'Final best avg_mAP_abn = {ap_best:.2f}', flush=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    setup_seed(args.seed)
    label_map = LABEL_MAP

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False,
                                label_map, True, json_path=args.train_json)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False,
                                 label_map, False, json_path=args.train_json)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size,
                                shuffle=True, drop_last=True)
    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device)

    # Finetune from baseline (non-strict: tcn head is new)
    if args.load_baseline:
        base = torch.load(args.load_baseline, weights_only=False, map_location=device)
        missing, unexpected = model.load_state_dict(base, strict=False)
        print(f'[load_baseline] missing={len(missing)} unexpected={len(unexpected)}', flush=True)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
```

- [ ] **Step 3.4: Run loss tests**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/test_losses.py -x -q
```

Expected: 3 passed.

- [ ] **Step 3.5: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add src/train.py src/tests/test_losses.py
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: rewrite train.py with 6 losses + 3-phase TCN curriculum"
```

---

## Phase 4: Eval Path

### Task 4: Update `src/test.py` — `tcn_logits` primary, drop BSN

**Files:**
- Modify: `src/test.py` — replace `logits1`-based localization with `tcn_logits`, drop s_logits/e_logits, add diagnostic block (AUC_tcn, bsh_med, peak_in_gt, over_cov, norm stats, frame metrics).

The new test function returns `(AUC_wsv, avg_mAP_abn, dmap_abn, AUC_tcn, diag_dict)` instead of `(AUC, avg_mAP_abn, dmap_abn, dmap_bsn, bsn_stats)`. The `diag_dict` carries the watchdog inputs.

- [ ] **Step 4.1: Replace the whole body of `src/test.py` with:**

```python
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.detection_map import (getDetectionMAP,
                                  getDetectionMAP_agnostic,
                                  getDetectionMAP_abnormal_only)
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _upsample(x, factor=16):
    """Snippet → frame repeat upsample for a 1-D array."""
    return np.repeat(np.asarray(x, dtype=np.float32), factor)


def _frame_metrics(scores, gt, thr=0.5):
    pred = (scores >= thr).astype(np.int32)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    FPR = fp / max(fp + tn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=F1, FPR=FPR)


def _boundary_sharpness(prob, gt_bin, k_frame=48):
    """Mean |prob(t) - prob(t-1)| over GT edges ±k_frame window."""
    diff = np.abs(np.diff(prob, prepend=prob[0]))
    gt_diff = np.abs(np.diff(gt_bin.astype(np.float32), prepend=0))
    edges = np.where(gt_diff > 0.5)[0]
    if len(edges) == 0:
        return 0.0
    mask = np.zeros_like(prob, dtype=bool)
    for ed in edges:
        lo, hi = max(0, ed - k_frame), min(len(prob), ed + k_frame)
        mask[lo:hi] = True
    return float(diff[mask].mean()) if mask.any() else 0.0


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         device, quiet=False, return_diag=False, eval_head='tcn'):
    """Evaluate model. Primary localization score = sigmoid(tcn_logits) when
    eval_head='tcn' else sigmoid(logits1)."""
    model.to(device)
    model.eval()

    tcn_per_video = []     # snippet-resolution probs for TCN head
    wsv_per_video = []     # snippet-resolution probs for logits1 (class-agnostic)
    cls_logits_stack = []  # softmax(logits2) per video for class-aware mAP

    with torch.no_grad():
        iterator = testdataloader if quiet else tqdm(
            testdataloader, desc='Testing', disable=not sys.stderr.isatty())
        for i, item in enumerate(iterator):
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
                    lengths[j] = maxlen; length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen; length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, logits2, tcn_logits = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[-1])
            logits2 = logits2.reshape(-1, logits2.shape[-1])
            tcn_logits = tcn_logits.reshape(-1, tcn_logits.shape[-1])

            prob_tcn = torch.sigmoid(tcn_logits[0:len_cur, 0]).cpu().numpy()
            prob_wsv = torch.sigmoid(logits1[0:len_cur, 0]).cpu().numpy()
            tcn_per_video.append(prob_tcn)
            wsv_per_video.append(prob_wsv)
            cls_logits_stack.append(
                logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy())

    primary = tcn_per_video if eval_head == 'tcn' else wsv_per_video
    primary_frame = np.concatenate([_upsample(v) for v in primary])
    wsv_frame = np.concatenate([_upsample(v) for v in wsv_per_video])

    AUC_tcn = roc_auc_score(gt, primary_frame)
    AUC_wsv = roc_auc_score(gt, wsv_frame)
    AP_tcn  = average_precision_score(gt, primary_frame)
    AP_wsv  = average_precision_score(gt, wsv_frame)

    agnostic_stack = [_upsample(v) for v in primary]
    dmap_abn, _ = getDetectionMAP_abnormal_only(agnostic_stack, gtsegments, gtlabels)
    avg_abn = float(np.mean(dmap_abn))

    # Diagnostic block (always computed; cheap)
    diag = _compute_diag(primary, gt, gtsegments, gtlabels, tcn_per_video,
                          wsv_per_video, maxlen)

    if not quiet:
        # Full standalone eval printout (5-block convention)
        print(f'[TCN  ] AUC={AUC_tcn:.4f} AP={AP_tcn:.4f}')
        print(f'[WSVAD] logits1 AUC={AUC_wsv:.4f} AP={AP_wsv:.4f}')
        abn_str = '/'.join(f'{v:.2f}' for v in dmap_abn[:5])
        print(f'[abn @rel=0.60] AVG={avg_abn:.2f} [{abn_str}]')

        dmap_thr, _ = getDetectionMAP_agnostic(agnostic_stack, gtsegments, gtlabels)
        avg_thr = float(np.mean(dmap_thr))
        thr_str = '/'.join(f'{v:.2f}' for v in dmap_thr[:5])
        print(f'[all @rel=0.60] AVG={avg_thr:.2f} [{thr_str}]')

        fm = _frame_metrics(primary_frame, gt, thr=0.5)
        print(f'[frame @thr=0.50] TP={fm["TP"]} FP={fm["FP"]} FN={fm["FN"]} TN={fm["TN"]}'
              f' | P={fm["P"]:.3f} R={fm["R"]:.3f} F1={fm["F1"]:.3f} FPR={fm["FPR"]:.4f}')
        print(f'[sep  anom={diag["n_anom"]}] inside={diag["inside_mean"]:.3f} '
              f'outside={diag["outside_mean"]:.3f} Δ=+{diag["delta"]:.3f} '
              f'peak_in_gt={diag["peak_in_gt"]:.3f}')
        print(f'[cov  anom={diag["n_anom"]}] over_cov(mean)={diag["over_cov_mean"]:.2f}x '
              f'over_cov(med)={diag["over_cov_med"]:.2f}x '
              f'coverage_inside={diag["cov_inside"]:.3f} '
              f'spillover={diag["spillover"]:.3f}')
        print(f'[bsh  anom={diag["n_anom"]}] boundary_sharp={diag["bsh_mean"]:.4f} '
              f'med={diag["bsh_med"]:.4f} k_frame=±48')
        print(f'[norm n={diag["n_norm"]}]    max(med)={diag["norm_max_med"]:.3f} '
              f'max(mean)={diag["norm_max_mean"]:.3f} '
              f'frac>0.5={diag["norm_frac_hi"]:.4f} n_max>0.9={diag["n_norm_hot"]}')

    if return_diag:
        return AUC_wsv, avg_abn, dmap_abn, AUC_tcn, diag
    return AUC_wsv, avg_abn, dmap_abn, AUC_tcn, None


def _compute_diag(primary, gt, gtsegments, gtlabels, tcn_probs, wsv_probs, maxlen):
    """Compute sep / cov / bsh / norm / watchdog diagnostics on primary head."""
    n_anom = 0; n_norm = 0
    inside_sum = 0.0; outside_sum = 0.0
    peak_in_gt = 0; peak_total = 0
    over_cov = []; cov_inside = []; spillover = []
    bsh = []
    norm_maxes_med = []; norm_maxes_mean = []; norm_hot = 0; norm_frac_hi = 0.0
    total_snip_norm = 0

    # per-video GT binary (derived from gtsegments at frame resolution)
    per_vid_len_frames = [len(_upsample(p)) for p in primary]
    offsets = np.cumsum([0] + per_vid_len_frames)
    for i, p_snip in enumerate(primary):
        p_frame = _upsample(p_snip)
        gt_slice = gt[offsets[i]:offsets[i] + len(p_frame)]
        is_anom = any(l != 'A' for l in gtlabels[i])
        if is_anom:
            n_anom += 1
            inside_mask = gt_slice > 0.5
            outside_mask = ~inside_mask
            if inside_mask.sum() > 0:
                inside_sum  += p_frame[inside_mask].mean()
                outside_sum += p_frame[outside_mask].mean() if outside_mask.sum() > 0 else 0.0
                pk = int(np.argmax(p_frame))
                peak_in_gt += int(inside_mask[pk]); peak_total += 1
                # coverage / over-coverage
                thr = 0.5 * p_frame.max()
                high_mask = p_frame >= thr
                cov_inside.append(float((high_mask & inside_mask).sum()
                                         / max(inside_mask.sum(), 1)))
                spillover.append(float((high_mask & outside_mask).sum()
                                        / max(high_mask.sum(), 1)))
                over_cov.append(float(high_mask.sum() / max(inside_mask.sum(), 1)))
                bsh.append(_boundary_sharpness(p_frame, inside_mask.astype(np.float32)))
        else:
            n_norm += 1
            norm_maxes_mean.append(float(p_frame.mean()))
            norm_maxes_med.append(float(np.median(p_frame)))
            norm_hot += int(p_frame.max() > 0.9)
            norm_frac_hi += float((p_frame > 0.5).sum())
            total_snip_norm += len(p_frame)

    delta = 0.0
    if n_anom > 0:
        inside_mean = inside_sum / n_anom
        outside_mean = outside_sum / n_anom
        delta = inside_mean - outside_mean
    else:
        inside_mean = outside_mean = 0.0

    diag = dict(
        n_anom=n_anom, n_norm=n_norm,
        inside_mean=inside_mean, outside_mean=outside_mean, delta=delta,
        peak_in_gt=peak_in_gt / max(peak_total, 1),
        over_cov_mean=float(np.mean(over_cov)) if over_cov else 0.0,
        over_cov_med=float(np.median(over_cov)) if over_cov else 0.0,
        cov_inside=float(np.mean(cov_inside)) if cov_inside else 0.0,
        spillover=float(np.mean(spillover)) if spillover else 0.0,
        bsh_mean=float(np.mean(bsh)) if bsh else 0.0,
        bsh_med=float(np.median(bsh)) if bsh else 0.0,
        norm_max_med=float(np.median(norm_maxes_med)) if norm_maxes_med else 0.0,
        norm_max_mean=float(np.mean(norm_maxes_mean)) if norm_maxes_mean else 0.0,
        n_norm_hot=norm_hot,
        norm_frac_hi=(norm_frac_hi / max(total_snip_norm, 1)),
        frac_tcn_high=(norm_frac_hi + sum((_upsample(p) > 0.5).sum() for p in tcn_probs))
                       / max(sum(len(_upsample(p)) for p in tcn_probs), 1),
    )
    return diag


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()
    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)
    model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                      map_location=device), strict=False)
    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels,
         device, eval_head=args.eval_head)
```

- [ ] **Step 4.2: Smoke-import test**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -c "from test import test; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4.3: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add src/test.py
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: replace BSN eval path with TCN-head primary + full diagnostic block"
```

---

## Phase 5: Option Flags

### Task 5: Update `src/option.py`

**Files:**
- Modify: `src/option.py` — add TCN flags, drop boundary/BSN flags, bump default `max_epoch`, `phase*`, milestones.

- [ ] **Step 5.1: REPLACE `src/option.py` with:**

```python
import argparse

parser = argparse.ArgumentParser(description='VadCLIP-UCF Exp19 (TCN head hướng A1)')
parser.add_argument('--seed', default=234, type=int)

# Model architecture
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

# Training
parser.add_argument('--max-epoch', default=20, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--lr', default=2e-5, type=float,
                    help='Backbone lr (Transformer+GCN+classifier+text)')
parser.add_argument('--lr-tcn', default=1e-4, type=float,
                    help='TCN head lr (separate param group)')
parser.add_argument('--scheduler-rate', default=0.1, type=float)
parser.add_argument('--scheduler-milestones', default=[6, 11], nargs='+', type=int)

# Paths
parser.add_argument('--model-path', default='final_model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='final_model/checkpoint.pth')
parser.add_argument('--load-baseline', default='model/model_ucf.pth',
                    help='Warm-start from this checkpoint (strict=False)')
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy')
parser.add_argument('--train-json',
                    default='HIVAU-70k-NEW/ucf_database_train_filtered.json')

# Curriculum
parser.add_argument('--phase1-epochs', default=3, type=int,
                    help='Epochs of CLAS2+CLASM warmup (TCN inactive)')
parser.add_argument('--phase2-epochs', default=6, type=int,
                    help='Up to this, only tcn_bce is active; after, dice+ctr join')

# Loss weights
parser.add_argument('--lambda-nce', default=1.0, type=float)
parser.add_argument('--lambda-cts', default=0.1, type=float,
                    help='Internal weight applied to loss_cts')
parser.add_argument('--contrast-margin', default=0.3, type=float,
                    help='Margin for tcn_ctr within-video contrast loss')

# TCN head
parser.add_argument('--tcn-pos-weight', default=6.0, type=float,
                    help='pos_weight scalar for tcn_bce (Gaussian target)')
parser.add_argument('--gauss-sigma', default=2.0, type=float,
                    help='Gaussian smoothing σ for TCN target (snippet units)')

# Eval
parser.add_argument('--eval-head', default='tcn', choices=['tcn', 'wsv'],
                    help="Primary head for localization metrics")
```

- [ ] **Step 5.2: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add src/option.py
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: option flags — TCN head, dual optimizer, 20ep curriculum"
```

---

## Phase 6: Dead-Code Removal

### Task 6: Strip BSN from `src/utils/detection_map.py`

**Files:**
- Modify: `src/utils/detection_map.py:311-483` — delete `_peak_pick`, `_iou_matching_ap`, `_is_anomaly_video`, `_bsn_generate_proposals`, `getDetectionMAP_agnostic_bsn`.

- [ ] **Step 6.1: Delete BSN block**

In `src/utils/detection_map.py`, delete everything from the line `# --------------- BSN-style proposal generation ---------------` to end-of-file (lines 311-483).

- [ ] **Step 6.2: Verify nothing else imports BSN**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -c "from utils.detection_map import getDetectionMAP_abnormal_only, getDetectionMAP_agnostic, getDetectionMAP; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6.3: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add src/utils/detection_map.py
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: drop BSN proposal generator (no longer used)"
```

### Task 7: Delete stale scripts

**Files:**
- Delete: `src/infer_variants.py`
- Delete: `src/diagnostic.py`
- Delete: `src/diag_compare.py`
- Delete: `src/diag_localization.py`
- Delete: `src/anomaly_diag.py`
- Delete: `src/normal_diag.py`

For each file not in the `Delete` list above (e.g., `src/diagnostic_category.py`, `src/visualize.py`, `src/make_list.py`, `src/analyze_hivau.py`), grep for boundary references first; delete if any found, else keep.

- [ ] **Step 7.1: Grep for boundary references in remaining scripts**

```
cd /Users/hunghehe2205/Projects/VHung && /bin/ls src/*.py
```

For each file that is NOT one of (`model.py`, `train.py`, `test.py`, `option.py`, `make_list.py`, `visualize.py`, `analyze_hivau.py`, `diagnostic_category.py`), verify it is on the delete list.

Then for the candidates to keep (`visualize.py`, `make_list.py`, `analyze_hivau.py`, `diagnostic_category.py`):

Run: `grep -l -E "start_head|end_head|s_logits|e_logits|bnd_targets|x_diff|build_boundary_offset" src/{visualize,make_list,analyze_hivau,diagnostic_category}.py`
If any file prints, add it to the delete list.

- [ ] **Step 7.2: Delete the files**

```
cd /Users/hunghehe2205/Projects/VHung && rm src/infer_variants.py src/diagnostic.py src/diag_compare.py src/diag_localization.py src/anomaly_diag.py src/normal_diag.py
```

Plus any extra files flagged in Step 7.1.

- [ ] **Step 7.3: Commit**

```
git -C /Users/hunghehe2205/Projects/VHung add -A src/
git -C /Users/hunghehe2205/Projects/VHung commit -m "exp19: delete stale diagnostics + inference variants (used removed boundary heads)"
```

---

## Phase 7: Verification

### Task 8: Smoke-test the whole pipeline on CPU

**Files:** none (just run)

- [ ] **Step 8.1: Run all pytest**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -m pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 8.2: Dry-run training for 1 step (CPU)**

```
cd /Users/hunghehe2205/Projects/VHung/src && conda run -n vadclip python -c "
import torch, option, sys
from torch.utils.data import DataLoader
from model import CLIPVAD
from utils.dataset import UCFDataset
from train import _split_param_groups, tcn_bce, tcn_dice, tcn_ctr, CLAS2, CLASM
from utils.tools import get_prompt_text, get_batch_label
from test import LABEL_MAP
args = option.parser.parse_args([])
# Tiny override
args.batch_size = 2
args.visual_length = 64
ds = UCFDataset(args.visual_length, args.train_list, False, LABEL_MAP, False,
                json_path=args.train_json)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                args.visual_width, args.visual_head, args.visual_layers,
                args.attn_window, args.prompt_prefix, args.prompt_postfix, 'cpu')
opt = torch.optim.AdamW(_split_param_groups(model, args.lr, args.lr_tcn))
it = iter(loader)
feat, lab, y_bin, y_soft, lens = next(it)
prompt = get_prompt_text(LABEL_MAP)
lab_t = get_batch_label(list(lab), prompt, LABEL_MAP)
_, l1, l2, tcn_l = model(feat, None, prompt, lens)
loss = CLAS2(l1, lab_t, lens, 'cpu', y_bin=y_bin) + CLASM(l2, lab_t, lens, 'cpu')
loss.backward()
opt.step()
print('smoke-train OK, loss=%.3f' % loss.item())
"
```

Expected: `smoke-train OK, loss=…` (non-NaN).

- [ ] **Step 8.3: Commit any fixups and tag the plan complete**

If any issues arose, fix and commit with `exp19: …fix…` messages before marking the plan complete.

```
git -C /Users/hunghehe2205/Projects/VHung log --oneline -10
```

Expected: visible `exp19:` commits in sequence.

---

## Self-Review Checklist

- [ ] Every `start_head`/`end_head`/`s_logits`/`e_logits`/`bnd_targets`/`build_boundary_offset`/`boundary_cls_loss`/`focal_bce_loss` reference has been removed or deleted.
- [ ] `model.forward` returns exactly 4 tensors and the last is `tcn_logits` of shape `[B, T, 1]`.
- [ ] `UCFDataset.__getitem__` returns 5-tuple in train mode (`feat, label, y_bin, y_soft, length`) and 3-tuple in test mode.
- [ ] `train.py` only imports `CLAS2`, `CLASM`, `tcn_bce`, `tcn_dice`, `tcn_ctr` (plus infrastructure) — no boundary/focal/fbce.
- [ ] `test.py` uses `tcn_logits` as primary score, emits the 5-block diagnostic, and no BSN.
- [ ] `option.py` has `--tcn-pos-weight`, `--gauss-sigma`, `--lr-tcn`, `--eval-head`, no `--lambda-boundary`/`--bsn-*`.
- [ ] `detection_map.py` still exports `getDetectionMAP`, `getDetectionMAP_agnostic`, `getDetectionMAP_abnormal_only`; BSN symbols gone.
- [ ] All stale diagnostic scripts deleted.
- [ ] `pytest` passes.
- [ ] Smoke train step finishes with a finite loss.

## Logging Reference (for future execution)

Per-epoch training log (two lines):

```
[ep  4/20 P2 72s] lam=(1,1.0,0.1,1,0,0) lr_bb=2.0e-5 lr_tcn=1.0e-4
  loss: CLAS2=0.038 CLASM=0.215 cts=0.002 tcn_bce=0.820 tcn_dice=0.000 tcn_ctr=0.000 total=1.075
```

Per-epoch eval log (one line):

```
[ep  4 eval] mAP_abn=22.40 AUC_tcn=0.8712 AUC_wsv=0.8601 bsh_med=0.0147 peak_in_gt=0.414 over_cov_med=3.65x [18.5/16.3/12.1/9.2/5.4]
```

Standalone eval (after `python test.py`):

```
[TCN  ] AUC=0.8712 AP=0.3124
[WSVAD] logits1 AUC=0.8601 AP=0.2884
[abn @rel=0.60] AVG=24.80 [30.2/28.1/25.3/20.4/19.9]
[all @rel=0.60] AVG=17.40 [...]

[frame @thr=0.50] TP=... FP=... FN=... TN=... | P=0.xxx R=0.xxx F1=0.xxx FPR=0.xxxx
[sep  anom=140] inside=0.612 outside=0.181 Δ=+0.431 peak_in_gt=0.414
[cov  anom=140] over_cov(mean)=4.12x over_cov(med)=3.65x coverage_inside=0.880 spillover=0.552
[bsh  anom=140] boundary_sharp=0.0182 med=0.0147 k_frame=±48
[norm n=150]    max(med)=0.201 max(mean)=0.114 frac>0.5=0.0321 n_max>0.9=3
```

Curriculum transitions:

```
[curriculum ep 4 P1->P2] activating: tcn_bce (λ=1.0)
[curriculum ep 7 P2->P3] activating: tcn_dice (λ=1.0), tcn_ctr (λ=1.0)
[ep 4 P1->P2 loss trace] Δ_total=+0.821 (expected from tcn_bce activation)
```

Watchdogs (fire only on violation):

```
[watchdog] WARN frac(tcn_prob>0.5)=0.0145 <2% — possible collapse
[watchdog] WARN AUC_tcn=0.7811 <0.80 at ep 6
```
