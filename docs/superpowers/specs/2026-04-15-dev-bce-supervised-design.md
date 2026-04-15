# dev_bce — Supervised Localization with Frame-BCE + Soft-IoU

**Branch**: `dev_bce` (tách từ `main`)
**Date**: 2026-04-15
**Status**: Design approved, awaiting implementation plan.

## 1. Goal

Chuyển bài toán VAD từ weakly-supervised (MIL video-level) sang **supervised** dùng
segment-level annotations trong `HIVAU-70k-NEW/ucf_database_train_filtered.json`.

**Mục tiêu duy nhất**: tối ưu **class-agnostic temporal localization**.

**Metric chính**: `mAP@IoU` tại các ngưỡng `{0.1, 0.2, 0.3, 0.4, 0.5}` + `AVG`.
Class-agnostic — collapse toàn bộ GT segment về 1 class "anomaly".
AUC/AP vẫn được log nhưng **không** dùng làm model-selection criterion; chấp nhận
AUC giảm nhẹ để đổi lấy localization tốt hơn.

## 2. Non-goals

- Không đổi model architecture. `CLIPVAD` giữ nguyên.
- Không bỏ các loss MIL hiện có — chúng được giữ lại làm "prior" regularization.
- Không tối ưu per-class mAP. Metric class-agnostic là metric duy nhất.
- Không thêm boundary regression head — để sau cho branch khác nếu cần.

## 3. Data Pipeline

### 3.1 Nguồn label mới

`HIVAU-70k-NEW/ucf_database_train_filtered.json` chứa các entry:

```json
"Abuse001_x264": {
  "n_frames": 2729,
  "fps": 30.0,
  "label": ["Abuse"],
  "events": [[3.933, 15.867]]
}
```

- `events` là danh sách `[start_sec, end_sec]`.
- Normal videos không cần có trong JSON (default: không event).

### 3.2 Join với feature CSV

`src/utils/dataset.UCFDataset` hiện đọc `list/ucf_CLIP_rgb.csv` có cột `[path, label]`.
Thêm logic:

1. Parse `video_name` từ `path` (basename không extension).
2. Load JSON 1 lần trong `__init__`, lookup `events`/`fps`/`n_frames` per video.
3. Video thiếu trong JSON → treat as 0 event (hợp lý cho Normal videos).

### 3.3 Label construction (binary, frame-level)

Feature được extract theo clip 16 frame. Với mỗi feature index `i` trong video (sau
khi qua `tools.process_feat`/`process_split`):

```
window_frame = [i*16, (i+1)*16)
events_frame = [(round(s*fps), round(e*fps)) for s,e in events]
y_bin[i] = 1 if any(overlap(window_frame, ev) > 0 for ev in events_frame) else 0
```

Rule: **any-overlap** (một frame chung cũng tính là anomaly). Đã được user chốt.

### 3.4 Sampling-aware labeling

`tools.process_feat` có thể down/up-sample features về `clip_dim=256`. Label phải
tính **trên index thật trước sampling**, sau đó apply cùng phép sampling / padding
lên `y_bin` để đảm bảo alignment. Cách tối giản:

1. Build `y_bin_raw` theo số feature gốc của video.
2. Trong `process_feat`, trả về thêm `sampled_indices` (hoặc `sampled_mask`).
3. `y_bin = y_bin_raw[sampled_indices]` — đảm bảo pad/truncate cùng pattern.

### 3.5 Dataset return signature (mới)

```
UCFDataset.__getitem__ → (feat[T,D], text_label:str, y_bin[T] float, length:int)
```

- `text_label` là class string (VD "Abuse"), dùng cho `CLAS2`/`CLASM` (giữ nguyên).
- `y_bin` là vector mới cho các loss frame-level.

### 3.6 Balanced sampling (giữ nguyên)

2 DataLoader: `normal_loader` (label=='Normal') và `anomaly_loader` (label!='Normal').
Mỗi iteration cat 2 batch → 50/50. Normal videos → toàn bộ `y_bin=0`.

### 3.7 `pos_weight` precompute

Script `src/utils/compute_pos_weight.py`:

- Duyệt toàn train set (sau khi build label).
- `N_pos = sum(y_bin==1)`, `N_neg = sum(y_bin==0)`.
- `pos_weight_bin = N_neg / N_pos` (scalar, dự kiến ~5-15).
- Lưu `list/pos_weight_bin.npy`.

## 4. Model

### 4.1 CLIPVAD — giữ nguyên architecture

- `logits1 = classifier(x) [B,T,1]` — C-Branch, binary anomaly head.
- `logits2 = sim(visual, text) [B,T,14]` — A-Branch, per-class alignment.

### 4.2 Thay đổi duy nhất ở `model.py`: bỏ `/ 0.07` ở `logits2`

```python
# BEFORE:
logits2 = visual_features_norm @ text_features_norm.type(...) / 0.07
# AFTER:
logits2 = visual_features_norm @ text_features_norm.type(...)
```

Lý do: `CLASM` (softmax CE) vẫn hoạt động đúng trên cosine similarity raw — chỉ là
temperature=1 thay vì 1/0.07. Nếu cần giữ behavior cũ, có thể rescale lại trong
`CLASM` local (chia 0.07 ngay trước softmax). Giải pháp **chốt**: **bỏ hẳn** `/0.07`
và chia trong `CLASM` để test pipeline không bị ảnh hưởng.

Thực tế: để tránh đụng vào behavior gốc khi `λ1=λ2=0`, **giữ** `/0.07` trong
forward. Sigmoid path KHÔNG được dùng cho test (vì metric class-agnostic dùng
`sigmoid(logits1)`). → Không có train/test mismatch vì `logits2` chỉ dùng cho loss
`CLASM` (train) và `getDetectionMAP_perclass` (test cũ, ta bỏ). Tóm lại: không cần
đổi gì ở `model.py`.

### 4.3 Quyết định cuối: không đổi `model.py`

- `logits2` giữ nguyên `/ 0.07`.
- `CLASM` dùng `logits2` như cũ (softmax CE video-level).
- `logits1` dùng cho cả `CLAS2` (giữ) + `L_frame_bce` + `L_iou` (mới) — raw logit
  hoặc sigmoid tùy loss.
- Test class-agnostic chỉ dùng `sigmoid(logits1)`.

## 5. Loss

### 5.1 Công thức tổng

```
L = L_bce_video + L_nce + λ_cts * L_cts
  + λ1 * L_frame_bce          # Phase 1 & 2
  + λ2 * L_iou                # Phase 2 only
```

Map sang code hiện tại:

| Symbol | Hàm | File:dòng (current main) | Trạng thái |
|---|---|---|---|
| `L_bce_video` | `CLAS2(logits1, ...)` | `src/train.py:31-43` | Giữ |
| `L_nce` | `CLASM(logits2, ...)` | `src/train.py:18-28` | Giữ |
| `L_cts` | text divergence | `src/train.py:97-102` (coef `1e-1`) | Giữ, `λ_cts=1e-1` |
| `L_frame_bce` | NEW | — | Thêm mới |
| `L_iou` | NEW (soft IoU) | — | Thêm mới |

### 5.2 `L_frame_bce`

```python
# logits1_flat:[2B,T], y_bin:[2B,T], mask_T:[2B,T] bool (valid frames from lengths)
pw = torch.tensor([pos_weight_bin], device=device)    # scalar, loaded once
L_frame_bce = F.binary_cross_entropy_with_logits(
    logits1.squeeze(-1)[mask_T],
    y_bin[mask_T],
    pos_weight=pw,
)
```

### 5.3 `L_iou` (Soft Temporal IoU, per-video rồi mean)

```python
def soft_iou_loss(probs, target, mask, eps=1e-6):
    # probs:[B,T] in [0,1] = sigmoid(logits1); target:[B,T] {0,1}; mask:[B,T] bool
    probs = probs * mask
    target = target * mask
    inter = (probs * target).sum(dim=-1)
    union = probs.sum(dim=-1) + target.sum(dim=-1) - inter
    iou = (inter + eps) / (union + eps)
    return (1 - iou).mean()
```

Sequence-level → gradient đẩy `probs` khớp mask GT theo cả overlap và union → giảm
false positive ngoài event (tăng union), tăng true positive trong event.

**Edge case**: video Normal thuần (target toàn 0) → `sum(target)=0` → `iou = eps/(sum(probs)+eps)`
→ `1 - iou ≈ 1 - eps/sum(probs)`. Loss này push `sum(probs) → 0` trên video Normal,
đúng hướng. Không cần mask video Normal khỏi loss.

## 6. Training Schedule

### 6.1 Lambda schedule (3 phase trong 1 run)

```python
def get_lambda(epoch):
    if epoch < args.phase1_epochs:          # default 3
        return 0.0, 0.0                      # baseline MIL only
    elif epoch < args.phase2_epochs:         # default 6
        return args.lambda1, 0.0             # +L_frame_bce warm-up
    else:
        return args.lambda1, args.lambda2    # +L_iou full hybrid
```

- `max_epoch=10` (default option.py) → 3 / 3 / 4 epoch cho 3 phase.
- Default `lambda1=0.1`, `lambda2=0.1`.
- Phase 1 (epoch 1-3) cho phép reproduce behavior gần `main` → sanity check pipeline
  không bị regression.

### 6.2 Training loop pseudocode

```python
for e in range(max_epoch):
    λ1, λ2 = get_lambda(e)
    for (n_batch, a_batch) in zip(normal_loader, anomaly_loader):
        feat = cat([n_batch.feat, a_batch.feat])
        text_labels = list(n_batch.text_label) + list(a_batch.text_label)
        y_bin = cat([n_batch.y_bin, a_batch.y_bin])
        lengths = cat([n_batch.len, a_batch.len])
        mask_T = build_mask(lengths, T)

        text_features, logits1, logits2 = model(feat, None, prompt_text, lengths)

        loss_bce_v = CLAS2(logits1, text_labels_encoded, lengths, device)
        loss_nce   = CLASM(logits2, text_labels_encoded, lengths, device)
        loss_cts   = text_divergence(text_features) * 1e-1

        loss_frame = 0.0
        loss_iou   = 0.0
        if λ1 > 0:
            loss_frame = bce_frame(logits1, y_bin, mask_T, pw_bin)
        if λ2 > 0:
            loss_iou = soft_iou_loss(sigmoid(logits1), y_bin, mask_T)

        loss = loss_bce_v + loss_nce + loss_cts + λ1*loss_frame + λ2*loss_iou
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        log every step: {loss_bce_v, loss_nce, loss_cts, loss_frame, loss_iou, λ1, λ2}

    # end-of-epoch eval
    AUC, avg_mAP = test(...)
    if avg_mAP > best: save(checkpoint)
    scheduler.step()
```

### 6.3 Model selection

- Checkpoint "best" chọn theo `avg_mAP_agnostic` (mean across 5 IoU thresholds).
- `final_model/model_cur.pth` lưu state sau mỗi epoch (như hiện tại).
- AUC log nhưng không ảnh hưởng selection.

## 7. Evaluation

### 7.1 Class-agnostic mAP — `src/utils/detection_map.py` (new function)

```python
def getDetectionMAP_agnostic(predictions_1d, gtsegments, gtlabels):
    """
    predictions_1d: list[np.ndarray] 1-D per video (sigmoid(logits1), upsampled x16).
    gtsegments, gtlabels: các .npy hiện có. Ignore class — collapse all segments.
    Return: (dmap_list, iou_list) — tương đồng getDetectionMAP cũ.
    """
```

Logic:

1. Copy `getLocMAP` từ code cũ, đặt `classes_num = 1`.
2. Bỏ vòng lặp `for c in range(14)` — chỉ 1 iteration "anomaly".
3. `segment_gt` = `[[i, s, e] for i, segs in enumerate(gtsegments) for (s,e) in segs]`
   (ignore `gtlabels`).
4. Threshold + NMS logic giữ nguyên.
5. `predictions[i]` là 1-D (shape `[n_frames]`), không phải `[n_frames, 14]`. Cập
   nhật slicing tương ứng.

### 7.2 `src/test.py` — update

```python
prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))  # [T]
prob1_frame = np.repeat(prob1.cpu().numpy(), 16, 0)    # upsample tới frame-level

ap1_concat.append(prob1.cpu().numpy())                 # for AUC
agnostic_stack.append(prob1_frame)                     # for mAP agnostic

# sau loop:
all_frame_scores = np.repeat(np.concatenate(ap1_concat), 16)
AUC = roc_auc_score(gt, all_frame_scores)
AP  = average_precision_score(gt, all_frame_scores)

dmap, iou = getDetectionMAP_agnostic(agnostic_stack, gtsegments, gtlabels)
avg_mAP = float(np.mean(dmap))
for i, v in zip(iou, dmap):
    print(f'mAP@{i:.1f} = {v:.2f}%')
print(f'AVG mAP = {avg_mAP:.2f}%')
print(f'AUC = {AUC:.4f}  AP = {AP:.4f}')

return AUC, avg_mAP
```

Loại bỏ `CLASM`-based `prob2`/`element_logits2` vì không dùng cho class-agnostic
metric. Per-class mAP cũ bị **bỏ** khỏi output log (user xác nhận chỉ quan tâm
class-agnostic).

## 8. Config (`src/option.py` bổ sung)

```python
# New args:
parser.add_argument('--train-json',
                    default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
parser.add_argument('--pos-weight-path', default='list/pos_weight_bin.npy')
parser.add_argument('--phase1-epochs', default=3, type=int)
parser.add_argument('--phase2-epochs', default=6, type=int)
parser.add_argument('--lambda1', default=0.1, type=float)
parser.add_argument('--lambda2', default=0.1, type=float)
```

`max_epoch`, `batch_size`, `lr`, `scheduler_*` giữ defaults hiện tại.

## 9. Files Impact Summary

| File | Change |
|---|---|
| `src/utils/dataset.py` | Load JSON, build `y_bin`, return thêm `y_bin` |
| `src/utils/tools.py` | `process_feat`/`process_split` trả thêm `sampled_indices` |
| `src/utils/compute_pos_weight.py` | NEW — script precompute `pos_weight_bin.npy` |
| `src/utils/detection_map.py` | Thêm `getDetectionMAP_agnostic` |
| `src/train.py` | Dataset unpacking, `get_lambda`, `soft_iou_loss`, tổng loss mới |
| `src/test.py` | `sigmoid(logits1)` path, class-agnostic mAP only, return `(AUC, avg_mAP)` |
| `src/option.py` | 5 args mới |
| `list/pos_weight_bin.npy` | NEW (generated, commit optional) |
| `model.py` | **không đổi** |

## 10. Open Questions / Future Work

- Phase split 3/3/4 epoch có thể cần tune theo convergence curve thực tế.
- `λ1=λ2=0.1` conservative — có thể cần quét 0.05/0.1/0.2/0.5 sau baseline.
- Soft-IoU có thể cần warmup per-video (chỉ apply trên anomaly videos) nếu video
  Normal dominate gradient. Theo dõi loss_iou curve trong Phase 2.
- Boundary regression head → branch riêng (`dev_boundary`).

## 11. Baseline Measurement (trước khi train dev_bce)

**Mục tiêu**: có số baseline class-agnostic `AVG mAP` từ checkpoint `main` hiện có
để Success Criteria (Section 12) tham chiếu đúng vào kết quả thực tế chứ không phải
ước lượng từ paper.

**Checkpoint có sẵn**: `model/model_ucf.pth` (trained trên `main`, weakly-supervised
MIL).

**Quy trình**:

1. Checkout tạm sang commit gốc của `main` (hoặc dùng code `main` hiện tại — test
   pipeline chưa đụng).
2. Cài đặt **trước** hàm `getDetectionMAP_agnostic` trong `src/utils/detection_map.py`
   (đây là phần duy nhất cần thiết để đo baseline — model và dataset không đổi).
3. Thêm flag `--eval-only` vào `test.py` in cả class-agnostic mAP song song với
   per-class mAP (log thêm, không thay thế) để có số so sánh trên cùng data.
4. Chạy:

   ```
   python src/test.py --model-path model/model_ucf.pth
   ```

5. Ghi lại: `AUC, AP, mAP@{0.1..0.5}, AVG mAP` — cả per-class (số cũ, reference)
   và class-agnostic (số mới, baseline thực cho dev_bce).

**Kết quả ghi lại ở**: `docs/superpowers/specs/2026-04-15-baseline-main.md` (tạo
sau khi đo).

**Thứ tự thực hiện trong implementation plan**:

1. Bước 0 (baseline): implement `getDetectionMAP_agnostic` + log thêm trong
   `test.py`, chạy trên `model_ucf.pth`, commit log kết quả.
2. Sau đó mới bắt đầu các thay đổi dataset/loss/training của dev_bce.

→ Cách làm này đảm bảo `getDetectionMAP_agnostic` được viết & verify 1 lần, dùng
cho cả baseline và evaluation dev_bce.

## 12. Success Criteria

- **Must**: `AVG mAP` (class-agnostic) Phase 3 (epoch 7+) > `AVG mAP` Phase 1 (epoch
  1-3, tương đương main baseline) ít nhất **+5 điểm** trên test set.
- **Should**: `AVG mAP` Phase 3 > Phase 2 (chứng minh `L_iou` có contribution).
- **May**: AUC Phase 3 có thể thấp hơn main baseline, chấp nhận được nếu ≤ **−2 điểm**.
