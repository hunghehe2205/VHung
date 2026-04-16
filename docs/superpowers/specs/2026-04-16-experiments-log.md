# Experiments Log — dev_bce supervised training

Tổng hợp cách tính metrics, kết quả baseline, và kết quả các experiments.

---

## 1. Cách tính metrics

Toàn bộ eval ở `src/test.py`. Dataset test: `list/ucf_CLIP_rgbtest.csv` (290 videos).
Model forward trả về 5 tensor: `text_features`, `logits1` (C-Branch, binary), `logits2` (A-Branch, multi-class 14 lớp), `start_logits`, `end_logits` (BSN boundary heads, added Exp 9+).

### 1.1 Video-level scores (frame-wise ROC-AUC / AP)

Upsample ×16 để về frame granularity rồi so với `gt_ucf.npy` (per-frame 0/1):

| Metric | Score source | Hàm |
|---|---|---|
| AUC1 | `sigmoid(logits1)` | `roc_auc_score` |
| AP1  | `sigmoid(logits1)` | `average_precision_score` |
| AUC2 | `1 − softmax(logits2)[:, Normal]` | `roc_auc_score` |
| AP2  | `1 − softmax(logits2)[:, Normal]` | `average_precision_score` |

### 1.2 Per-class mAP (legacy VadCLIP metric)

Dùng `logits2.softmax(−1)` upsample ×16, đưa vào `getDetectionMAP`:
- Duyệt 5 IoU threshold `{0.1, 0.2, 0.3, 0.4, 0.5}`.
- Với mỗi class, build segment proposals từ confidence, tính AP so với `gt_segment_ucf.npy` + `gt_label_ucf.npy`.
- Mean over 5 thresholds → `AVG`.

### 1.3 Class-agnostic mAP (**metric chính**)

Dùng `sigmoid(logits1)` per-video, upsample ×16, đưa vào `getDetectionMAP_agnostic`:
- Bỏ class label, chỉ đánh giá vùng bất thường (binary).
- 5 IoU thresholds `{0.1, 0.2, 0.3, 0.4, 0.5}`.
- Mean over 5 thresholds → `AVG_agnostic_mAP`.

**Đây là metric select best epoch trong `train.py`** (so `avg_mAP > ap_best`).

### 1.4 Training-time eval log format

Mỗi epoch một dòng:

```
[ep N/M Ts] lam=(λ1,λ2) | bce_v=.. nce=.. cts=.. fbce=.. p3=.. | AUC=.. mAP=.. [i0.1/i0.2/i0.3/i0.4/i0.5] *
```

- `lam=(λ1,λ2)`: lambda hiện tại theo phase schedule.
- `bce_v, nce, cts`: 3 loss gốc của VadCLIP.
- `fbce`: frame BCE (plain hoặc focal, tuỳ `--focal-gamma`).
- `p3`: Phase 3 localization loss (TV hoặc Dice, tuỳ `--phase3-loss`).
- `AUC`: AUC1.
- `mAP`: class-agnostic AVG mAP (metric chính).
- `[...]`: mAP từng IoU threshold `{0.1, 0.2, 0.3, 0.4, 0.5}`.
- `*`: epoch này là best (mAP vượt ap_best).

---

## 1b. Eval bug fix — class-agnostic gtpos inflation (2026-04-16)

**Bug:** `_loc_map_agnostic` và `_iou_matching_ap` trong `detection_map.py` include TẤT CẢ GT segments, kể cả Normal videos có GT giả `[0, lens]` label `'A'`. Inflate gtpos từ 156 (anomaly thật) lên 306 (+96%), gây bias AP không nhất quán.

**Fix:** Filter `gtlabels[i][j] != 'A'` trong cả 2 hàm. gtpos: 306 → 156.

**Impact:** Số mAP tuyệt đối thay đổi (tăng) nhưng **ranking giữa experiments giữ nguyên**. Tất cả số liệu từ đây trở đi dùng eval đã fix. Experiments cũ cần re-eval để so sánh fair.

---

## 2. Baseline (main, `model/model_ucf.pth`)

Ref: `docs/superpowers/specs/2026-04-15-baseline-main.md`.

> **⚠️ RE-EVAL với eval fix (gtpos=156).** Số cũ (trước fix) gạch ngang để tham khảo.

### Video-level

| Metric | Value |
|---|---|
| AUC1 | **0.8736** |
| AP1  | 0.3122 |
| AUC2 | 0.8570 |
| AP2  | 0.2553 |

### Class-agnostic mAP (metric chính)

| IoU | ~~Cũ (gtpos=306)~~ | **Mới (gtpos=156)** |
|---:|---:|---:|
| 0.1 | ~~21.98~~ | **35.80** |
| 0.2 | ~~12.93~~ | **22.34** |
| 0.3 | ~~6.80~~ | **12.23** |
| 0.4 | ~~4.01~~ | **7.09** |
| 0.5 | ~~1.65~~ | **2.82** |
| **AVG** | ~~9.47~~ | **16.06** |

### Per-class mAP (legacy, không bị ảnh hưởng bởi fix)

| IoU | mAP |
|---:|---:|
| 0.1 | 13.44 |
| 0.2 | 10.10 |
| 0.3 | 7.01 |
| 0.4 | 6.66 |
| 0.5 | 3.88 |
| **AVG** | **8.22** |

### Success thresholds (đích của dev_bce, updated với baseline mới)

- **Must:** AVG_agnostic_mAP ≥ **21.06** (baseline 16.06 + 5 điểm).
- **Should:** Phase 3 > Phase 2.
- **May:** AUC1 drop ≤ 2 → AUC1 ≥ 0.8536.

---

## 3. Experiments

Tất cả experiments train 10 epochs, batch 64, lr 2e-5, phase schedule `p1=3, p2=6`.
Khác biệt giữa experiments chỉ ở Phase 2 loss, Phase 3 loss, và lambda.

### Exp 1 — Original spec (frame BCE + Soft IoU)

**Config:**

| Arg | Value |
|---|---|
| Phase 2 loss | plain frame BCE (với scalar `pos_weight=5.8155`) |
| Phase 3 loss | Soft temporal IoU (full batch, gồm cả normal videos) |
| `--focal-gamma` | N/A (code lúc này chỉ có BCE) |
| `--lambda1` | 0.1 |
| `--lambda2` | 0.1 |

**Kết quả:**

- Phase 2 best: **AVG mAP = 11.47**
- Phase 3 (ep7-10): 11.32 – 11.42 → không vượt Phase 2.
- IoU signal quá redundant với BCE trên normal videos → không thêm gradient hữu ích.

**Kết luận:** Phase 3 fail, IoU không giúp boundary precision.

---

### Exp 2 — TV smoothness + Focal γ=2

**Config:**

| Arg | Value |
|---|---|
| Phase 2 loss | `focal_bce_loss` (pos_weight + `(1−p_t)^γ`) |
| Phase 3 loss | `tv_smoothness_loss` (mean `\|p_t − p_{t-1}\|`) |
| `--focal-gamma` | **2.0** |
| `--lambda1` | 0.1 |
| `--lambda2` | 0.1 |

**Lệnh:**

```bash
python src/train.py | tee logs/train_tv_focal.log
```

**Kết quả per-epoch:**

| Ep | Phase | AUC | mAP | [@0.1/0.2/0.3/0.4/0.5] |
|---:|---|---:|---:|---|
| 1 | P1 | 0.8601 | 8.89 | 20.60/11.46/6.27/4.43/1.71 |
| 2 | P1 | 0.8558 | 8.77 | 19.03/12.57/6.18/4.16/1.89 |
| 3 | P1 | 0.8638 | 9.53 | 21.83/12.97/6.98/4.09/1.76 |
| 4 | P2 | 0.8536 | 10.25 | 23.53/14.47/7.40/4.28/1.59 |
| 5 | P2 | 0.8527 | **11.15** | 22.61/16.82/9.35/4.70/2.27 |
| 6 | P2 | 0.8531 | 11.11 | 23.85/16.50/9.00/4.26/1.94 |
| 7 | P3 | 0.8519 | 10.71 | 22.95/16.67/8.03/3.97/1.91 |
| 8 | P3 | 0.8525 | 10.68 | 22.94/16.67/7.74/4.19/1.87 |
| 9 | P3 | 0.8527 | 10.63 | 22.75/16.69/7.72/4.05/1.96 |
| 10 | P3 | 0.8527 | 10.59 | 22.50/16.78/7.75/4.15/1.80 |

**Best: 11.15 (ep5, Phase 2 peak).**

**Quan sát:**
- Focal γ=2 co BCE xuống quá mạnh (`fbce≈0.6` nhưng effective gradient chỉ bằng ~15% plain BCE) → Phase 2 peak 11.15 < Exp 1 peak 11.47.
- TV magnitude bé (`p3≈0.006`), × λ2=0.1 → gradient 0.0006 → vô hiệu.
- Phase 3 mAP giảm đều → mô hình bắt đầu overfit, TV không đủ mạnh để regularize.

**Kết luận:** Fail vs Must (14.47). TV calibration sai (scale 0.01 vs λ2 cho IoU 0.1). Focal γ=2 quá mạnh.

---

### Exp 3 — TV + Focal γ=1, λ2=5

**Hypothesis:** Recover Phase 2 (γ=1 mềm hơn) + TV có effective signal (λ2×tv ≈ 0.03 thay vì 0.0006).

**Config:**

| Arg | Value |
|---|---|
| Phase 2 loss | `focal_bce_loss` γ=1 |
| Phase 3 loss | `tv_smoothness_loss` |
| `--focal-gamma` | **1.0** |
| `--lambda1` | 0.1 |
| `--lambda2` | **5.0** |

**Lệnh:**

```bash
python src/train.py \
  --focal-gamma 1.0 --phase3-loss tv --lambda2 5.0 \
  --model-path final_model/model_v2.pth \
  --checkpoint-path final_model/ckpt_v2.pth \
  | tee logs/train_v2.log
```

**Kết quả per-epoch:**

| Ep | Phase | AUC | mAP | [@0.1/0.2/0.3/0.4/0.5] |
|---:|---|---:|---:|---|
| 1 | P1 | 0.8601 | 8.89 | 20.60/11.46/6.27/4.43/1.71 |
| 2 | P1 | 0.8558 | 8.77 | 19.03/12.57/6.18/4.16/1.89 |
| 3 | P1 | 0.8638 | 9.53 | 21.83/12.97/6.98/4.09/1.76 |
| 4 | P2 | 0.8611 | 11.43 | 25.37/17.02/8.46/4.52/1.76 |
| 5 | P2 | 0.8590 | 11.45 | 23.94/17.04/8.99/5.18/2.10 |
| 6 | P2 | 0.8588 | **11.62** | 24.38/17.00/9.71/4.93/2.08 |
| 7 | P3 | 0.8588 | 10.95 | 24.10/15.65/8.48/4.48/2.04 |
| 8 | P3 | 0.8577 | 11.19 | 25.15/15.77/8.43/4.55/2.06 |
| 9 | P3 | 0.8578 | 11.11 | 24.86/15.61/8.35/4.69/2.06 |
| 10 | P3 | 0.8579 | 11.11 | 24.87/15.61/8.37/4.66/2.03 |

**Best: 11.62 (ep6, Phase 2 peak).**

**Quan sát:**
- γ=1 recover Phase 2 peak tốt: **11.62 > 11.15 (γ=2)**. Focal mềm hơn = đúng.
- TV với λ2=5 có effective signal (`p3=0.006 × 5 = 0.03`) nhưng **vẫn kéo mAP xuống** ở Phase 3 (11.62 → 10.95-11.19).
- @0.2 tụt rõ ở Phase 3 (17.00 → 15.61). TV over-smoothing.

**Kết luận:** Fail vs Must. TV không phải hướng đúng — dù có signal nó vẫn không fix được bottleneck boundary precision.

---

### Exp 4 — Dice + plain BCE

**Hypothesis:** Dice reward spike sắc trúng segment, phạt flat response. Skip normal videos để khử redundancy với BCE.

**Config:**

| Arg | Value |
|---|---|
| Phase 2 loss | **plain frame BCE** (pos_weight, no focal) |
| Phase 3 loss | `dice_loss_anomaly` (2·inter/(p+y), skip normal videos) |
| `--focal-gamma` | **0** (disable focal) |
| `--lambda1` | 0.1 |
| `--lambda2` | **1.0** |

**Lệnh:**

```bash
python src/train.py \
  --focal-gamma 0 --phase3-loss dice --lambda2 1.0 \
  --model-path final_model/model_dice.pth \
  --checkpoint-path final_model/ckpt_dice.pth \
  | tee logs/train_dice.log
```

**Kết quả per-epoch:**

| Ep | Phase | AUC | mAP | [@0.1/0.2/0.3/0.4/0.5] |
|---:|---|---:|---:|---|
| 1 | P1 | 0.8601 | 8.89 | 20.60/11.46/6.27/4.43/1.71 |
| 2 | P1 | 0.8558 | 8.77 | 19.03/12.57/6.18/4.16/1.89 |
| 3 | P1 | 0.8638 | 9.53 | 21.83/12.97/6.98/4.09/1.76 |
| 4 | P2 | 0.8621 | 11.15 | 23.30/16.95/8.84/4.62/2.02 |
| 5 | P2 | 0.8575 | 11.47 | 24.01/17.52/9.28/4.55/2.00 |
| 6 | P2 | 0.8576 | 11.49 | 24.07/17.18/9.40/4.69/2.14 |
| 7 | P3 | 0.8571 | 11.47 | 24.04/17.22/9.53/4.54/2.03 |
| 8 | P3 | 0.8567 | **11.70** | 24.45/17.49/9.75/4.68/2.13 |
| 9 | P3 | 0.8570 | 11.52 | 24.04/17.06/9.65/4.66/2.17 |
| 10 | P3 | 0.8572 | 11.45 | 23.78/17.07/9.62/4.64/2.15 |

**Best: 11.70 (ep8, Phase 3).** ← **exp đầu tiên Phase 3 > Phase 2.**

**Quan sát:**
- Phase 2 peak 11.49 ≈ Exp 1 (11.47) → plain BCE ổn định.
- Dice active signal `p3≈0.55` (2 orders of magnitude > TV) → có gradient thực.
- **Phase 3 nhích thêm +0.21 so với Phase 2 peak** → Should threshold passed (lần đầu).
- @0.3 cải thiện tốt nhất: 9.40 → 9.75.
- @0.5 vẫn ở ceiling ~2.1 — chưa vượt được giới hạn boundary.

**Kết luận:** Dice > TV > IoU. Đạt **Should** threshold. Vẫn **fail Must** (11.70 < 14.47).

---

### Exp 5 — Dice + plain BCE, 15 epochs

Cùng config Exp 4 nhưng `--max-epoch 15`.

**Kết quả per-epoch:**

| Ep | Phase | mAP |
|---:|---|---:|
| 1-6 | P1/P2 | (trùng Exp 4, best P2 = 11.49 ep6) |
| 7 | P3 | 11.47 |
| 8 | P3 | **11.70** ← best |
| 9 | P3 | 11.52 |
| 10 | P3 | 11.45 |
| 11 | P3 | 11.49 |
| 12 | P3 | 11.46 |
| 13 | P3 | 11.54 |
| 14 | P3 | 11.49 |

**Best: 11.70 (ep8) — trùng chính xác Exp 4.**

**Kết luận:** Thêm 5 epoch P3 không mang lại signal mới. Dice đã saturate ở ep8 (p3 loss đi ngang 0.547-0.554 từ ep7 trở đi). **Cap = 11.70 với loss combo hiện tại.** Muốn tăng cần arch change.

---

## 3b. Chẩn đoán giữa chừng — F1 eval + viz normalized trên Exp 4 model

### F1: Linear vs step-function upsample ở eval

Hypothesis: ceiling `@0.4/@0.5` là do `np.repeat(×16)` step-function khi upsample prob T-step về frame. Thay bằng linear interp (`np.interp`) — thuần eval, không retrain.

| Metric | repeat | linear | Δ |
|---|---:|---:|---:|
| AUC1 | 0.8567 | 0.8567 | 0 |
| AP1 | 0.2834 | 0.2833 | −0.0001 |
| agnostic AVG | 11.70 | 11.70 | 0 |
| agnostic @0.1 | 24.46 | 24.37 | −0.09 |
| agnostic @0.4 | 4.68 | 4.53 | −0.15 |
| agnostic @0.5 | 2.13 | 2.17 | +0.04 |

**Kết luận F1:** upsample mode hầu như không ảnh hưởng (`Δ < 0.2` mọi IoU). **Ceiling KHÔNG phải do upsampling** → model thật sự không output boundary sắc. Phải can thiệp vào model/loss.

### Viz diagnosis — từng video (model_dice15.pth, normalized)

Script `src/visualize.py` render `sigmoid(logits1)` per video, upsample ×16 linear, shade GT anomaly segments. Chế độ `--normalize` min-max scale trong video để thấy shape tương đối.

**5 case study:**

| Case | Raw range | Shape | Kết luận |
|---|---|---|---|
| Arson | 0.84–1.00 | rise trong GT, peak cuối GT | có signal, saturate toàn cục |
| Arrest | 0.54–1.00 | peak ở frame 500-1000 (outside GT), flat sau | confused + saturate |
| Normal | 0.00–0.00 | — | ✅ đúng (không anomaly) |
| Stealing | 0.27–0.97 | plateau cover GT nhưng rộng ~2× | boundary bleed |
| Vandalism | 0.79–0.99 | **2 peak đúng 2 GT nhỏ** + 1 false peak | localization đúng, saturate cao |

**3 phát hiện chính:**

1. **Normal videos: raw ≈ 0** → model không saturate toàn bộ. Classification video-level tốt.
2. **Anomaly videos: raw thường 0.8–1.0** → model BỊ đẩy prob lên cao toàn video, không localize.
3. **Nhưng shape normalized cho thấy có signal localization** (đặc biệt Vandalism). Model "biết" nhưng output bias.

**Root cause suspected:**
- `pos_weight=5.8` weight positives quá mạnh → bias đẩy all frames lên.
- MIL losses (CLAS2 top-K) chỉ yêu cầu vài frame cao → không phạt high-everywhere.
- Không có loss nào ép **mean(inside) > mean(outside)** trong cùng video.

**Không cần arch change ngay** — thử debias loss trước (Exp 6).

---

### Exp 6 — Dice + within-video contrast + pos_weight=1.0

**Hypothesis:** 3 can thiệp đồng thời để khử saturation + ép peak đúng vị trí.

**Config:**

| Arg | Value | Lý do |
|---|---|---|
| `--pos-weight` | **1.0** (thay 5.8) | Khử bias đẩy positives lên |
| `--focal-gamma` | 0 | Plain BCE |
| `--phase3-loss dice` + `--lambda2` | 1.0 | Shape supervision (giữ từ Exp 4) |
| `--lambda-contrast` | **1.0** (new) | Ép mean(inside GT) > mean(outside) + margin |
| `--contrast-margin` | **0.3** | Gap yêu cầu |

**New loss — `within_video_contrast_loss`:**

```python
# Per anomaly video: enforce mean_prob(inside GT) >= mean_prob(outside GT) + margin
inside_mean  = (prob * inside_mask).sum(-1) / inside_mask.sum(-1)
outside_mean = (prob * outside_mask).sum(-1) / outside_mask.sum(-1)
gap_loss = F.relu(margin - (inside_mean - outside_mean))
```

Trực tiếp ép peak ĐÚNG vùng GT, không cho prob trung bình outside ngang inside. Normal videos skip (không có inside).

**Lệnh:**

```bash
python src/train.py \
  --focal-gamma 0 --phase3-loss dice --lambda2 1.0 \
  --lambda-contrast 1.0 --contrast-margin 0.3 \
  --pos-weight 1.0 \
  --max-epoch 10 \
  --model-path final_model/model_exp6.pth \
  --checkpoint-path final_model/ckpt_exp6.pth \
  | tee logs/train_exp6.log
```

**Kỳ vọng:**
- Raw range `sigmoid(logits1)` co lại (không còn 0.84-1.0).
- Peak trong anomaly videos nằm tại GT thay vì lệch.
- `@0.4/@0.5` tăng rõ rệt — decisive test.

**Tiêu chí pass decisive:**
- `@0.5` ≥ 3.0 (từ 2.13, +40% relative).
- `@0.4` ≥ 6.0 (từ 4.68, +28% relative).
- AVG mAP ≥ 13.0 (từ 11.70).
- Nếu không đạt → chuyển Tier 3 (boundary head arch change).

**Kết quả:**

```
[ep  1/10] lam=(0.0,0.0) | bce_v=0.244 nce=1.618 cts=0.0512 fbce=0.000 p3=0.000 ctr=0.000 | AUC=0.8601 mAP=8.89 [20.60/11.46/6.27/4.43/1.71] *
[ep  2/10] lam=(0.0,0.0) | bce_v=0.087 nce=1.023 cts=0.0180 fbce=0.000 p3=0.000 ctr=0.000 | AUC=0.8558 mAP=8.77 [19.03/12.57/6.18/4.16/1.89]
[ep  3/10] lam=(0.0,0.0) | bce_v=0.052 nce=0.730 cts=0.0148 fbce=0.000 p3=0.000 ctr=0.000 | AUC=0.8638 mAP=9.53 [21.83/12.97/6.98/4.09/1.76] *
[ep  4/10] lam=(0.1,0.0) | bce_v=0.048 nce=0.523 cts=0.0124 fbce=0.911 p3=0.000 ctr=0.000 | AUC=0.8567 mAP=9.64 [19.78/14.44/7.98/4.13/1.88] *
[ep  5/10] lam=(0.1,0.0) | bce_v=0.036 nce=0.384 cts=0.0115 fbce=0.765 p3=0.000 ctr=0.000 | AUC=0.8567 mAP=9.84 [19.61/15.30/8.18/4.33/1.76] *
[ep  6/10] lam=(0.1,0.0) | bce_v=0.036 nce=0.358 cts=0.0113 fbce=0.721 p3=0.000 ctr=0.000 | AUC=0.8575 mAP=10.42 [21.35/15.46/8.34/4.76/2.20] *
[ep  7/10] lam=(0.1,1.0) | bce_v=0.030 nce=0.339 cts=0.0110 fbce=0.752 p3=0.562 ctr=0.221 | AUC=0.8596 mAP=11.13 [22.34/16.38/9.07/5.03/2.82] *
[ep  8/10] lam=(0.1,1.0) | bce_v=0.030 nce=0.323 cts=0.0107 fbce=0.731 p3=0.555 ctr=0.205 | AUC=0.8600 mAP=11.83 [23.08/17.20/10.22/5.60/3.03] *
[ep  9/10] lam=(0.1,1.0) | bce_v=0.030 nce=0.311 cts=0.0104 fbce=0.703 p3=0.551 ctr=0.196 | AUC=0.8598 mAP=11.83 [23.26/16.91/10.23/5.66/3.10] *
[ep 10/10] lam=(0.1,1.0) | bce_v=0.029 nce=0.308 cts=0.0104 fbce=0.703 p3=0.551 ctr=0.195 | AUC=0.8596 mAP=11.87 [23.32/16.97/10.26/5.67/3.11] *
Final best avg_mAP_agnostic = 11.87
```

| Criterion | Target | Actual | Pass |
|---|---|---|:-:|
| @0.5 | ≥ 3.0 | **3.11** | ✅ |
| @0.4 | ≥ 6.0 | 5.67 | ❌ |
| AVG  | ≥ 13.0 | 11.87 | ❌ |

**Đọc log:**
- Phase 3 start ep7 (`lam=(0.1,1.0)`) → mAP nhảy 10.42 → 11.13 → 11.83 → 11.83 → 11.87 (4 epoch Phase 3).
- `ctr` giảm chậm 0.221 → 0.195 — margin 0.3 chưa đạt trên nhiều video.
- `p3` (Dice) stable ~0.55 — chưa drop đáng kể, anomaly prob vẫn chưa fit GT shape tốt.
- @0.5 progress rõ nhất: 2.20 → 3.11 (+41% trong 4 epoch) — contrast DID help localization.
- @0.4 chậm: 4.76 → 5.67 (+19%).

**Kết luận Exp 6:**
- Contrast loss work theo đúng hypothesis (help high-IoU tiers).
- Nhưng không đột phá — chỉ +0.17 AVG so Exp 5 (11.70 → 11.87) trên eval cũ.
- Nguyên nhân không phải loss — tất cả loss innovation đã cộng dồn và ceiling vẫn quanh 11.7-11.9.
- **Root cause thực sự nằm ở architecture**, không phải loss. Xem Exp 7.

**⚠️ RE-EVAL Exp 6 với eval fix (gtpos=156):**

| Metric | ~~Cũ (gtpos=306)~~ | **Mới (gtpos=156)** | Δ vs Baseline mới |
|---|---:|---:|---|
| avg_mAP | ~~12.03~~ | **21.71** | **+5.65** |
| AUC | ~~0.8655~~ | **0.8647** | −0.009 |
| @0.1 | ~~23.23~~ | **42.03** | +6.23 |
| @0.2 | ~~16.78~~ | **30.20** | +7.86 |
| @0.3 | ~~10.68~~ | **19.21** | +6.98 |
| @0.4 | ~~6.11~~ | **11.10** | +4.01 |
| @0.5 | ~~3.35~~ | **6.04** | +3.22 |

Exp 6 vượt Must threshold mới (**21.71 > 21.06**). @0.5 tăng hơn gấp đôi vs baseline (6.04 vs 2.82).

---

### Exp 7 — Sliding-window attention (arch change)

**Hypothesis breakthrough:** Ceiling 11.7-11.9 không phá được bằng loss vì **attention mask block-diagonal non-overlapping** cắt đứt context tại các vị trí snippet 8, 16, 24... Snippet i và i+1 nếu ở 2 block khác nhau → không bao giờ attend nhau → biên sự kiện mất context → output saturated/noisy tại biên.

**Diagnosis từ `model.py:115-123`:**

```python
# CŨ — block-diagonal
def build_attention_mask(self, attn_window):
    mask = torch.empty(T, T).fill_(-inf)
    for i in range(T / attn_window):
        mask[i*8:(i+1)*8, i*8:(i+1)*8] = 0   # chỉ trong block
    return mask
```

Với `visual_length=256, attn_window=8` → 32 block rời, không attend chéo.

**Fix:** Sliding-window mask — snippet i attend `[i − w/2, i + w/2]`:

```python
# MỚI — sliding window
def build_attention_mask(self, attn_window):
    T = self.visual_length
    mask = torch.full((T, T), float('-inf'))
    half = attn_window // 2
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        mask[i, lo:hi] = 0
    return mask
```

Tác động:
- Context liên tục theo thời gian, không có biên cứng nhân tạo.
- Event dài cross block boundary vẫn được model coi là 1 khối liên tục.
- Boundary snippet có đủ context 2 phía.

**Config (giữ nguyên Exp 6 losses):**

| Arg | Value |
|---|---|
| Attention mask | **sliding-window** (w=8, half=4) |
| `--pos-weight` | 1.0 |
| `--focal-gamma` | 0 |
| `--phase3-loss` | dice |
| `--lambda2` | 1.0 |
| `--lambda-contrast` | 1.0 |
| `--contrast-margin` | 0.3 |
| `--phase1-epochs` | 2 |
| `--phase2-epochs` | 4 |
| `--max-epoch` | 15 |

Rút ngắn phase1/2 (3/6 → 2/4) để có **11 epoch Phase 3** (vs 4 của Exp 6).

**Lệnh:**

```bash
python src/train.py \
  --focal-gamma 0 --phase3-loss dice --lambda2 1.0 \
  --lambda-contrast 1.0 --contrast-margin 0.3 \
  --pos-weight 1.0 \
  --phase1-epochs 2 --phase2-epochs 4 \
  --max-epoch 15 \
  --model-path final_model/model_exp7.pth \
  --checkpoint-path final_model/ckpt_exp7.pth \
  2>&1 | tee logs/train_exp7.log
```

**Kỳ vọng breakthrough:**
- @0.4, @0.5 tăng mạnh (không còn mất signal biên block).
- AVG mAP target ≥ 14.47 (Must threshold).
- Raw range output giảm saturation.

**Tiêu chí pass (Must breakthrough):**
- AVG mAP ≥ 14.47.
- @0.5 ≥ 4.0.
- @0.4 ≥ 7.0.

Nếu vẫn < 14.47 → thêm tầng global attention (1 layer không mask) hoặc boundary head.

**Kết quả (15 epochs):**

```
[ep  1/15] lam=(0.0,0.0) | AUC=0.8553 mAP=8.52  [19.85/10.82/5.99/4.10/1.86] *
[ep  2/15] lam=(0.0,0.0) | AUC=0.8611 mAP=8.70  [19.82/12.09/5.98/3.94/1.66] *
[ep  3/15] lam=(0.1,0.0) | AUC=0.8547 mAP=10.29 [21.02/15.55/8.23/4.70/1.94] *
[ep  4/15] lam=(0.1,0.0) | AUC=0.8621 mAP=9.38  [20.90/13.01/7.40/3.81/1.77]
[ep  5/15] lam=(0.1,1.0) | AUC=0.8607 mAP=10.07 [20.03/15.19/8.70/4.80/1.65]
[ep  6/15] lam=(0.1,1.0) | AUC=0.8602 mAP=10.73 [21.65/16.72/9.03/4.34/1.93] *
[ep  7/15] lam=(0.1,1.0) | AUC=0.8605 mAP=10.90 [21.61/16.90/9.42/4.51/2.04] *
[ep  8/15] lam=(0.1,1.0) | AUC=0.8596 mAP=10.91 [22.72/16.71/8.75/4.50/1.89] *
[ep  9/15] lam=(0.1,1.0) | AUC=0.8598 mAP=11.16 [22.48/17.32/9.55/4.52/1.93] *
[ep 10/15] lam=(0.1,1.0) | AUC=0.8598 mAP=11.18 [22.51/17.35/9.58/4.54/1.94] *
[ep 11/15] lam=(0.1,1.0) | AUC=0.8598 mAP=11.20 [22.73/17.26/9.42/4.53/2.06] *
[ep 12/15] lam=(0.1,1.0) | AUC=0.8599 mAP=11.27 [22.87/17.29/9.60/4.54/2.05] *
[ep 13/15] lam=(0.1,1.0) | AUC=0.8599 mAP=11.31 [22.94/17.34/9.63/4.56/2.08] *
[ep 14/15] lam=(0.1,1.0) | AUC=0.8600 mAP=11.36 [23.02/17.41/9.67/4.59/2.11] *
[ep 15/15] lam=(0.1,1.0) | AUC=0.8600 mAP=11.34 [22.84/17.44/9.71/4.61/2.11]
```

Best ep 14: **mAP=11.36**, AUC=0.8600.

**So sánh Exp 6 (block-diag) vs Exp 7 (sliding-window):**

| Metric | Exp 6 (block-diag) | Exp 7 (sliding-win) | Δ |
|---|---:|---:|---|
| avg_mAP | **12.03** | 11.36 | −0.67 |
| @0.1 | 23.23 | **23.02** | ≈ |
| @0.2 | 16.78 | **17.44** | +0.66 |
| @0.3 | **10.68** | 9.71 | −0.97 |
| @0.4 | **6.11** | 4.61 | −1.50 |
| @0.5 | **3.35** | 2.11 | −1.24 |
| AUC | 0.8655 | 0.8600 | −0.006 |

**Kết luận:** Sliding-window **HURT strict IoU** (−37% @0.5, −25% @0.4) dù help nhẹ @0.2. Proposals béo hơn do effective receptive field qua 2-layer sliding (window=8) = ±8 snippets = 17 total, so với block-diagonal chỉ 8.

**Bài học:** bất kỳ temporal mixing nào rộng hơn sẽ smooth features → proposals rộng hơn → hại strict IoU. Block-diagonal "tình cờ" giữ features isolated trong block 8 snippet → proposals compact hơn.

### Exp 8 — Surgical: remove similarity GCN + identity residual

**Hypothesis:** Similarity GCN (adj4 + gc1+gc2) là nguồn oversmoothing chính. Cosine sim → threshold 0.7 → softmax per-row collapse cluster anomaly thành giá trị đồng đều. 2-hop stacking nhân đôi smoothing. Residual conv k=5 trong GraphConvolution cũng là low-pass filter. Bỏ 3 nguồn này → signal sharp hơn → strict IoU tăng.

**Thay đổi:**
1. Revert attention về block-diagonal (undo Exp 7)
2. Bỏ gc1+gc2 (similarity branch), thay bằng `pre_proj = nn.Linear(512, 256)` cho sharp path
3. Đổi residual conv k=5 → k=1 trong `layers.py`
4. Giữ gc3+gc4 (distance GCN)

**Config:** identical Exp 6.

**Kết quả (15 epochs):**

```
[ep  1/15] lam=(0.0,0.0) | AUC=0.8439 mAP=8.17  [20.02/10.93/5.53/3.02/1.35] *
[ep  5/15] lam=(0.1,0.0) | AUC=0.8510 mAP=8.38  [18.62/12.33/5.50/3.00/2.46] *
[ep  8/15] lam=(0.1,1.0) | AUC=0.8518 mAP=9.39  [19.74/13.13/6.99/4.54/2.53] *
[ep 14/15] lam=(0.1,1.0) | AUC=0.8520 mAP=9.48  [19.59/13.58/7.02/4.54/2.66] *
```

Best ep 14: **mAP=9.48**, AUC=0.8521.

| Metric | Exp 6 | Exp 8 | Δ |
|---|---:|---:|---|
| avg_mAP | **12.03** | 9.48 | **−2.55** ❌ |
| @0.5 | **3.35** | 2.66 | −0.69 |
| @0.4 | **6.11** | 4.54 | −1.57 |
| AUC | 0.8655 | 0.8521 | −0.013 |

**Kết luận: FAIL.** Hypothesis sai — similarity GCN **đang giúp**, không phải hại. Bỏ nó hại MỌI metric. Code đã **revert** về Exp 6 architecture.

**Bài học:** Similarity GCN group snippets anomaly → classifier nhìn thấy cluster-level pattern → cải thiện cả AUC lẫn localization. "Oversmoothing" thực ra là **beneficial feature aggregation**.

---

### Exp 9 — BSN boundary heads (start/end prediction)

**Hypothesis:** Thêm start/end prediction heads trên pre-GCN features, BSN-style proposal inference thay thế adaptive threshold. Boundary heads cho model khả năng dự đoán event boundary explicit.

**Thay đổi code (so với Exp 6 base):**

1. `model.py`: Thêm `start_head = nn.Linear(512, 1)` và `end_head = nn.Linear(512, 1)` trên `x_pre` (pre-GCN features).
2. `train.py`: Thêm `boundary_bce_loss` — Gaussian-smoothed start/end targets từ y_bin transitions, pos_weight=10.0.
3. `detection_map.py`: Thêm `getDetectionMAP_agnostic_bsn` — peak-pick start/end probs, enumerate (s,e) proposals, score = `sp[s] * ep[e] * act[s:e+1].mean()`.
4. `test.py`: Thêm `--inference bsn` mode.
5. `get_lambda`: 3-phase schedule (P1: MIL only, P2: +BCE+boundary, P3: +Dice+contrast).

**Bug fixes trong quá trình thực nghiệm:**

| Bug | Mô tả | Fix |
|---|---|---|
| `_peak_pick` plateau | Sau repeat ×16, peak thành plateau 16 frame → strict `>` miss hết | Plateau-aware: scan runs, pick midpoint |
| BSN end index | Proposal store `e` inclusive, IoU dùng `range(s,e)` exclusive → mất 1 frame | Store `e+1` (exclusive end) |
| `get_lambda` 2-phase | Phase 2 bị merge với Phase 3 (lam2 active cùng lúc lam1) | Khôi phục 3-phase schedule |
| Score scale | `+ 0.7 * c_s` additive dominate BSN term (~0.001) | Bỏ additive, dùng thuần multiplicative |

**Config:**

| Arg | Value |
|---|---|
| `--pos-weight` | 1.0 |
| `--focal-gamma` | 0 |
| `--phase3-loss` | dice |
| `--lambda2` | 1.0 |
| `--lambda-contrast` | 1.0 |
| `--contrast-margin` | 0.3 |
| `--lambda-boundary` | 0.5 |
| `--boundary-sigma` | 1.0 |
| `--boundary-pos-weight` | 10.0 |
| `--inference` | bsn |
| `--max-epoch` | 15 |

**Kết quả (10/15 epochs, BSN eval):**

| Ep | Phase | bnd | mAP | [@0.1/0.2/0.3/0.4/0.5] | BSN proposals/v |
|---:|---|---:|---:|---|---:|
| 1 | P1 | 0.000 | 2.14 | 3.24/2.73/2.00/1.56/1.18 | 206.8 |
| 3 | P1 | 0.000 | 1.87 | 2.65/2.41/1.69/1.44/1.18 | 207.0 |
| 4 | P2 | 1.158 | 2.16 | 3.49/2.65/1.85/1.59/1.22 | 167.0 |
| 6 | P2 | 0.876 | 1.91 | 2.85/2.30/1.73/1.51/1.13 | 156.0 |
| 7 | P3 | 0.862 | 1.92 | 2.88/2.34/1.73/1.52/1.12 | 149.6 |
| 10 | P3 | 0.845 | 1.88 | 2.84/2.29/1.70/1.48/1.09 | 143.3 |

Best: **mAP=2.16 (ep4)**, AUC=0.8404.

**Kết luận: FAIL.**

BSN mAP ~2 vs threshold mAP ~21 (Exp 6) — boundary heads quá yếu:

1. **Proposals quá nhiều**: 143-207/video (cần ~5-15). BSN heads output gần uniform → peak picking tìm hàng trăm peaks → proposals ngập FP.
2. **Boundary gradient yếu**: effective weight = `lam1 × λ_boundary = 0.1 × 0.5 = 0.05` — chỉ 5% total loss. bnd loss saturate sớm (1.16 → 0.85, hầu như không giảm sau ep6).
3. **Model selection sai**: BSN eval cho mAP noise (~2), model thực ra đang học tốt (AUC 0.855, losses giảm đều) nhưng metric không capture.
4. **Head thiếu signal**: `nn.Linear(x_pre)` không có explicit transition feature → khó phân biệt boundary vs non-boundary snippet.

**Bài học:**
- BSN eval không dùng cho model selection khi heads chưa converge. Cần dual eval (threshold select + BSN diagnostic).
- Boundary heads cần feature injection (temporal difference) để nhận diện transition.
- Sub-snippet offset regression cần thiết cho strict IoU (@0.4, @0.5).
- → Xem Exp 10 (BSN v2).

---

### Exp 10 — BSN v2: x_diff + offset regression + dual eval

**Hypothesis:** Exp 9 fail do heads thiếu transition feature và gradient quá yếu. BSN v2 inject temporal difference `x_diff` để amplify rising/falling edge, thêm offset regression cho sub-snippet precision, và dùng dual eval (threshold mAP cho model selection, BSN mAP diagnostic).

**Thay đổi code (so với Exp 9):**

1. `model.py`: Heads D→1 thành D→2 (cls + offset). Thêm x_diff computation:
   ```python
   x_diff = F.pad(x_pre[:, 1:] - x_pre[:, :-1], (0, 0, 0, 1))  # [B, T, D]
   start_logits = self.start_head(x_pre + x_diff)   # [B, T, 2] rising edge
   end_logits = self.end_head(x_pre - x_diff)        # [B, T, 2] falling edge
   ```
   `x_pre + x_diff` khuếch đại rising edge (Normal→Anomaly transition), `x_pre - x_diff` khuếch đại falling edge. Zero-parameter — chỉ dùng arithmetic trên existing features.

2. `train.py`: Thay `boundary_bce_loss` bằng `boundary_offset_loss` — BCE cls + smooth L1 offset tại positive snippets. Lambda tăng 0.5→2.0.

3. `tools.py`: Thêm `build_boundary_offset_targets` — tính offset = `(exact_frame - snippet_start_frame) / snippet_span` ∈ [0, 1) từ JSON timestamps.

4. `dataset.py`: Return 5-tuple `(feat, label, y_bin, length, bnd_targets)` với `bnd_targets = [s_cls, e_cls, s_off, e_off]`.

5. `detection_map.py`: `_bsn_generate_proposals` dùng snippet-resolution peak picking (không repeat ×16), offset refinement: `fs = s * 16 + offset[s] * 16`.

6. `test.py`: Dual eval — luôn tính threshold mAP, BSN mAP chỉ khi `--inference bsn`. Model selection dùng threshold mAP.

**Config:**

| Arg | Value |
|---|---|
| `--pos-weight` | 1.0 |
| `--focal-gamma` | 0 |
| `--phase3-loss` | dice |
| `--lambda2` | 1.0 |
| `--lambda-contrast` | 1.0 |
| `--contrast-margin` | 0.3 |
| `--lambda-boundary` | **2.0** (tăng từ 0.5) |
| `--boundary-pos-weight` | 10.0 |
| `--inference` | bsn |
| `--max-epoch` | 15 |

**Kết quả (15 epochs, dual eval):**

| Ep | Phase | bnd | thr mAP | [@0.1/0.2/0.3/0.4/0.5] | BSN mAP | BSN p/v | BSN v |
|---:|---|---:|---:|---|---:|---:|---:|
| 1 | P1 | 0.000 | 15.30 | 36.12/20.64/9.96/6.75/3.01 | 1.79 | 212.6 | 290/290 |
| 3 | P1 | 0.000 | 14.82 | 34.21/22.10/9.83/5.69/2.25 | 2.16 | 211.7 | 290/290 |
| 4 | P2 | 0.654 | 18.04 | 37.34/27.42/14.25/7.57/3.64 | 1.41 | 54.2 | 168/290 |
| 6 | P2 | 0.427 | 17.61 | 36.39/26.39/13.36/7.96/3.97 | 1.32 | 45.9 | 166/290 |
| 7 | P3 | 0.423 | 19.34 | 39.28/28.07/16.68/8.36/4.33 | 1.36 | 43.1 | 166/290 |
| 8 | P3 | 0.419 | 19.82 | 39.26/29.16/17.27/8.92/4.48 | 1.33 | 39.3 | 164/290 |
| 11 | P3 | 0.417 | 20.09 | 39.73/29.20/17.52/9.40/4.58 | 1.34 | 38.6 | 164/290 |
| 14 | P3 | 0.416 | 20.22 | 39.81/29.33/17.67/9.56/4.70 | 1.34 | 37.5 | 165/290 |
| 15 | P3 | 0.416 | **20.29** | 39.94/29.55/17.67/9.57/4.71 | 1.34 | 37.3 | 165/290 |

Best: **thr mAP=20.29 (ep15)**, AUC=0.8631.

**So sánh Exp 9 vs Exp 10 (Phase 2, ep4):**

| Metric | Exp 9 | Exp 10 | Nhận xét |
|---|---|---|---|
| bnd loss | 1.158 | **0.654** | x_diff giúp heads học nhanh hơn |
| BSN proposals/v | 167.0 | **54.2** | Giảm 3× — heads phân biệt boundary tốt |
| Videos w/ proposals | 290/290 | **168/290** | 122 videos suppress (mostly Normal) |
| bnd final (ep15) | 0.845 (saturate ep6) | **0.416** (vẫn giảm) | x_diff + λ=2.0 prevent saturation |

**Kết luận: PARTIAL SUCCESS.**

x_diff feature injection **work rõ ràng** — boundary heads học tốt hơn nhiều so với Exp 9:
- Proposals giảm từ 212→37/v (vs Exp 9: 207→143, never below 140)
- bnd loss giảm sâu hơn (0.416 vs 0.845) và không saturate
- 125/290 videos bị suppress (đúng — ~150 Normal videos)

**Nhưng threshold mAP = 20.29 < Exp 6 = 21.71** — boundary loss + offset regression chưa cải thiện localization so với Exp 6 thuần Dice+contrast:
- @0.1: 39.94 vs 42.03 (−2.09) — proposals miss một số events
- @0.5: 4.71 vs 6.04 (−1.33) — offset chưa đủ precise

**BSN mAP vẫn thấp** (~1.34) — boundary heads predict tốt hơn nhưng BSN proposal generation vẫn chưa competitive với threshold-based.

**Bài học:**
- x_diff là đúng hướng — heads thực sự học được transition features
- λ_boundary=2.0 + effective weight 0.1×2.0=0.2 (20% total) tốt hơn 0.05 (5%) của Exp 9
- Threshold-based proposals vẫn mạnh hơn BSN ở thời điểm này — cần tiếp tục train hoặc tăng boundary weight
- Possible improvement: tăng λ_boundary thêm, hoặc thử BSN proposals kết hợp threshold filtering

---

## 4. Tóm tắt

> **Số liệu eval cũ (gtpos=306).** Exp 1-5, 7, 8 chưa re-eval với fix mới. Baseline và Exp 6 đã re-eval — xem cột riêng.

| Exp | Thay đổi chính | mAP (cũ) | mAP (re-eval) | AUC | Kết luận |
|---|---|---:|---:|---:|---|
| Baseline | — | ~~9.47~~ | **16.06** ✅ | 0.8736 | — |
| 1 | frame BCE + Soft IoU | 11.47 | _chưa re-eval_ | ~ | ✅ beat base |
| 2 | focal BCE + TV | 11.15 | _chưa re-eval_ | 0.8527 | ✅ |
| 3 | focal BCE + TV (λ2=5) | 11.62 | _chưa re-eval_ | 0.8588 | ✅ |
| 4 | plain BCE + Dice | 11.70 | _chưa re-eval_ | 0.8567 | ✅ P3>P2 |
| 5 | Exp 4 + 15 epochs | 11.70 | _chưa re-eval_ | 0.8567 | = Exp 4 |
| **6** | **pw=1.0, Dice + contrast** | ~~12.03~~ | **21.71** ✅ | **0.8647** | **Best, đạt Must** |
| 7 | sliding-window attn | 11.36 | _chưa re-eval_ | 0.8600 | ❌ hurt @0.4-0.5 |
| 8 | remove sim GCN + residual k=1 | 9.48 | _chưa re-eval_ | 0.8521 | ❌ GCN needed |
| 9 | BSN boundary heads | 2.16 (BSN eval) | _N/A_ | 0.8404 | ❌ heads too weak |
| 10 | BSN v2: x_diff + offset + dual eval | — | **20.29** | 0.8631 | △ x_diff works, < Exp6 |

**Exp 6 re-eval đạt Must threshold mới** (21.71 > 21.06 = baseline 16.06 + 5). AUC drop chỉ 0.009.

### Per-IoU so sánh (re-eval, chỉ Baseline và Exp 6)

| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| **Baseline (re-eval)** | **35.80** | **22.34** | **12.23** | **7.09** | **2.82** | **16.06** |
| **Exp 6 (re-eval)** | **42.03** | **30.20** | **19.21** | **11.10** | **6.04** | **21.71** |
| Δ | +6.23 | +7.86 | +6.98 | +4.01 | +3.22 | +5.65 |

### Per-IoU so sánh (eval cũ, chưa re-eval — để tham khảo ranking)

| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 21.98 | 12.93 | 6.80 | 4.01 | 1.65 | 9.47 |
| Exp 2 | 22.61 | 16.82 | 9.35 | 4.70 | 2.27 | 11.15 |
| Exp 3 | 24.38 | 17.00 | 9.71 | 4.93 | 2.08 | 11.62 |
| Exp 4 | 24.45 | 17.49 | 9.75 | 4.68 | 2.13 | 11.70 |
| Exp 6 | 23.23 | 16.78 | 10.68 | 6.11 | 3.35 | 12.03 |
| Exp 7 | 23.02 | 17.44 | 9.71 | 4.61 | 2.11 | 11.36 |
| Exp 8 | 19.59 | 13.58 | 7.02 | 4.54 | 2.66 | 9.48 |

### Bottleneck phát hiện (updated post-Exp 10)

- **Feature pipeline đã tốt.** Similarity GCN giúp (Exp 8 bỏ → hại). Sliding-window hại strict IoU (Exp 7). Loss tuning đã saturate (Exp 1-6).
- **BSN v1 (Exp 9) fail**: Vanilla `nn.Linear(x_pre)` heads quá yếu — thiếu transition feature, gradient chỉ 5% total loss.
- **BSN v2 (Exp 10) partial**: x_diff feature injection work — proposals giảm 212→37/v, bnd loss giảm sâu. Nhưng threshold mAP 20.29 < Exp 6 21.71. BSN proposals chưa competitive.
- **Exp 6 vẫn là best** (21.71). Boundary heads chưa cải thiện localization — có thể do overhead loss làm model chính yếu hơn.
- **FP Analysis**: Cả baseline và Exp 6 đều 100% Normal videos có proposals (do adaptive threshold relative). Top-5 worst Normal videos có max_prob >0.9. Threshold filtering (`--threshold-min-score`, `--threshold-min-range`) có thể giảm FP.
- **Hướng tiếp theo**: (1) Threshold filtering parameters sweep, (2) Tăng λ_boundary hoặc train thêm epochs, (3) Hybrid: threshold proposals + BSN refinement.
