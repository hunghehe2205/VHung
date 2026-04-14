# Experiment Log: logits3 Anomaly Map Head

## Objective

Thêm anomaly map head (logits3) vào VadCLIP để sinh anomaly score map chất lượng cao cho downstream Anomaly-focused Temporal Sampler (CDF cumsum → keyframe extraction).

**Targets:**
- AUC > 88.01 (VadCLIP baseline trên UCF-Crime)
- Map quality: anomaly~1, normal~0, plateau trên vùng anomaly, flat trên normal
- Pipeline giải thích được

**Baseline:** VadCLIP reported AUC = 88.01

---

## Hard Constraints (fixed, không bàn lại)

Xác lập ngày 2026-04-14 sau Experiment 5:

1. **Không thể thay đổi env**. CUDA/PyTorch/cuDNN numerics của máy này là cố định. Env đồng nghiệp reproduce được 0.8801 nhưng env này chỉ đạt 0.8736 trên cùng pretrained → gap ~0.65pp là env numerics, không phải method.
2. Không thể train lại baseline để cố reproduce, kết quả không khác đâu.
3. 
Mọi experiment tiếp theo phải thỏa các ràng buộc trên.

---

## Experiment 1: Phase A — Detached + BCE + Smoothness

**Branch:** `dev_logits3` (cũ, đã xóa)
**Ngày:** 2026-04-13

### Config
- Architecture: `logits3 = map_head(visual_features.detach())`
- Losses: BCE (soft labels, σ=1.5) + Conditional TV smoothness
- `lambda_bce=1.0, lambda_smooth=0.1`
- Optimizer: AdamW, lr=2e-5, scheduler decay at epoch 4, 8

### Kết quả (3 epochs)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| AUC1 (logits1) | 0.8579 | 0.8607 | 0.8627 |
| AUC3 (logits3) | 0.8437 | 0.8557 | 0.8593 |
| anomaly_mean | 0.296 | 0.328 | — |
| normal_mean | 0.171 | 0.161 | — |
| gap | 0.125 | 0.168 | — |
| coverage (>0.5) | 8.6% | 8.5% | — |

### Diagnostic (checkpoint best AUC1=0.8630)

```
logits1: normal=0.032, anomaly=0.648, gap=0.616
logits3: normal=0.067, anomaly=0.279, gap=0.212
Raw logits3 gap: 2.065 (normal=-3.13, anomaly=-1.07)
Correlation p1 vs p3: 0.926
```

### Analysis

**AUC OK, map quality FAIL.**

1. **Detach blocks calibration**: `visual_features.detach()` ngắt gradient → shared features chỉ được optimize bởi MIL losses (logits1/logits2), không nhận signal từ frame-level supervision → features không encode frame-level intensity, chỉ encode ranking
2. **Scores nằm trong band hẹp**: anomaly=0.28, normal=0.07. BCE loss không đủ mạnh để đẩy scores ra biên [0, 1] khi features không hỗ trợ
3. **Coverage cực thấp (8.5%)**: Hầu hết anomaly frames có score < 0.5. Không có plateau, chỉ có vài peak nhẹ
4. **AUC1 không bị ảnh hưởng**: Xác nhận detach hoạt động đúng — nhưng đây cũng chính là limitation

**Kết luận:** Detached features từ GCN encode ranking tốt nhưng không encode absolute anomaly intensity. Thêm loss trên detached features không thể fix calibration.

---

## Experiment 2: Phase B — Detached + BCE + Smoothness + Coverage + Ranking

**Branch:** `dev_logits3_phaseB` (đã xóa)
**Ngày:** 2026-04-13 ~ 2026-04-14

### Config
- Architecture: Giống Phase A (`detach()`)
- Losses: BCE + Smoothness + Coverage loss + Ranking loss
- `lambda_bce=1.0, lambda_smooth=0.1, lambda_coverage=1.0, lambda_ranking=1.0`
- `coverage_threshold=0.5, ranking_margin=0.3`

### Kết quả (4 epochs)

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 |
|---|---|---|---|---|
| AUC1 | 0.8579 | 0.8607 | 0.8627 | 0.8625 |
| AUC3 | 0.8589 | 0.8612 | 0.8696 | 0.8688 |
| anomaly_mean | 0.483 | 0.503 | 0.513 | — |
| normal_mean | 0.278 | 0.253 | 0.245 | — |
| gap | 0.204 | 0.250 | 0.268 | — |
| coverage | 50.5% | 56.6% | 59.4% | — |

### Loss Trends

| Loss | Epoch 1 | Epoch 3 | Trend |
|---|---|---|---|
| L_bce | 0.492 | 0.445 | Dropping |
| L_smooth | 0.016 | 0.011 | Converged |
| L_coverage | 0.069 | 0.037 | Dropping |
| L_ranking | 0.317 | 0.300 | **Barely moving** |

### Analysis

**Coverage improved significantly, but normal scores also increased.**

1. **Coverage 8.5% → 59.4%**: Coverage loss works — anomaly frames get pushed above 0.5. Nhưng vẫn chưa đạt target >70%
2. **Normal mean tăng từ 0.16 → 0.25**: Coverage/ranking loss đẩy anomaly lên nhưng kéo normal lên theo. Không có loss nào explicit ép normal xuống ngoài BCE
3. **L_ranking barely moving (0.317→0.300)**: Ranking constraint gần như không thỏa mãn. max(anomaly) - max(normal) ≈ margin=0.3 nhưng loss vẫn ~0.3
4. **AUC1 giống hệt Phase A**: Xác nhận detach isolate hoàn toàn — nhưng cũng có nghĩa thêm losses không giúp shared features
5. **Plateau sau epoch 3**: AUC3 peak 0.8696, sau đó giảm nhẹ. Model bắt đầu overfit

**Kết luận:** Thêm losses trên detached features cải thiện coverage nhưng gặp ceiling — không thể vượt qua limitation của detached features. Normal suppression là missing piece nhưng vẫn không giải quyết root cause (features không encode frame-level info).

---

## Key Insight từ Diagnostic

```
Raw logits3 (before sigmoid):
  Normal:  mean=-3.13, range=[-5.27, -0.63]
  Anomaly: mean=-1.07, range=[-3.49, -0.04]
  Gap: 2.065
```

**Model đã học phân biệt trong logit space** (gap=2.07). Vấn đề là sigmoid(x) với x âm → scores thấp cho cả hai. Platt scaling (learn temperature + bias) có thể calibrate post-hoc:

```
Simulation: temp=0.1 → normal=0.198, anomaly=0.913, gap=0.714
```

**Ensemble tiềm năng hạn chế**: Correlation p1 vs p3 = 0.926 (rất cao, redundant).

---

## Experiment 3: No Detach — gradient flows back

**Branch:** `dev_logits3`
**Ngày:** 2026-04-14
**Status:** Crashed tại Epoch 4 (OOM/IO error), checkpoint saved tại Epoch 2

### Config
- Architecture: `logits3 = map_head(visual_features)` — **KHÔNG detach**
- Losses: BCE + Smoothness (2 losses đơn giản)
- `lambda_bce=1.0, lambda_smooth=0.1`
- Gradient từ logits3 losses chảy ngược qua GCN/Transformer

### Hypothesis
- Shared features được optimize bởi cả MIL losses (logits1/logits2) VÀ frame-level supervision (logits3)
- Multi-task learning → features tốt hơn cho tất cả heads
- AUC có thể tăng (hoặc ít nhất giữ nguyên)
- Post-training: Platt scaling để calibrate absolute scores

### Kết quả (checkpoint Epoch 2)

| Metric | Value |
|---|---|
| Best AUC1 | **0.8684** |
| Best AUC3 | 0.8510 |
| anomaly_mean | 0.352 (uncalibrated) |
| normal_mean | 0.154 (uncalibrated) |
| gap | 0.198 |
| coverage | 16.1% |

### Platt Scaling Analysis

Raw logits3: mean=-1.963, std=1.081, range [-4.733, 1.185]

**Balanced MLE Platt** (`class_weight='balanced'`):
- Temperature: 0.62, Bias: 0.35
- Normal: 0.057, Anomaly: 0.206, Gap: 0.149, Coverage: 3.8%
- → Platt kéo ANOMALY xuống do raw logit gap nhỏ

### Analysis

1. **AUC1 = 0.8684**: best so far across all experiments, nhưng vẫn **dưới baseline 0.8801 (-1.17%)**
2. **Map quality kém**: Coverage chỉ 16.1%, anomaly_mean 0.35 — không đạt "anomaly~1"
3. **Platt scaling không fix được**: Raw logits gap quá nhỏ (1.2 points), Platt chỉ trade-off được anomaly xuống hoặc normal lên
4. **Kết luận**: BCE+Smooth alone không đủ mạnh để đẩy raw logits ra biên. Cần thêm Coverage + Ranking

---

## Experiment 4: No Detach + Coverage + Ranking

**Branch:** `dev_logits3`
**Ngày:** 2026-04-14
**Status:** Completed full 10 epochs

### Config
- Architecture: `logits3 = map_head(visual_features)` — **KHÔNG detach**
- Losses: BCE + Smoothness + Coverage + Ranking (all 4 losses)
- `lambda_bce=1.0, lambda_smooth=0.1, lambda_coverage=1.0, lambda_ranking=1.0`
- `coverage_threshold=0.5, ranking_margin=0.3`
- Gradient từ ALL losses chảy ngược qua GCN/Transformer

### Hypothesis
- No-detach + Coverage/Ranking = Phase B coverage gains + better features
- Phase B (detach) đạt coverage 59.4% nhưng bị ceiling do detach. No-detach có thể vượt qua
- Ranking loss hiệu quả hơn khi features có thể adapt

### Kết quả (10 epochs, best AUC1 tại Epoch 3)

| Metric | Epoch 1 | Epoch 3 (best) | Epoch 10 (final) |
|---|---|---|---|
| AUC1 | 0.8520 | **0.8653** | 0.8634 |
| AUC3 | 0.8515 | 0.8602 | 0.8581 |
| anomaly_mean | 0.492 | ~0.51 | 0.519 |
| normal_mean | 0.270 | ~0.24 | 0.235 |
| gap | 0.222 | ~0.28 | 0.283 |
| coverage | 52.4% | ~60% | **62.2%** |

### Loss Trends (Epoch 10)

| Loss | Value | Trend |
|---|---|---|
| loss1 (MIL BCE) | 0.055 | Dropping steady |
| loss2 (classification) | 0.82 | Dropping steady |
| L_bce | 0.43 | Plateau early |
| L_smooth | 0.01 | Converged |
| L_cov | 0.03 | Plateau early |
| L_rank | 0.29 | **Barely moving** (~0.32 → 0.29) |

### Platt Scaling Analysis

Raw logits3: mean=-1.522, std=1.284, range [-5.380, 1.689]

**Uncalibrated (best map quality without post-hoc):**
- Normal: 0.225, Anomaly: 0.478, Gap: 0.253, Coverage: 55.5%

**Balanced MLE Platt** (`class_weight='balanced'`):
- Temperature: 0.63, Bias: -0.69
- Normal: 0.304, Anomaly: 0.696, Gap: 0.392, Coverage: **85.4%**
- → Calibration cải thiện rõ rệt, anomaly peak gần đạt target

**Unbalanced MLE Platt** (baseline comparison):
- Temperature: 0.61, Bias: 0.90
- Normal: 0.058, Anomaly: 0.194 (too low!), Gap: 0.135, Coverage: 0.5%
- → Class imbalance (85% normal frames) kéo anomaly xuống

### Analysis

**Map quality: CẢI THIỆN RÕ RỆT**
1. Coverage 62% (raw) → 85% (balanced Platt) — plateau behavior đạt được
2. Gap 0.28 → 0.39 — phân biệt tốt hơn
3. Anomaly mean 0.48 → 0.70 (balanced Platt) — gần target ~1

**AUC: VẪN KHÔNG ĐỦ**
1. Best AUC1 = 0.8653 — **dưới baseline 0.8801 (-1.48%)**
2. AUC3 = 0.8604 < AUC1 (ranking logits1 tốt hơn logits3)
3. AUC1 plateau tại Epoch 3, sau đó dao động 0.86-0.865 — không cải thiện thêm

**Gradient balance (root cause):**
```
Total loss breakdown (Epoch 10):
  loss1 + loss2 + loss3 = 0.055 + 0.816 + 0.016 = 0.887 (original VadCLIP)
  λ_bce*L_bce + λ_smooth*L_smooth + λ_cov*L_cov + λ_rank*L_rank
    = 0.43 + 0.001 + 0.03 + 0.29 = 0.760 (map losses)

→ Map losses chiếm ~46% tổng gradient budget
→ Shared features bị kéo ~một nửa về hướng phục vụ logits3
→ Hại cho logits1 AUC
```

**Kết luận:**
- Approach này đạt được **map quality tốt** sau Platt scaling (Coverage 85%, Gap 0.39)
- Nhưng **AUC trả giá**: -1.48% so với baseline do gradient interference
- Trade-off này không chấp nhận được nếu muốn AUC ≥ 0.8801

---

## Experiment 5: Re-baselining + Inference-time tricks

**Branch:** `dev_logits3`
**Ngày:** 2026-04-14
**Status:** Completed (3 sub-experiments diagnostic, không train)

### Motivation

Trước khi đi tiếp (ban đầu định làm Two-Stage Training), cần verify 3 câu hỏi chưa được answer:

1. Ensemble `α·p1 + (1−α)·p3` trên Exp 4 có vượt 0.8801 không?
2. Original VadCLIP (`model_ucf.pth`) ở env này thực tế đạt AUC bao nhiêu + map quality ra sao?
3. Có đẩy AUC vượt 0.8801 bằng inference-time tricks không (không train)?

### Sub-exp 5.1: Ensemble evaluation on Exp 4 checkpoint

Script: `python src/ensemble_eval.py --checkpoint-path final_model/checkpoint_exp4.pth --use-platt`

Grid search α ∈ [0.0, 1.0] step 0.05 trên `α·p1 + (1−α)·p3_calibrated`:

| Variant | AUC | Gap | Coverage | Anom mean | Norm mean |
|---|---|---|---|---|---|
| p1 raw | 0.8653 | 0.548 | 85.7% | 0.830 | 0.282 |
| p3 raw | 0.8602 | 0.253 | 55.5% | 0.478 | 0.225 |
| p3 Platt balanced | 0.8602 | 0.392 | 85.4% | 0.696 | 0.304 |
| Best ensemble | 0.8653 (α=**1.00**) | — | — | — | — |

**Findings**:
- Ensemble optimum là α=1.0 → p3 **không đóng góp signal** gì so với p1.
- p1 raw (Gap 0.548, Cov 85.7%) **tốt hơn** p3 đã calibrate (Gap 0.392, Cov 85.4%).
- Logits3 head và Platt scaling đều **không cần thiết**.

### Sub-exp 5.2: Original VadCLIP eval

Script: `python src/ensemble_eval.py --checkpoint-path model/model_ucf.pth` (load raw state_dict, `strict=False` cho map_head).

Results:

| Metric | Original (`model_ucf.pth`) | Exp 4 (trained) | Delta |
|---|---|---|---|
| AUC1 | **0.8736** | 0.8653 | −0.83pp |
| Map Gap | **0.572** | 0.548 | −0.024 |
| Coverage | **89.2%** | 85.7% | −3.5pp |
| Anomaly mean | **0.872** | 0.830 | −0.042 |
| Normal mean | 0.300 | 0.282 | — |

**Critical findings**:
1. Env baseline thực tế = **0.8736**, không phải 0.8801 paper. Gap môi trường **+0.65pp**.
2. Exp 4 training đã **làm hại** mọi metric so với original pretrained: AUC thấp hơn, map quality kém hơn.
3. Original VadCLIP p1 raw **đã đạt objective map quality** (Gap 0.57, Cov 89%) — không cần logits3 head.
4. Kết hợp với 5.1 → toàn bộ approach logits3 + map losses là **dead end**.

### Sub-exp 5.3: Temporal smoothing + TTA trên original

Script mới `src/tta_smooth_eval.py`. Apply Gaussian smoothing clip-level p1 và test-time augmentation (Gaussian noise trên visual features).

Config: σ ∈ {0, 1, 2, 3, 5}, TTA K=5, noise std=0.03.

Results (`model_ucf.pth`):

| Variant | AUC1 | Δ vs baseline |
|---|---|---|
| clean, σ=0 (baseline) | 0.8736 | — |
| clean, σ=1 | 0.8736 | 0.0000 |
| clean, σ=2 | 0.8734 | −0.0002 |
| clean, σ=3 | 0.8730 | −0.0006 |
| clean, σ=5 | 0.8718 | −0.0018 |
| TTA K=5, σ=0 | 0.8737 | +0.0001 |
| TTA K=5, σ=2 | 0.8734 | −0.0002 |
| Best overall | **0.8737** | **+0.0001** |

**Findings**:
- Smoothing σ>0: slightly **hurt** (p1 đã temporally coherent, smoothing làm leak scores qua boundaries).
- TTA noise=0.03: gain +0.0001 (noise level). Model robust với perturbation nhỏ này.
- **Total gain: +0.0001pp**, cần +0.65pp → inference tricks không đủ xa.

### Kết luận Experiment 5

1. **Logits3 head approach = dead end** (xác nhận bằng 5.1, 5.2).
2. **Map quality = solved problem**. Original p1 raw đủ dùng cho downstream (Gap 0.57, Cov 89%).
3. **Vấn đề duy nhất là AUC gap 0.65pp**, và gap này là **env numerics**, không phải method.
4. **Inference tricks rẻ (TTA/smooth) đã thử và thất bại** (+0.0001pp).
5. Mọi experiment tiếp theo phải làm việc với baseline 0.8736 cố định trên env này (xem Hard Constraints).

---

## Experiment Backlog

| # | Experiment | Mô tả | Status |
|---|---|---|---|
| 5 | Stage 1: Baseline purely | Train VadCLIP không logits3 losses, verify 0.8801 | TODO |
| 6 | Stage 2: Frozen + map_head | Train map_head trên frozen backbone | Depends on Exp 5 |
| 7 | Full pipeline eval | AUC ≥ 0.8801 + map quality OK? | Depends on Exp 5+6 |
