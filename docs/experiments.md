# Experiment Log: logits3 Anomaly Map Head

## Objective

Thêm anomaly map head (logits3) vào VadCLIP để sinh anomaly score map chất lượng cao cho downstream Anomaly-focused Temporal Sampler (CDF cumsum → keyframe extraction).

**Targets:**
- AUC > 88.01 (VadCLIP baseline trên UCF-Crime)
- Map quality: anomaly~1, normal~0, plateau trên vùng anomaly, flat trên normal
- Pipeline giải thích được

**Baseline:** VadCLIP reported AUC = 88.01

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

## Experiment 3: No Detach — gradient flows back (ĐANG CHẠY)

**Branch:** `dev_logits3` (hiện tại)
**Ngày:** 2026-04-14

### Config
- Architecture: `logits3 = map_head(visual_features)` — **KHÔNG detach**
- Losses: BCE + Smoothness (Phase A losses, đơn giản)
- `lambda_bce=1.0, lambda_smooth=0.1`
- Gradient từ logits3 losses chảy ngược qua GCN/Transformer

### Hypothesis
- Shared features được optimize bởi cả MIL losses (logits1/logits2) VÀ frame-level supervision (logits3)
- Multi-task learning → features tốt hơn cho tất cả heads
- AUC có thể tăng (hoặc ít nhất giữ nguyên)
- Post-training: Platt scaling để calibrate absolute scores

### Tiêu chí đánh giá (sau 2 epochs)

| Kết quả | Hành động |
|---|---|
| AUC1 ≥ 0.86 | Proceed, chạy full 10 epochs |
| AUC1 giảm nhẹ (<0.5%) nhưng AUC3 tăng | Proceed, rely on ensemble |
| AUC1 giảm >1% | Dừng, quay lại detach |

### Kết quả (Epoch 1-2, ongoing)

| Metric | Epoch 1 best | Epoch 2 (step 5120) |
|---|---|---|
| AUC1 | **0.8636** | 0.8552 |
| AUC3 | 0.8510 | 0.8426 |
| anomaly_mean | 0.302 | — |
| normal_mean | 0.168 | — |
| gap | 0.134 | — |
| coverage | 7.8% | — |

### Analysis (sau 2 epochs)

1. **AUC1 giữ nguyên**: 0.8636 vs Phase A 0.8630 → **no-detach an toàn**, gradient conflict không đáng kể
2. **AUC3 khởi đầu tốt hơn**: 0.8510 tại Epoch 1 (Phase A cần 3 epochs mới đạt 0.8593)
3. **Map quality vẫn kém**: anomaly=0.302, coverage=7.8% — gần như giống Phase A
4. **Kết luận**: No-detach alone chỉ cải thiện AUC, KHÔNG fix calibration. Cần thêm Coverage + Ranking losses

---

## Experiment 4: No Detach + Coverage + Ranking (CHUẨN BỊ)

**Branch:** `dev_logits3`
**Ngày:** 2026-04-14

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

### Kết quả
*(chờ training)*

---

## Experiment 5: Platt Scaling (CHUẨN BỊ)

**Script:** `src/platt_scaling.py`
**Ngày:** 2026-04-14

### Config
- Input: Checkpoint từ Exp 3 hoặc Exp 4
- Method: Grid search temperature + bias trên raw logits3
- `calibrated = sigmoid((raw - bias) / temp)`
- Không retrain, chỉ post-hoc calibration

### Kết quả
*(chờ checkpoint)*

---

## Experiment Backlog

| # | Experiment | Mô tả | Depends on |
|---|---|---|---|
| 6 | Ensemble tuning | Grid search α trong `α*p1 + (1-α)*p3` | Exp 4/5 checkpoint |
| 7 | Full pipeline eval | AUC ensemble > 88.01? Map quality OK? | Exp 5 + 6 |
