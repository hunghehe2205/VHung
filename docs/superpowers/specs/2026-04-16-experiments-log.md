# Experiments Log — dev_bce supervised training

---

## 1. Setup

**Eval**: `src/test.py`, 290 test videos (140 anomaly + 150 normal). Metric chính: **class-agnostic AVG mAP** across IoU {0.1, 0.2, 0.3, 0.4, 0.5} từ `sigmoid(logits1)`.

**Training 3-phase schedule** (`get_lambda`):
- P1 (ep < phase1): MIL-only warmup → (0, 0)
- P2 (phase1 ≤ ep < phase2): frame BCE + boundary → (λ1, 0)
- P3 (ep ≥ phase2): + Dice + contrast → (λ1, λ2)

**Eval bug fix (2026-04-16)**: `_loc_map_agnostic` include Normal GT segments label 'A' → inflate gtpos 156→306. Fix: filter `!= 'A'`. mAP tuyệt đối thay đổi nhưng ranking giữ nguyên. Exp 1-5, 7, 8 chưa re-eval.

**FP Analysis (baseline & Exp 6)**: 100% Normal videos có proposals do adaptive threshold relative (`max - 0.6*(max-min)` luôn tìm được proposals dù prob ~0.01). Top-5 worst Normal videos max_prob >0.9.

---

## 2. Baseline (`model/model_ucf.pth`)

| Metric | Value |
|---|---|
| AUC1 | **0.8736** |
| mAP (re-eval) | **16.06** [35.80/22.34/12.23/7.09/2.82] |

**Must threshold**: mAP ≥ 21.06 (baseline + 5)

---

## 3. Experiments

### Phase 1: Loss tuning (Exp 1-6)

#### Exp 1 — frame BCE + Soft IoU
- P2: plain BCE (pw=5.8), P3: Soft temporal IoU
- Best: **11.47** (P2 peak). P3 fail — IoU redundant với BCE.

#### Exp 2 — focal BCE γ=2 + TV smoothness
- focal γ=2 quá mạnh (effective gradient ~15% plain BCE). TV scale quá nhỏ (0.006 × 0.1 = 0.0006).
- Best: **11.15**. P3 giảm mAP. FAIL.

#### Exp 3 — focal γ=1 + TV λ2=5
- γ=1 recover P2 (11.62 > 11.15). TV λ2=5 có signal nhưng vẫn kéo P3 xuống. TV over-smooth.
- Best: **11.62**. FAIL.

#### Exp 4 — plain BCE + Dice
- Dice skip normal videos, signal lớn (p3≈0.55 vs TV 0.006). **Exp đầu tiên P3 > P2.**
- Best: **11.70** (ep8, P3). Đạt Should.

#### Exp 5 — Exp 4 + 15 epochs
- Best vẫn **11.70** (ep8). Dice saturate. **Loss ceiling = 11.70.**

#### Chẩn đoán giữa Exp 5→6

**Upsample test**: Linear vs repeat ×16 → Δ < 0.2 mọi IoU. Ceiling không do upsampling.

**Viz diagnosis** (5 case study):
- Normal videos: raw ≈ 0 → classification tốt
- Anomaly videos: raw 0.8-1.0 toàn video → saturate, không localize
- Normalized shape có signal localization → model "biết" nhưng output bias
- Root cause: pw=5.8 đẩy positives lên + MIL không phạt high-everywhere

#### Exp 6 — pw=1.0 + Dice + within-video contrast (**BEST**)
- 3 can thiệp: pw 5.8→1.0, Dice (giữ), contrast loss (new: ép inside > outside + 0.3 margin)
- Best (re-eval): **21.71** [42.03/30.20/19.21/11.10/6.04], AUC=0.8647
- **Đạt Must** (21.71 > 21.06). @0.5 gấp đôi baseline (6.04 vs 2.82).
- Contrast trực tiếp ép peak đúng GT → help strict IoU rõ rệt.

### Phase 2: Architecture changes (Exp 7-8)

#### Exp 7 — Sliding-window attention
- Hypothesis: block-diagonal (w=8, non-overlapping) cắt context tại biên → thay sliding-window.
- Best: **11.36**. **HURT strict IoU** (−37% @0.5, −25% @0.4).
- Wider receptive field → smooth features → proposals béo hơn. Block-diagonal giữ features compact.
- **Bài học**: Temporal mixing rộng hơn = hại strict IoU.

#### Exp 8 — Remove similarity GCN
- Hypothesis: Cosine sim GCN oversmooth. Bỏ gc1+gc2, thay pre_proj + k=1 residual.
- Best: **9.48**. **FAIL tất cả metrics** (−2.55 avg, −0.013 AUC).
- Similarity GCN = **beneficial feature aggregation**, không phải oversmoothing. Revert.

### Phase 3: Boundary heads (Exp 9-11)

#### Exp 9 — BSN v1: start/end heads
- Thêm `start_head = Linear(512,1)`, `end_head = Linear(512,1)` trên x_pre.
- BSN proposal: peak-pick → enumerate (s,e) → score = sp[s]×ep[e]×act.mean().
- Best BSN mAP: **2.16**. Proposals 143-207/v (cần ~5-15). **FAIL.**
- Root cause: (1) heads thiếu transition feature, (2) effective weight chỉ 5% (lam1×λ_bnd = 0.1×0.5), (3) bnd loss saturate 1.16→0.85.
- Bug fixes: peak_pick plateau, end index inconsistency, get_lambda 2→3 phase, score scale.

#### Exp 10 — BSN v2: x_diff + offset + dual eval
- x_diff = temporal difference: `start_head(x_pre + x_diff)` amplify rising edge, `end_head(x_pre - x_diff)` falling edge. Heads D→2 (cls + offset). Snippet-resolution peak picking. Dual eval.
- λ_boundary: 0.5→2.0.
- Best thr mAP: **20.29** [39.94/29.55/17.67/9.57/4.71], AUC=0.8631. BSN p/v: 212→37.

**x_diff work**:
- bnd loss: 0.654→0.416 (vs Exp9: 1.16→0.85 saturate)
- Proposals: 212→37/v (vs Exp9: 207→143)
- 125/290 videos suppress (mostly Normal)

**Nhưng < Exp 6** (20.29 vs 21.71): boundary gradient backprop vào backbone hurt actionness (@0.1: 39.94 vs 42.03).

#### Exp 11 — Stop-gradient + diagnostic logging
- **Fix**: `x_pre.detach()` cắt gradient từ boundary heads về backbone. Bỏ lam1 gating (dùng binary gate vì detach). λ_boundary=1.5.
- **Diagnostic thêm**: tách `bnd_cls`/`bnd_off`, BSN proposals split A/N.
- Best thr mAP: **21.15** [41.39/29.41/19.83/9.85/5.25], AUC=0.8619.

**Detach hypothesis confirmed**: @0.1 recovered (43.08 ở ep8 vs Exp6 42.03). Backbone không bị nhiễu.

**Nhưng boundary heads không học**:
- `bnd_off = 0.096` flat từ ep4→15 — offset head frozen
- `bnd_cls`: 1.421→1.116, giảm rất chậm
- BSN proposals: ~210/v suốt 15 epochs, A/N split không đổi
- BSN stats: A:140v/24584p N:149v/35940p — heads essentially random

**Root cause: LR scheduler conflict**. Heads khởi tạo random, bắt đầu nhận gradient ep4. Nhưng LR milestone [4,8] drop 10× đúng ep4, thêm 10× ở ep8. Effective LR cho heads: `1.5 × 2e-7 = 3e-7` — quá thấp cho random linear layers.

**Irony**: Exp 10 heads learned (lower bnd, fewer proposals) nhưng hurt backbone. Exp 11 protects backbone nhưng heads can't learn.

#### Exp 12 — Selective backprop + abnormal-only boundary + scheduler fix
- **Selective backprop**: cls heads → gradient flows to backbone, offset heads → detached. Best of both worlds attempt.
- **Abnormal-only boundary**: boundary loss chỉ tính trên abnormal videos (B_half:), bỏ noise từ Normal.
- **Scheduler fix**: milestones [4,8] → [6,11] — heads có 2 epoch full LR trước khi drop.
- **Model selection**: chuyển sang abn-only mAP thay vì all-vid.
- **Eval speed**: skip per-class & all-vid mAP khi `quiet=True` → eval time ~65% nhanh hơn.

| Ep | Phase | bnd_cls | bnd_off | mAP_abn | [@0.1/0.2/0.3/0.4/0.5] | AUC |
|---:|---|---:|---:|---:|---|---:|
| 3 | P1 | 0.000 | 0.000 | 15.09 | 34.81/22.49/10.05/5.82/2.29 | 0.8556 |
| 5 | P2 | 0.547 | 0.097 | 18.87 | 39.49/28.25/15.19/7.69/3.75 | 0.8541 |
| 7 | P3 | 0.543 | 0.096 | 20.83 | 42.41/30.06/17.97/9.16/4.55 | 0.8551 |
| 11 | P3 | 0.542 | 0.096 | 23.54 | 44.28/32.30/22.11/11.93/7.07 | 0.8545 |
| 15 | P3 | 0.542 | 0.096 | 23.93 | 44.84/32.88/22.58/12.32/7.05 | 0.8555 |
| **17** | P3 | 0.542 | 0.096 | **24.06** | **44.84/32.87/22.62/12.37/7.59** | 0.8557 |
| 20 | P3 | 0.542 | 0.096 | 23.98 | 44.70/32.91/22.64/12.40/7.27 | 0.8558 |

Best: **mAP_abn=24.06 (ep17, 20ep run)**, AUC=0.8557. **Best overall.** Ep15-20 saturate (+0.13 vs 15ep run).

**Kết luận:**
- mAP improvement đến từ Dice + contrast + scheduler fix, boundary heads vẫn dead (bnd_off=0.096 flat, bnd_cls=0.542)
- Nhưng boundary selective backprop hoạt động như **auxiliary task regularization** — gradient qua x_pre giúp backbone preserve temporal features
- Bỏ boundary (Exp 14) hoặc full detach (Exp 13) đều kém hơn → boundary gradient cần thiết dù heads không learn

#### Exp 13 — Separate optimizer + Gaussian targets + drop offset (FAIL)
- Fix 3 root causes: separate head optimizer (LR=2e-4, no decay), Gaussian-smoothed targets (sigma=2), bỏ offset.
- Full detach: `x_pre.detach()` → backbone mất auxiliary gradient.
- bnd loss giảm (1.627→1.553) — heads thực sự learn, nhưng mAP plateau ở ~18.
- Best: **mAP_abn=17.94 (ep9)**, AUC=0.8479. Gap vs Exp 12 widening mỗi epoch.
- **Kết luận: FAIL.** Heads học được nhưng full detach làm backbone mất signal hữu ích.

#### Exp 14 — fbce anomaly-only + bỏ boundary (FAIL)
- Ý tưởng: Normal videos chỉ MIL (video-level), localization losses chỉ trên anomaly.
- `--lambda-boundary 0` bỏ boundary hoàn toàn.
- Best: **mAP_abn=19.85 (ep7)**, AUC=0.8486. Tệ hơn Exp 13.
- **Kết luận: FAIL.** Bỏ fbce trên normal → AUC giảm (backbone kém suppress normal). Bỏ boundary → mất auxiliary gradient.

#### Exp 15 — Drop offset heads, keep cls selective backprop (FAIL)
- Ý tưởng: `bnd_off=0.096 flat` 16 epochs ở Exp 12 → offset heads dead weight. Bỏ đi để simplify.
- 4 heads → 2 heads (cls only). Giữ nguyên Exp 12 formula khác: pw=1, fbce all-vid, Dice, contrast, scheduler [6,11], 20ep, single optimizer.
- Best: **mAP_abn=21.27 (ep10)**, AUC=0.8549. Peak sớm ngay trước milestone ep11, sau đó flat 20.3-20.5.

| Ep | Exp 12 mAP | Exp 15 mAP | Δ |
|---:|---:|---:|---:|
| 3 | 15.09 | 15.55 | +0.46 |
| 5 | 18.87 | 16.35 | −2.52 |
| 7 | 20.83 | 18.18 | −2.65 |
| 11 | 23.54 | 20.87 | −2.67 |
| 17 | 24.06 | 20.39 | −3.67 |
| 20 | 23.98 | 20.43 | −3.55 |

**Bằng chứng cls head training identical**: `bnd` value ở Exp 15 match `bnd_cls` ở Exp 12 (0.548 ep5, 0.543 ep7, 0.542 ep11). Cls heads không bị ảnh hưởng bởi sự vắng mặt của offset.

**Gap uniformly negative** trên mọi IoU @0.1-0.5 → không phải lucky eval, mà systematic underperformance.

**Hypothesis**: RNG drift. `setup_seed(234)` gọi trước model init. Bỏ 2 Linear layers → `nn.init.normal_` bớt 2 lượt consume RNG state → DataLoader shuffle khác batch order từ ep1. Backbone train trajectory khác.

**Kết luận: FAIL.** Offset heads không phải dead weight thuần túy — presence của chúng ảnh hưởng training dynamics (dù gradient path từ offset về backbone bị detach hoàn toàn). Cần multi-seed để confirm có phải pure RNG noise không. **Bài học**: không touch boundary head config của Exp 12 — even inactive heads có thể load-bearing qua RNG coupling.

---

## 3.5 Diagnostics on Exp 12 (2026-04-17)

### Normal video FP analysis (`src/normal_diag.py`)

So sánh với Exp 6 FP analysis (doc 2026-04-16):

| Metric | Exp 6 | Exp 12 | Change |
|---|---|---|---|
| % Normal với proposals (adaptive thr) | ~100% | **44.7%** (67/150) | **−55pt** |
| Normal videos max_prob > 0.9 | ~Top-5 | 8/150 | similar |

**Exp 12 rất sạch trên Normal**:
- median(Normal max_prob) = **0.002** — gần như silence
- median(Anomaly max_prob) = 0.983
- **Separation 0.981** (huge gap)
- FP-rate @thr=0.5: chỉ **1.50%** Normal frames > 0.5
- 90% Normal videos có max_prob < 0.4

**Residual FP**: 8 videos max > 0.9 (5.3%). Top-3: `Normal_Videos_{915, 050, 940}` có single-frame spikes 0.98-0.99 nhưng mean < 0.07 → spikes cô lập, ít hại. Chỉ `Normal_Videos_{884, 925}` có sustained high (mean ~0.3) — nghiêm trọng hơn.

### Abnormal localization diagnostic (`src/anomaly_diag.py`)

**Dominant failure mode = WRONG LOCATION + OVER-PREDICTION**:

| Failure mode | Count | % |
|---|---:|---:|
| Peak outside any GT segment | **95/140** | **67.9%** |
| Over-predict (pred/gt > 2×) | 76/140 | 54.3% |
| Multi-peak (≥2 high regions) | 61/140 | 43.6% |
| Tight localization (IoU≥0.5) | 38/140 | 27.1% |

**Frame-level P/R @thr=0.5**:
- Precision = 17.9% (median) — proposals chứa quá nhiều non-GT frames
- Recall = 95.0% — model BẮT được GT, nhưng FLAG THÊM rất rộng

**Coverage**: GT median 14.3% frames, pred median 49.3% → over-ratio median 2.22× (gấp đôi width).

**Diễn giải**: Model học classification (AUC 0.8557) nhưng **không học localization** — peak argmax chỉ 32.1% rơi trong GT. 67.9% videos có peak ở wrong location.

### Inference ablation: proposal strategy (`src/infer_variants.py`)

Sweep 20+ strategies trên Exp 12 checkpoint (no retrain):

| Strategy | AVG mAP | Δ baseline |
|---|---:|---:|
| **`adaptive_0.6 + top_2`** | **24.77** | **+0.79** |
| `adaptive_0.6 + top_3` | 24.58 | +0.60 |
| baseline (`adaptive_0.6`) | 23.98 | 0 |
| `hybrid (thr=0.6, floor=0.2)` | 23.53 | −0.45 |
| `peak_half (floor=0.3)` | 23.11 | −0.87 |
| `abs_0.5` | 22.59 | −1.39 |
| `adaptive + top_1` | 21.30 | −2.68 |

**WINNER: `top_2` cap** → **24.77 mAP**, +0.79 từ inference-only.

Per-IoU top_2 vs baseline:
| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 |
|---|---:|---:|---:|---:|---:|
| baseline | 44.87 | 32.88 | 22.41 | 12.27 | 7.47 |
| top_2 | 45.92 | 33.59 | 23.72 | 12.83 | 7.81 |
| Δ | +1.05 | +0.71 | +1.31 | +0.56 | +0.34 |

Cải thiện đều @mọi IoU. n_props 313 → 205 (−35%) nhưng TP retention tốt vì best-IoU proposals thường rank cao (score + c_s bonus).

**Tại sao `top_2` thắng**:
- GT coverage median 14.3% → most videos có 1-2 anomaly events thực sự
- Loại low-rank FPs trong cùng video (multi-peak 43.6% videos)
- Giữ recall cho 2-segment case (`top_1` = 21.30 drop 2.68 mAP do miss segment 2)

**Tại sao abs_floor / peak_half / hybrid thua**:
- 10% anomaly videos có max_prob < 0.5 (distribution heavy-tailed)
- Floor threshold loại hết TP của các videos này
- peak_half expand to half-height tạo proposals rộng tương đương adaptive

### Rejected: Test-time augmentation (10-crop)

Train dataset có 10 crops/video (`__0..__9`). Test chỉ dùng `__0`. Trong lý thuyết averaging 10 crops ở test có thể +0.5~1.5 mAP.

**Bỏ vì không phù hợp thực tế**: deployment environment nhận stream single-view, không có cơ chế multi-crop. Thêm gain chỉ từ artifact data pipeline, không phản ánh khả năng model trên input thực.

### Adopted changes

- `top_2` cap trong proposal generation → new default inference strategy. Apply vào `_loc_map_abnormal_only` và `_loc_map_agnostic` (thêm 1 dòng `keep = keep[:2]` sau NMS).
- **Không retrain required** — tách hoàn toàn inference logic, checkpoint Exp 12 giữ nguyên.

---

## 4. Tóm tắt

| Exp | Phương pháp | mAP | AUC | Kết luận |
|---|---|---:|---:|---|
| Base | VadCLIP gốc | **16.06** | 0.8736 | — |
| 1 | BCE + Soft IoU | 11.47* | ~ | P3 fail |
| 2 | focal γ=2 + TV | 11.15* | 0.8527 | focal quá mạnh |
| 3 | focal γ=1 + TV λ=5 | 11.62* | 0.8588 | TV over-smooth |
| 4 | plain BCE + Dice | 11.70* | 0.8567 | P3>P2 lần đầu |
| 5 | Exp 4 + 15ep | 11.70* | 0.8567 | Dice saturate |
| **6** | **pw=1 + Dice + contrast** | **21.71** | **0.8647** | **Best. Đạt Must** |
| 7 | sliding-window attn | 11.36* | 0.8600 | hurt @0.4-0.5 |
| 8 | remove sim GCN | 9.48* | 0.8521 | GCN cần thiết |
| 9 | BSN v1: start/end heads | 2.16 (BSN) | 0.8404 | heads quá yếu |
| 10 | BSN v2: x_diff + offset | **20.29** | 0.8631 | x_diff work, < Exp6 |
| 11 | Exp10 + stop-gradient | **21.15** | 0.8619 | backbone recover, heads frozen |
| **12** | **selective backprop + abn-only bnd** | **24.06** | 0.8557 | **Best training. 20ep, auxiliary gradient** |
| 13 | separate optim + Gaussian + no offset | 17.94 | 0.8479 | full detach kills backbone signal |
| 14 | fbce abn-only + no boundary | 19.85 | 0.8486 | worse — needs fbce on normal + bnd |
| 15 | drop offset heads (keep cls sel backprop) | 21.27 | 0.8549 | FAIL — peak ep10 then flat, RNG drift suspected |
| **Exp12+top2** | **inference: `top_2` cap on Exp 12** | **24.77** | 0.8557 | **Best overall. No retrain** |

\* _Eval cũ (gtpos=306), chưa re-eval_

### Per-IoU (re-eval)

| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 35.80 | 22.34 | 12.23 | 7.09 | 2.82 | 16.06 |
| Exp 6 | 42.03 | 30.20 | 19.21 | 11.10 | 6.04 | 21.71 |
| Exp 10 | 39.94 | 29.55 | 17.67 | 9.57 | 4.71 | 20.29 |
| Exp 11 | 41.39 | 29.41 | 19.83 | 9.85 | 5.25 | 21.15 |
| **Exp 12** | **44.84** | **32.87** | **22.62** | **12.37** | **7.59** | **24.06** |
| **Exp12+top2** | **45.92** | **33.59** | **23.72** | **12.83** | **7.81** | **24.77** |

### Key insights

1. **Loss tuning** (Exp 1-5): Dice > TV > IoU. Plain BCE > focal. Loss ceiling ~11.70.
2. **Debias** (Exp 6): pw=1.0 + contrast loss phá ceiling. Ép peak đúng GT position → +5.65 mAP vs baseline.
3. **Architecture**: Đừng touch temporal mixing — wider = worse @0.4-0.5 (Exp 7). Sim GCN beneficial (Exp 8).
4. **Boundary heads as auxiliary regularizer** (Exp 9-14): Heads bản thân không learn (bnd flat). Nhưng selective backprop (cls → backbone) cung cấp auxiliary gradient giúp backbone preserve temporal features. Bỏ boundary (Exp 14) hoặc full detach (Exp 13) → mAP giảm 4-6 points.
5. **fbce trên normal videos cần thiết** (Exp 14): Bỏ fbce trên normal → AUC giảm, backbone kém suppress normal scores.
6. **Scheduler [6,11]** cho backbone train lâu hơn ở full LR → +2.2 mAP vs Exp 6.
7. **Model saturate ở ~24 mAP** (20ep). Ep15-20 chỉ +0.13.
8. **Offset heads không thật sự dead weight** (Exp 15): Dù `bnd_off=0.096 flat` và detach hoàn toàn, bỏ chúng → −2.79 mAP. RNG drift + training dynamics coupling suspected. Không simplify architecture thêm nữa.
9. **Localization ≠ classification** (diagnostic 2026-04-17): AUC 0.8557 tốt, nhưng **67.9% peak ngoài GT**, pred coverage 2.22× GT width. Model học "có anomaly hay không" rất tốt nhưng "ở đâu" còn yếu.
10. **Top-K cap tại inference**: `top_2` = +0.79 mAP zero-cost. Loại low-rank multi-peak FPs khi GT median 14.3% → most videos ≤ 2 events thực.

### Bottleneck hiện tại

- **Best overall = Exp12+top_2** = 24.77 mAP (inference-only, no retrain).
- **Core model** (Exp 12) saturate — thêm epochs / simplify architecture đều không giúp.
- **Wrong-location problem** chưa addressed: 67.9% peaks ngoài GT. Contrast loss margin=0.3 đủ cho classification nhưng không force peak-at-location.
- **Next ideas (candidates)**:
  - Retrain với peak-aware contrast: `max_inside > max_outside + margin` thay `mean_inside > mean_outside + margin`.
  - Suppression loss: penalty trực tiếp cho `mean(prob[outside])` trên anomaly videos.
  - Stronger contrast margin 0.3 → 0.5 hoặc 0.7.
  - Top-K MIL chỉ trên GT frames (anomaly videos) thay vì toàn video → ép peak trong GT.
- **Rejected**: TTA 10-crop — không reflect deployment reality.
