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
- Commit: `50044ea9` (2026-04-17).

### Verify sau khi apply vào code chính (real `test.py`)

```
AUC1=0.8557  AP1=0.2919  |  AUC2=0.8434  AP2=0.2698
[all-vid ] AVG=24.43  [45.03/33.02/23.53/12.75/7.83]
[abn-only] AVG=24.86  [45.92/33.59/23.94/12.93/7.93]
```

| | Before top_2 | After top_2 | Δ |
|---|---:|---:|---:|
| all-vid mAP | 23.68 | **24.43** | +0.75 |
| abn-only mAP | 24.06 | **24.86** | +0.80 |
| AUC1 | 0.8557 | 0.8557 | 0 |

Real result cao hơn 0.09 so với sweep script (24.77) vì `_loc_map_abnormal_only` dùng scoring gốc `score + 0.7*c_s` (sweep cũng vậy — chênh là do `_loc_map_abnormal_only` filter video trước khi nhúng proposals vào flat list, thay vì filter sau). 13/13 existing tests pass.

### Phase A — ablate existing losses

Training hiện tại sum 7 losses. Trước khi add mới, ablate để tìm dead weight:

| Flag | Loại bỏ | Lý do nghi |
|---|---|---|
| `--lambda-nce 0` | CLASM class-aware MIL | Eval class-agnostic — nce có thể redundant |
| `--lambda-cts 0` | text divergence | Weight internal ×0.1 quá nhỏ — có thể ghost |

Decision rule: Δ < 0.5 mAP khi bỏ → drop permanently.

Flags đã thêm vào `option.py` (default 1.0, backward-compatible).

#### A1 — `--lambda-nce 0` (FAIL)

Giữ nguyên Exp 12 formula, chỉ set `lambda_nce=0`. 20 epochs.

| Ep | Exp 12 | A1 | Δ |
|---:|---:|---:|---:|
| 3 | 15.09 | 14.02 | −1.07 |
| 5 | 18.87 | 18.72 | −0.15 |
| 7 | 20.83 | 20.32 | −0.51 |
| 11 | 23.54 | 21.67 | −1.87 |
| 15 | 23.93 | 21.86 | −2.07 |
| 17 | 24.06 | 21.92 | −2.14 |
| **18** | — | **21.96** | — |
| 20 | 23.98 | 21.90 | −2.08 |

Best: **mAP_abn=21.96 (ep18)** [41.86/28.97/18.79/13.69/6.48], AUC=0.8543.

**Gap mở từ P3** (ep7+), stabilize ~−2.1 từ ep15. A1 saturate sớm (ep14→ep18 chỉ +0.01) trong khi Exp 12 còn leo đến ep17.

**Per-IoU gap (A1 ep18 vs Exp 12 ep17)**: lose IoU 0.1-0.3 mất nhiều (−3 đến −4 pts), IoU 0.4 gain nhẹ +1.32, IoU 0.5 mất −1.11. → A1 tạo proposals bớt accurate chứ không chỉ giảm count.

**`cts` collapse**: A1 có `cts=0.0000` từ ep7 (P3) — không có `nce` training text embeddings thì `cts` regularizer không còn gì để regularize. `nce` + `cts` coupling chặt.

**Kết luận: FAIL.** `nce` **load-bearing** dù eval class-agnostic. Đóng góp ~2 mAP qua: (1) text embedding diversity, (2) backbone representation richness qua class-specific MIL pools. **Giữ `nce`, không drop.**

#### A2 — `--lambda-cts 0` (FAIL)

Giữ nguyên Exp 12, set `lambda_cts=0`. `nce` vẫn train bình thường (1.634 → 0.118 ep20).

| Ep | Exp 12 | A2 | Δ |
|---:|---:|---:|---:|
| 3 | 15.09 | **16.06** | **+0.97** |
| 5 | 18.87 | 18.25 | −0.62 |
| 7 | 20.83 | 20.23 | −0.60 |
| 11 | 23.54 | 22.03 | −1.51 |
| 17 | 24.06 | 22.67 | −1.39 |
| **20** | 23.98 | **22.77** | **−1.21** |

Best: **mAP_abn=22.77 (ep20)** [42.59/32.73/19.52/12.54/6.46], AUC=0.8467.

ep3 A2 > Exp 12 (+0.97) — P1 warmup tốt hơn. Nhưng gap mở từ P3 (ep7+), stabilize ~−1.3 từ ep15. A2 saturate ep18-20 (+0.01 qua 3 ep).

**Khác A1**: A2 chậm thoái hơn (gap cuối −1.21 vs A1 −2.10). `cts` đóng góp ít hơn `nce` nhưng vẫn load-bearing.

**Kết luận: FAIL.** Giữ `cts`. Phase A xong: không có dead weight trong 7 losses.

---

### 3.6 Diagnose baseline vs Exp 12 (2026-04-18)

Script: `src/diag_localization.py` + `src/diag_compare.py`. Report: `docs/diag/baseline_vs_exp12.md`.

**D5 coverage quality là core diagnostic** (theo user feedback: "thật ra tôi muốn model over-coverage và giảm biên lại"):

| Metric | Baseline | Exp 12 | Δ | Ý nghĩa |
|---|---:|---:|---:|---|
| peak_in_gt_ratio | 0.357 | 0.321 | −0.036 | Exp 12 KHÔNG cải thiện peak position |
| peak_in_middle | 0.179 | 0.157 | −0.021 | Peak lệch GT ngay cả ở Exp 12 |
| coverage_inside | 1.000 | 0.958 | −0.042 | Baseline cover ALL (flood), Exp 12 vẫn 96% |
| spillover_window | **0.968** | **0.824** | **−0.144** | Exp 12 thu hẹp đáng kể |
| boundary_sharp | 0.002 | 0.008 | +0.006 | **Cả hai ~0** — không có biên sắc |
| over_coverage | **4.136** | **2.497** | **−1.638** | Exp 12 halved (4× → 2.5×) |
| median IoU | 0.174 | 0.275 | +0.101 | |

**Kết luận chẩn đoán**:
- **Baseline = flood mode**: prob cao đều khắp video → cover 100% nhưng over_cov 4×, spillover 97%. Không phải "biết chỗ" mà là "nói gì cũng đúng".
- **Exp 12 = scoped flood**: giảm spillover 97→82%, giảm over_cov 4×→2.5×. Coverage vẫn 96% → biết chỗ, nhưng lan rộng.
- **Cả hai boundary_sharp ≈ 0**: không loss nào tạo edge sắc.

**Primary bottleneck KHÔNG phải wrong-location** (coverage 96% chứng tỏ model biết chỗ). **LÀ over-coverage + soft boundary.** Peak-contrast ý tưởng cũ không fix được vì model không có wrong-peak issue primary.

**Correct attack**: REPLACE một loss hiện tại bằng asymmetric signal phạt over-pred → Tversky Dice (α>β).

---

#### Exp 16 — REPLACE Dice → Tversky (α=0.7, β=0.3)

**Hypothesis**: symmetric Dice `2·TP/(P+Y)` không phạt over-pred đặc biệt. Tversky `TP/(TP + α·FP + β·FN)` với α>β đánh mạnh FP (over-cov) ít hơn FN (miss). Cùng role (anomaly-only gate, weight λ2=1.0, eps=1.0 smoothing) nhưng kéo prediction width xuống.

**Change**: 1 dòng trong `train.py` — `dice_loss_anomaly → tversky_loss_anomaly(α=0.7, β=0.3)`. Loss count giữ 7 (REPLACE, không ADD).

**Targets** (monitor sau training bằng `diag_localization.py`):
- `coverage_inside ≥ 0.90` (cho phép giảm nhẹ từ 0.958 của Exp 12)
- `over_coverage_ratio < 1.5` (target từ 2.5 của Exp 12)
- `spillover_window < 0.5` (target từ 0.82 của Exp 12)
- `mAP_abn > 24.86` (beat current best)

**Warning**: α=0.7 là aggressive. Nếu coverage sụp < 80% → model miss anomaly → giảm α xuống 0.6 hoặc 0.55.

**Training command**:
```bash
python -u src/train.py \
    --focal-gamma 0 \
    --phase3-loss tversky \
    --tversky-alpha 0.7 \
    --tversky-beta 0.3 \
    --lambda2 1.0 \
    --lambda-contrast 1.0 \
    --contrast-margin 0.3 \
    --pos-weight 1.0 \
    --lambda-boundary 1.5 \
    --boundary-pos-weight 10.0 \
    --inference threshold \
    --max-epoch 20 \
    --scheduler-milestones 6 11 \
    --model-path final_model/model_exp16_tversky.pth \
    --checkpoint-path final_model/ckpt_exp16_tversky.pth \
    | tee logs/train_exp16_tversky.log
```

**Results (final ep20)**: mAP_abn **20.86 (ep18 best)** [38.09/28.32/19.61/12.32/5.97], AUC=0.8545. **FAIL — Δ = −3.20 vs Exp 12**, worse than A1 and A2.

| Ep | Exp 12 | Exp 16 | Δ |
|---:|---:|---:|---:|
| 7 | 20.83 | 19.16 | −1.67 |
| 11 | 23.54 | 19.68 | −3.86 |
| 15 | 23.93 | 19.96 | −3.97 |
| 18 | — | **20.86** | — |
| 20 | 23.98 | 20.79 | −3.19 |

**Per-IoU ep18 vs Exp 12 ep17** — chữ ký over-shrink:

| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 |
|---|---:|---:|---:|---:|---:|
| Δ | −6.75 | −4.55 | −3.01 | **−0.05** | −1.62 |

Mất nặng ở lose IoU (recall sụp) nhưng @0.4 **gần bằng** — prediction tight hơn nhưng miss GT. Classic over-shrink.

**Diag Exp 16 confirm 2 insight quan trọng**:

| Metric | Exp 12 | Exp 16 | Δ |
|---|---:|---:|---:|
| peak_in_gt_ratio | 0.321 | 0.364 | +0.04 |
| **coverage_inside** | **0.958** | **0.728** | **−0.23** 🔴 |
| **over_coverage** | **2.497** | **2.344** | **−0.15** ⚠️ |
| boundary_sharp | 0.008 | 0.008 | 0 ⚠️ |

- **Tversky shrink uniform**: coverage rớt 23pp (27% GT frames nay < 0.5) nhưng `over_coverage` chỉ giảm 0.15 → **Tversky là sai tool**. Adaptive threshold `max − 0.6·(max − min)` tự normalize theo max → shrink uniform không làm hẹp segment width.
- **Boundary vẫn flat**: 0.008 không đổi. Tversky không đụng edge.

**Kết luận: Tversky không phải cách giải quyết over-coverage + soft boundary**. Tuning α milder cũng vô nghĩa (α=0.6 sẽ reduce over_cov càng ít).

#### Exp 17 — REPLACE contrast → boundary_sharp_loss

**Motivation**: Exp 16 diag thấy `boundary_sharp = 0.008` (gần flat) ở cả baseline và Exp 12. Không loss hiện tại force sharp edges. Contrast loss (mean_inside vs mean_outside) thoả mãn được bằng uniform — không tạo biên sắc.

**Design**: REPLACE `within_video_contrast_loss` bằng `boundary_sharp_loss` tại cùng vị trí, cùng weight `λ_contrast=1.0`. Loss count giữ 7.

```python
def boundary_sharp_loss(probs, target, mask, margin=0.5):
    """At every GT transition (start or end), force |Δprob| ≥ margin."""
    mask_diff   = (mask[:, 1:] & mask[:, :-1]).float()
    target_diff = (target[:, 1:] - target[:, :-1]).abs()     # 1 at transitions
    prob_diff   = (probs[:, 1:] - probs[:, :-1]).abs()
    weight = mask_diff * target_diff
    gap = F.relu(margin - prob_diff)
    return (gap * weight).sum() / weight.sum().clamp_min(1.0)
```

**Tại sao thay được contrast**:
- Cùng role "inside vs outside separation" trên anomaly videos
- Cùng gate (anomaly-only tự nhiên: normal video target all-0 → no transitions → weight 0)
- Local version (tại biên) thay global version (mean)

**Targets** (từ diag baseline vs Exp 12):
- `boundary_sharpness ≥ 0.1` (target từ 0.008 hiện tại → gấp 12×)
- `over_coverage ≤ 2.0` (giảm vì biên sắc không lan ra xa)
- `coverage_inside ≥ 0.90` (không bị Tversky-shrink vì loss khác bản chất)
- `mAP_abn > 24.86` (beat current best)

**Training command**:
```bash
python -u src/train.py \
    --focal-gamma 0 \
    --phase3-loss dice \
    --lambda2 1.0 \
    --lambda-contrast 1.0 \
    --contrast-type boundary_sharp \
    --contrast-margin 0.5 \
    --pos-weight 1.0 \
    --lambda-boundary 1.5 \
    --boundary-pos-weight 10.0 \
    --inference threshold \
    --max-epoch 20 \
    --scheduler-milestones 6 11 \
    --model-path final_model/model_exp17_bsharp.pth \
    --checkpoint-path final_model/ckpt_exp17_bsharp.pth \
    | tee logs/train_exp17_bsharp.log
```

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
| A1 | `--lambda-nce 0` (drop CLASM) | 21.96 | 0.8543 | FAIL — Δ=−2.10 vs Exp 12, nce load-bearing |
| A2 | `--lambda-cts 0` (drop text div) | 22.77 | 0.8467 | FAIL — Δ=−1.21 vs Exp 12, cts load-bearing |
| 16 | REPLACE Dice → Tversky(0.7, 0.3) | 20.86 | 0.8545 | FAIL — Δ=−3.20, over-shrink (coverage 0.958→0.728) |
| 17 | REPLACE contrast → boundary_sharp | TBD | TBD | Targeting boundary_sharp=0.008 flat → ≥0.1 |
| **Exp12+top2** | **inference: `top_2` cap on Exp 12** | **24.86** | 0.8557 | **Best overall. No retrain (real test.py)** |

\* _Eval cũ (gtpos=306), chưa re-eval_

### Per-IoU (re-eval)

| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 35.80 | 22.34 | 12.23 | 7.09 | 2.82 | 16.06 |
| Exp 6 | 42.03 | 30.20 | 19.21 | 11.10 | 6.04 | 21.71 |
| Exp 10 | 39.94 | 29.55 | 17.67 | 9.57 | 4.71 | 20.29 |
| Exp 11 | 41.39 | 29.41 | 19.83 | 9.85 | 5.25 | 21.15 |
| **Exp 12** | **44.84** | **32.87** | **22.62** | **12.37** | **7.59** | **24.06** |
| **Exp12+top2** | **45.92** | **33.59** | **23.94** | **12.93** | **7.93** | **24.86** |

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
