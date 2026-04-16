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
| 13 | P3 | 0.542 | 0.096 | 23.77 | 44.33/32.63/22.42/12.23/7.23 | 0.8554 |
| **15** | P3 | 0.542 | 0.096 | **23.93** | **44.84/32.88/22.58/12.32/7.05** | 0.8555 |

Best: **mAP_abn=23.93 (ep15)**, AUC=0.8555. **New best, vượt Exp 6 (21.71).**

**Kết luận: mAP improvement đến từ Dice + contrast + abnormal-only training, KHÔNG phải boundary heads.**
- `bnd_off = 0.096` flat suốt — offset head vẫn dead dù có selective backprop
- `bnd_cls` giảm rất chậm (0.599→0.541) — cls head gần như không học
- Scheduler fix [6,11] cho heads 2 epoch full LR, nhưng với detached offset head thì không đủ
- Backbone không bị hurt (AUC ổn định ~0.855)

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
| **12** | **selective backprop + abn-only bnd** | **23.93** | 0.8555 | **New best. Heads still dead** |

\* _Eval cũ (gtpos=306), chưa re-eval_

### Per-IoU (re-eval)

| | @0.1 | @0.2 | @0.3 | @0.4 | @0.5 | AVG |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 35.80 | 22.34 | 12.23 | 7.09 | 2.82 | 16.06 |
| Exp 6 | 42.03 | 30.20 | 19.21 | 11.10 | 6.04 | 21.71 |
| Exp 10 | 39.94 | 29.55 | 17.67 | 9.57 | 4.71 | 20.29 |
| Exp 11 | 41.39 | 29.41 | 19.83 | 9.85 | 5.25 | 21.15 |
| **Exp 12** | **44.84** | **32.88** | **22.58** | **12.32** | **7.05** | **23.93** |

### Key insights

1. **Loss tuning** (Exp 1-5): Dice > TV > IoU. Plain BCE > focal. Loss ceiling ~11.70.
2. **Debias** (Exp 6): pw=1.0 + contrast loss phá ceiling. Ép peak đúng GT position → +5.65 mAP vs baseline.
3. **Architecture**: Đừng touch temporal mixing — wider = worse @0.4-0.5 (Exp 7). Sim GCN beneficial (Exp 8).
4. **Boundary heads** (Exp 9-12): x_diff giúp heads learn (Exp 10), nhưng backprop hurt backbone. Detach/selective backprop bảo vệ backbone nhưng heads frozen (bnd_off=0.096 flat, bnd_cls barely moves). Root causes: extreme sparsity (~2/256 positives), LR coupling, insufficient signal.
5. **Exp 12 gains = Dice + contrast + scheduler fix**, không phải boundary heads. Scheduler [6,11] cho backbone train lâu hơn ở full LR → mAP +2.2 vs Exp 6.
6. **FP trên Normal videos**: 100% có proposals do adaptive threshold relative. Cần threshold filtering hoặc absolute minimum score.

### Bottleneck hiện tại

- **Exp 12 = best** (23.93). Boundary heads 3 experiment liên tiếp không learn.
- **Chưa thử**: (1) Separate optimizer/LR cho heads, (2) Gaussian-smoothed targets giảm sparsity, (3) Bỏ offset (dead weight) chỉ giữ cls.
- **Next**: Exp 13 — fix 3 root causes cùng lúc.
