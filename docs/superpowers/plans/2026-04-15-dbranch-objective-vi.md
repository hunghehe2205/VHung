# D-Branch — Mục Tiêu, Cách Tiếp Cận, và Động Lực Thiết Kế

> Tài liệu này giải thích **TẠI SAO** D-Branch tồn tại và được thiết kế như hiện tại. Phần kỹ thuật chi tiết (config, file, lệnh chạy) nằm trong `2026-04-15-logits3-map-objective.md`.

---

## 1. Bối cảnh & Motivation

### 1.1. Vấn đề của baseline VadCLIP

VadCLIP gốc dùng hai head:
- **Logits1 (MIL)** — top-k pooling + sigmoid + BCE ở **video-level**. Cho ra score per-frame nhưng chỉ có *tính chất ranking* — frame nào "anomaly hơn" thì score cao hơn, không cam kết về giá trị tuyệt đối.
- **Logits2 (text-alignment)** — softmax over 14 class prompts. Phục vụ semantic identification.

Cả hai đều **không được supervise ở frame level**. Hậu quả:
- Score `s_t` từ logits1 không calibrated — không có ý nghĩa "xác suất frame là anomaly".
- Đường cong `s_t` thường peaky (do top-k pooling reward các frame cực đại).
- Biên event mờ — vùng `s_t > τ` không khớp biên ground truth.

Đây là lý do VadCLIP đạt AUC cao (88.02) — vì AUC chỉ quan tâm ranking — nhưng **mAP@IoU rất thấp (6.68)** — vì localization yêu cầu biên chính xác.

### 1.2. Phase-C đã thử nhưng đi sai hướng

Phase-C trước đó cố tạo dense score `s_t` qua một head `AnomalyMapHead` (Conv1D k=7) với 4 loss term: BCE + smoothness + mass-ratio + density. Kết quả: composite "MapScore" tăng nhưng **không có term nào trực tiếp ép pointwise correctness, valley separation, hay boundary alignment**. Gradient bị phân tán giữa các mục tiêu tranh chấp.

### 1.3. D-Branch ra đời để giải quyết điều gì

D-Branch (**Dense Branch**) là một head mới với một mục đích duy nhất:

> **Sinh ra `s_t ∈ [0,1]` per-frame mà giá trị có ý nghĩa trực tiếp:**
> `s_t ≈ 1` ⇔ frame `t` ở trong event; `s_t ≈ 0` ⇔ frame `t` ngoài event.

Khi `s_t` thoả mãn yêu cầu này, downstream ứng dụng (mAP@IoU, CDF sampler, threshold-based localization) đều hưởng lợi mà không cần thêm calibration step.

---

## 2. Ba Objectives

### Objective 1 — Pointwise correctness (đúng tại từng frame)

**Phát biểu**: `s_t → 1` khi `t ∈ event`, `s_t → 0` khi `t ∉ event`. Đây là **binary classification ở frame level**.

**Tại sao là điều kiện cần**: tất cả metric VAD (AUC, Ano-AUC, mAP) đều giả định một score "có nghĩa" theo từng frame. Nếu pointwise sai, mọi metric sau đều không đáng tin.

### Objective 2 — Cluster separation (thung lũng giữa events)

**Phát biểu**: với mỗi video có *k* events,
`min_{t ∈ event} s_t − max_{t ∉ event} s_t ≥ m` với margin `m > 0`.

Hệ quả: đường cong `s_t` có *k* peaks tách biệt, không phải một "đồi" trải dài bao trùm cả vùng normal giữa các events.

**Tại sao quan trọng**:
- Nếu hai events sát nhau và `s_t` ở giữa cao gần bằng đỉnh event → segment extraction sẽ merge thành một đoạn duy nhất → mAP@IoU giảm vì biên sai.
- Cho downstream (CDF sampler): valley giữa event tạo plateau trên cumsum → sampler không lấy frames ở vùng normal nằm giữa.

### Objective 3 — Boundary alignment + plateau

**Phát biểu**: trong mỗi event,
- (a) **Biên chính xác**: vùng `{t : s_t > τ}` có biên đầu/cuối khớp biên event GT.
- (b) **Plateau**: `s_t` trong event gần phẳng (variance thấp), không peaky.

**Tại sao quan trọng**:
- Boundary trực tiếp quyết định IoU giữa segment dự đoán và segment GT → quyết định mAP@IoU.
- Plateau đảm bảo: nếu chỉ có 2-3 frame "dễ" (đầu/cuối) cao, thì threshold τ sẽ chọn miss phần giữa event hoặc tạo nhiều segment vụn.

### Mối quan hệ giữa 3 objectives

| Objective | Là điều kiện | Ảnh hưởng metric chính |
|-----------|--------------|------------------------|
| 1 — Pointwise | **Cần** (foundation) | AUC, Ano-AUC |
| 2 — Separation | Mạnh hơn 1 | mAP@IoU (tránh merge sai) |
| 3 — Boundary + plateau | Mạnh hơn 1 và 2 | mAP@IoU (biên chính xác) |

Obj 1 đảm bảo "tốt trung bình", Obj 2 + 3 đảm bảo "tốt theo cấu trúc" — đây mới là cái mAP cần.

---

## 3. Cách Tiếp Cận Từng Objective

Mỗi objective được map vào **chính xác một loss term** để gradient không lẫn lộn và dễ debug.

### 3.1. Objective 1 → `L_bce`

```
L_bce = BCE(s_t, y_t)  với pos_weight = N_neg / N_pos
```

- `y_t ∈ {0,1}` là hard binary label, derived từ HIVAU events qua `events_to_clip_mask`.
- `pos_weight` cân bằng tự động vì anomaly frames thường chiếm ~10-30% (event_time_ratio). Không cần tune.
- BCE là loss "đúng-kiểu" cho binary frame classification.

### 3.2. Objective 2 → `L_margin`

```
L_margin = max(0,  m − (soft_min_{t ∈ event} s_t  −  soft_max_{t ∉ event} s_t))
```

- `m = 0.3` — margin ép tối thiểu 0.3 distance giữa "đáy event" và "đỉnh normal".
- Soft-min/max bằng `±logsumexp(... · T) / T` với `T = 5`. Differentiable, gradient flow đều.
- Skip cho video normal (không có event frames).
- Per-video formulation phù hợp vì 80% video UCF-Crime có ≤ 2 events — global min/max đủ cover.

**Tại sao margin loss thay vì TV smoothness?** TV smoothness chỉ ép `s_t` thay đổi mượt giữa các frames — không có cam kết về *độ sâu* của valley. Margin loss thẳng tay ép gap.

### 3.3. Objective 3 → `L_dice` + `L_var`

**Boundary alignment**:
```
L_dice = 1 − 2·Σ(s_t · y_t) / (Σ s_t + Σ y_t + ε)
```

- Soft Dice là relaxation của IoU — direct optimize cho overlap region.
- Khác BCE: BCE quan tâm tới từng frame độc lập; Dice quan tâm tới *toàn bộ overlap*. Hai loss bổ sung nhau.
- Skip video normal (Dice undefined khi `y` toàn 0).

**Plateau**:
```
L_var = mean_k Var_{t ∈ event_k}(s_t)
```

- Per-event variance: ép mỗi event riêng phẳng, không peaky.
- Implementation pure-tensor (`torch.diff` để tìm event spans) — GPU/AMP safe.
- Skip video normal.

### 3.4. Tổng hợp

```
L_D = 1.0·L_bce + 0.5·L_margin + 0.5·L_dice + 0.1·L_var
L_total = L_C/A-branch + β · L_D
```

Trọng số (1.0 / 0.5 / 0.5 / 0.1) chọn theo nguyên tắc:
- BCE = nền tảng (Obj 1) → weight cao nhất.
- Margin + Dice = ép cấu trúc (Obj 2 + boundary của Obj 3) → weight trung bình.
- Variance = "polishing" plateau → weight thấp, không lấn át BCE.

---

## 4. Motivation Thiết Kế D-Branch

### 4.1. Tại sao là Conv1D, không phải Transformer hay MLP?

| Lựa chọn | Vấn đề | Quyết định |
|----------|--------|-----------|
| **MLP per-frame** | Không có temporal mixing → không học được plateau, biên. Bias sai cho VAD. | Loại |
| **Transformer decoder** | Backbone đã có Transformer rồi. Thêm nữa = redundant. Khó interpret. | Loại |
| **Bi-LSTM** | Học tốt plateau, nhưng tuần tự → chậm, khó parallel. | Loại |
| **Conv1D stack** | Local temporal mixing, induction bias đúng cho "anomaly là chuỗi liên tục", tham số ít, dễ interpret. | **Chọn** |

**Lý do quyết định**: anomaly có *temporal locality* — hành vi bất thường kéo dài vài frames, không nhảy bậc. Conv1D với kernel nhỏ match đúng tính chất này. MLP không có induction bias này → buộc loss phải gánh toàn bộ việc shape → kém hiệu quả.

### 4.2. Tại sao kernel=3 (không 5, không 7)?

Receptive field hiệu quả của 2 layer Conv1D k=3 ≈ 5 clip ≈ 80 raw frames ≈ **2-3 giây**.

| Kernel | RF (clip) | RF (giây) | Trade-off |
|--------|-----------|-----------|-----------|
| k=3 | 5 | 2-3s | Đủ cho local pattern, biên sắc. **Chọn**. |
| k=5 | 9 | 4-5s | Bắt event dài hơn, nhưng biên bị smooth → hại mAP@IoU. |
| k=7 | 13 | 6-7s | Quá lớn, biên mờ. |

Backbone (Transformer + GCN) đã cung cấp global context. D-Branch chỉ cần refine local — kernel nhỏ đúng vai trò.

### 4.3. Tại sao progressive channels 512 → 256 → 128 → 1?

- Giảm dần tạo *feature pyramid*: layer sớm giữ thông tin rich, layer muộn summarize.
- Tránh "bottleneck to 1 ngay" → không mất thông tin sớm.
- Tổng params ~492K, dưới <500K target — nhẹ, train nhanh, không overfit.

### 4.4. Tại sao LayerNorm (không BatchNorm)?

- Video trong batch có length khác nhau → batch statistics của BatchNorm không ổn định.
- LayerNorm consistent với Transformer của backbone (đã dùng LN khắp nơi).
- VadCLIP gốc cũng dùng LN.

### 4.5. Tại sao GELU + Dropout 0.2 + Sigmoid?

- **GELU**: smooth hơn ReLU, gradient flow tốt cho dense regression. Backbone đã dùng GELU.
- **Dropout 0.2**: regularize vừa phải, không quá mạnh để dập tín hiệu.
- **Sigmoid**: binary per-frame output trong [0,1], match BCE/Dice loss.

### 4.6. Tại sao đặt trên features POST-GCN?

- Features post-GCN đã có temporal context global từ Transformer + dual-GCN (similarity adj + distance adj).
- D-Branch chỉ cần học mapping `feature → score` với local refinement.
- Reuse VadCLIP backbone → fair comparison, train rẻ.

---

## 5. Schedule C — Motivation Cho Training Strategy

### 5.1. Vấn đề khi train naive (không warmup)

Nếu train D-Branch + backbone đồng thời từ epoch 0:
1. D-Branch random init → output `s_t` random → loss D-Branch lớn → gradient lớn.
2. Gradient này lan ngược vào backbone qua shared features.
3. Backbone (đã train kỹ trên VadCLIP) bị "đẩy" sai hướng → AUC drop nhanh.

### 5.2. Schedule C giải quyết bằng 3 giai đoạn

| Phase | Epoch | β | Mục đích |
|-------|-------|---|----------|
| **Warmup** | 0-2 | 0.0 | Backbone settle ở LR mới, optimizer state stabilize. D-Branch không pull gradient. |
| **Ramp** | 3-5 | 0 → 1 linear | D-Branch dần dần được expose. Gradient nhỏ lúc đầu để backbone không sốc. |
| **Full** | 6+ | 1.0 | D-Branch train full lực. Backbone vẫn fine-tune nhẹ (LR 5e-6). |

### 5.3. Layered LR — tại sao 3 mức khác nhau?

| Group | LR | Lý do |
|-------|-----|-------|
| Backbone (Transformer + GCN) | `5e-6` | Đã pretrained — chỉ cần adapt nhẹ. LR thấp = drift chậm. |
| C/A-Branch (logits1 + logits2 heads) | `2e-5` | LR cũ của VadCLIP — giữ để C/A-branch không deteriorate. |
| D-Branch (random init) | `1e-4` | Cần train từ đầu — LR cao để converge nhanh trong epoch limit. |

LR ratio 1:4:20 phản ánh "đã trained nhiều" → "cần train thêm".

---

## 6. Tổng Kết Triết Lý Thiết Kế

1. **Một objective ↔ một loss term** — không gộp, không proxy. Dễ debug, dễ ablate.
2. **Hard label, không soft** — supervision rõ ràng từ HIVAU events. Không che đậy noise bằng smoothing.
3. **Architecture có induction bias đúng** — Conv1D match temporal locality của anomaly. Loss không phải "gánh" việc shape.
4. **Bảo vệ backbone** — Schedule C tránh random-init D-Branch phá AUC baseline.
5. **Một score `s_t` đọc tất cả metrics** — không ensemble, không fusion. Story rõ ràng cho thesis: "với D-Branch, AUC drop X, mAP@IoU tăng Y".

---

## 7. Liên Kết Tài Liệu

- **Plan kỹ thuật chi tiết** (file, config, lệnh): `2026-04-15-logits3-map-objective.md`
- **Source code**:
  - `src/dbranch.py` — D-Branch class
  - `src/losses_dbranch.py` — 4 loss functions
  - `src/model.py` — integration vào CLIPVAD
  - `src/train.py` — Schedule C training loop
  - `src/test.py` — evaluation với 3 metrics
- **Tests**: `tests/test_dbranch.py`, `tests/test_losses_dbranch.py`
