# Plan: InternVAD Full Pipeline — Classification Branch Only (UCF)

## Context
Chuyển VadCLIP sang dùng InternVL vision encoder (`ppxin321/HolmesVAU-2B`, hidden_size=1024) thay cho CLIP (hidden_size=512). Chỉ dùng **branch classification** (binary anomaly detection), bỏ text-visual alignment branch. Chỉ focus UCF-Crime dataset.

## Cấu trúc thư mục

```
/Users/hunghehe2205/Projects/VHung/
├── VadCLIP/                  # KHÔNG ĐỤNG
├── intern_vad.py             # Model (sửa: bỏ text branch, simplify)
├── crop_intern.py            # Feature extraction (đã có, cần mở rộng cho batch UCF)
├── internvl_utils.py         # Utils cho InternVL (đã có)
├── src/                      # THƯ MỤC MỚI
│   ├── extract_features.py   # Script extract features toàn bộ UCF dataset
│   ├── ucf_option.py         # Config (visual_width=1024)
│   ├── ucf_train.py          # Training loop (chỉ CLAS2 loss)
│   ├── ucf_test.py           # Testing (chỉ logits1 → sigmoid → AUC/AP)
│   └── utils/
│       ├── dataset.py        # UCFDataset cho InternVL features
│       └── tools.py          # Reuse từ VadCLIP
├── list/                     # THƯ MỤC MỚI
│   ├── ucf_intern_rgb.csv    # Train list (path tới InternVL .npy features)
│   └── ucf_intern_rgbtest.csv# Test list
```

---

## Pipeline chi tiết

### Phase 1: Feature Extraction — `src/extract_features.py`

**Mục đích:** Extract CLS token features từ InternVL cho toàn bộ UCF-Crime videos.

**Input:** UCF-Crime raw videos (`.avi`/`.mp4`)
**Output:** `.npy` files, mỗi file shape `[T, 1024]` — T = số frames / 16 (uniform sampling)

**Logic (dựa trên `crop_intern.py` đã có):**
1. Load InternVL model (`ppxin321/HolmesVAU-2B`) — freeze, eval, bfloat16
2. Duyệt qua toàn bộ videos trong UCF-Crime dataset
3. Mỗi video:
   - Đọc toàn bộ frames bằng OpenCV
   - Với mỗi crop type (0-9, 10-crop augmentation giống VadCLIP):
     - Resize → crop 448x448 (+ flip cho type 5-9)
     - Mỗi 16 frames liên tiếp → uniform sample → batch qua InternVL vision model
     - Lấy CLS token `[1, 1024]` mỗi frame
     - Stack → `[T, 1024]` → save `.npy`
4. Naming convention: `{VideoName}_{crop_id}.npy` (giống VadCLIP: `Abuse001_x264__0.npy`)

**Tham khảo format VadCLIP:**
- CSV format: `path,label`
- Mỗi video có 10 crops (`_0` tới `_9`)
- Train: 16100 entries (1610 videos x 10 crops)
- Test: 290 entries (290 videos x 1 — không crop)

### Phase 2: Tạo CSV lists — `list/`

**`list/ucf_intern_rgb.csv`** (train):
```
path,label
/path/to/InternFeatures/Abuse/Abuse001_x264__0.npy,Abuse
/path/to/InternFeatures/Abuse/Abuse001_x264__1.npy,Abuse
...
```

**`list/ucf_intern_rgbtest.csv`** (test):
```
path,label
/path/to/InternFeatures/Abuse/Abuse028_x264__0.npy,Abuse
...
```

Có thể generate CSV tự động trong `extract_features.py` hoặc script riêng.

### Phase 3: Chỉnh Model — `intern_vad.py`

**Thay đổi so với hiện tại:**

| Bỏ | Giữ/Thêm |
|----|-----------|
| `embed_dim` param | `visual_width=1024` |
| `prompt_prefix`, `prompt_postfix` params | `visual_length`, `visual_head`, `visual_layers`, `attn_window` |
| `mlp1` (text branch MLP) | `mlp2` (classification MLP) |
| `vision_model` (InternVL trong constructor) | `encode_video()` giữ nguyên |
| `text_prompt_embeddings` (đã bỏ) | `classifier`, `gc1-4`, `temporal` |

**Forward mới:**
```python
def forward(self, visual, lengths):
    visual_features = self.encode_video(visual, None, lengths)  # [B, T, 1024]
    logits = self.classifier(visual_features + self.mlp2(visual_features))  # [B, T, 1]
    return logits
```

**Lưu ý:** `vision_model` (InternVL) chỉ dùng trong `extract_features.py`, KHÔNG cần load trong model training (features đã pre-extracted thành `.npy`).

### Phase 4: Config — `src/ucf_option.py`

```python
# Thay đổi so với VadCLIP:
--visual-width    1024    # (VadCLIP: 512) — khớp InternVL hidden_size
--visual-head     8       # (VadCLIP: 1) — 1024/8 = 128 dim/head
--visual-length   256     # giữ nguyên
--visual-layers   2       # giữ nguyên
--attn-window     8       # giữ nguyên
--batch-size      64      # giữ nguyên
--lr              2e-5    # giữ nguyên
--max-epoch       10      # giữ nguyên

# Bỏ: --embed-dim, --prompt-prefix, --prompt-postfix, --classes-num
# Paths: trỏ tới list/ mới và ground truth UCF (reuse từ VadCLIP/list/)
```

### Phase 5: Training — `src/ucf_train.py`

**Loss duy nhất: CLAS2 (Binary Classification + MIL)**
```python
def CLAS2(logits, labels, lengths, device):
    # logits: [B, T, 1] → sigmoid → top-k → mean → BCE
    # labels: binary [0 or 1] (0=normal, 1=anomaly)
    # Top-k: k = length/16 + 1 (giống VadCLIP)
```

**Training loop:**
1. 2 DataLoader: `normal_loader` + `anomaly_loader` (giống VadCLIP)
2. Mỗi batch: concat normal + anomaly features
3. Labels: binary `[1,0]` normal, `[0,1]` anomaly → extract cột 0 → `1 - labels[:,0]`
4. Forward → CLAS2 loss → backward → AdamW
5. Eval mỗi 1280 steps → save best checkpoint theo AUC

### Phase 6: Testing — `src/ucf_test.py`

- Forward: `logits = model(visual, lengths)` → `sigmoid(logits)` → prob
- Metrics: `roc_auc_score(gt, prob)`, `average_precision_score(gt, prob)`
- Repeat x16 để match frame-level ground truth (giống VadCLIP)

### Phase 7: Dataset & Utils — `src/utils/`

**`dataset.py`:**
- `UCFDataset` giống VadCLIP nhưng load `.npy` shape `[T, 1024]`
- Train: `process_feat()` — uniform extract hoặc pad tới `visual_length`
- Test: `process_split()` — split long clips thành overlapping windows

**`tools.py`:**
- Reuse trực tiếp từ VadCLIP:
```python
from VadCLIP.src.utils.tools import (
    process_feat, process_split, pad, uniform_extract,
    get_batch_mask, get_batch_label, get_prompt_text
)
```

---

## Thứ tự thực hiện

1. **Chỉnh `intern_vad.py`** — bỏ text branch, bỏ vision_model, simplify forward
2. **Tạo `src/ucf_option.py`** — config mới (1024 width)
3. **Tạo `src/utils/`** — dataset.py + tools.py
4. **Tạo `src/ucf_test.py`** — test function (chỉ logits1)
5. **Tạo `src/ucf_train.py`** — training loop + CLAS2 loss
6. **Tạo `src/extract_features.py`** — batch extract InternVL features cho UCF

## Verification

1. Extract features 1 video test → check shape `[T, 1024]`
2. `python src/ucf_train.py` → model init OK, loss backward OK
3. AUC/AP output mỗi 1280 steps
4. So sánh AUC với VadCLIP branch 1 baseline
