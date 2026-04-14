"""Per-video score normalization eval for logits1 (p1).

Tests whether normalizing p1 within each test video improves global AUC.
Variants: z-score, percentile rank, and alpha-blend with raw p1.

Usage:
    python src/pervideo_norm_eval.py --model-path final_model/model_ucf.pth
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from test import LABEL_MAP
import option


def collect_per_video_p1(model, testdataloader, maxlen, prompt_text, device):
    """Returns list of per-video prob1 arrays (each expanded 16× to frames)."""
    model.to(device)
    model.eval()
    per_video = []

    with torch.no_grad():
        for item in tqdm(testdataloader, desc='Collecting p1'):
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
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, _, _ = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1)[0:len_cur]
            prob1 = torch.sigmoid(logits1).cpu().numpy()
            per_video.append(np.repeat(prob1, 16))

    return per_video


def znorm(x, eps=1e-8):
    m = x.mean()
    s = x.std() + eps
    # map to (0, 1) via sigmoid of z-score to keep comparable range
    z = (x - m) / s
    return 1.0 / (1.0 + np.exp(-z))


def percentile_rank(x):
    ranks = rankdata(x, method='average') - 1
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return (ranks / (len(x) - 1)).astype(np.float32)


def apply_per_video(per_video, fn):
    return np.concatenate([fn(v) for v in per_video])


def report(name, scores, gt):
    auc = roc_auc_score(gt, scores)
    ap = average_precision_score(gt, scores)
    anom = scores[gt == 1]
    norm = scores[gt == 0]
    gap = anom.mean() - norm.mean()
    cov = (anom > 0.5).mean()
    print(f"  {name:>30}: AUC={auc:.4f} AP={ap:.4f} "
          f"gap={gap:.3f} cov={cov*100:.1f}% "
          f"n_mean={norm.mean():.3f} a_mean={anom.mean():.3f}")
    return auc


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)

    ckpt = torch.load(args.model_path, weights_only=False, map_location=device)
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.model_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    per_video = collect_per_video_p1(model, testdataloader, args.visual_length, prompt_text, device)

    # baseline
    p1_raw = np.concatenate(per_video)
    assert p1_raw.shape == gt.shape, f"{p1_raw.shape} vs {gt.shape}"

    print("\n=== Normalization variants ===")
    print(f"  {'variant':>30}   metric")
    baseline = report('p1 raw (baseline)', p1_raw, gt)

    # Variant 1: per-video z-score → sigmoid
    p1_z = apply_per_video(per_video, znorm)
    report('per-video z→sigmoid', p1_z, gt)

    # Variant 2: per-video percentile rank
    p1_pr = apply_per_video(per_video, percentile_rank)
    report('per-video percentile_rank', p1_pr, gt)

    # Variant 3: alpha-blend raw + z
    print(f"\n=== alpha · p1_raw + (1-alpha) · p1_z ===")
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        blend = alpha * p1_raw + (1 - alpha) * p1_z
        report(f'alpha_z={alpha:.1f}', blend, gt)

    # Variant 4: alpha-blend raw + percentile
    print(f"\n=== alpha · p1_raw + (1-alpha) · p1_pr ===")
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        blend = alpha * p1_raw + (1 - alpha) * p1_pr
        report(f'alpha_pr={alpha:.1f}', blend, gt)

    print(f"\n=== Summary ===")
    print(f"  Target: 0.8801 (baseline raw ≈ {baseline:.4f})")
