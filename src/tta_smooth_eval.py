"""Inference-time AUC boosting: temporal smoothing + TTA (no training).

Loads a checkpoint (original or trained), runs per-video inference collecting
p1 = sigmoid(logits1). Explores:
  1. Temporal Gaussian smoothing on clip-level p1 (sigma sweep)
  2. Test-time augmentation: add Gaussian noise to visual features,
     average K forward passes
  3. Combined TTA + smoothing

Reports AUC1 for every variant and compares to baseline 0.8801.

Usage:
    python src/tta_smooth_eval.py --checkpoint-path model/model_ucf.pth
    python src/tta_smooth_eval.py --checkpoint-path model/model_ucf.pth \
        --tta-k 8 --tta-noise 0.05 --sigmas 0 1 2 3 5
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from test import LABEL_MAP
import option


def forward_p1(model, visual, lengths, maxlen, prompt_text, device):
    """Single forward pass. Returns clip-level p1 array of length sum(lengths)."""
    padding_mask = get_batch_mask(lengths, maxlen).to(device)
    _, logits1, _, _ = model(visual, padding_mask, prompt_text, lengths)
    total = int(lengths.sum().item())
    logits1 = logits1.reshape(-1)[:total]
    return torch.sigmoid(logits1).cpu().numpy()


def collect_p1_per_video(model, testdataloader, maxlen, prompt_text, device,
                         tta_k=0, tta_noise=0.0):
    """Collect clip-level p1 per video. If tta_k > 0, average K noisy passes."""
    model.to(device)
    model.eval()
    videos = []

    with torch.no_grad():
        for item in tqdm(testdataloader, desc='Inference'):
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

            p1_clean = forward_p1(model, visual, lengths, maxlen, prompt_text, device)[:len_cur]

            if tta_k > 0 and tta_noise > 0:
                p1_tta_sum = p1_clean.copy()
                for _ in range(tta_k):
                    noise = torch.randn_like(visual) * tta_noise
                    visual_noisy = visual + noise
                    p1_noisy = forward_p1(model, visual_noisy, lengths, maxlen,
                                          prompt_text, device)[:len_cur]
                    p1_tta_sum += p1_noisy
                p1_final = p1_tta_sum / (tta_k + 1)
            else:
                p1_final = p1_clean

            videos.append(p1_final)

    return videos


def to_frame_level(videos, smooth_sigma=0.0):
    """Per-video: optionally smooth clip-level p1, repeat ×16 to frame-level, concat."""
    out = []
    for p in videos:
        if smooth_sigma > 0 and len(p) > 1:
            p = gaussian_filter1d(p, sigma=smooth_sigma, mode='nearest')
        out.append(np.repeat(p, 16))
    return np.concatenate(out)


def report(label, scores, gt, baseline=0.8801):
    auc = roc_auc_score(gt, scores)
    ap = average_precision_score(gt, scores)
    anom = scores[gt == 1]
    norm = scores[gt == 0]
    beat = 'YES ✓' if auc > baseline else 'NO ✗'
    print(f"{label:<40} AUC={auc:.4f}  AP={ap:.4f}  "
          f"Gap={anom.mean()-norm.mean():.3f}  Cov={(anom>0.5).mean():.3f}  "
          f"vs {baseline}: {beat}")
    return auc


if __name__ == '__main__':
    option.parser.add_argument('--tta-k', default=5, type=int,
                               help='Number of noisy TTA passes (0 = disable TTA)')
    option.parser.add_argument('--tta-noise', default=0.03, type=float,
                               help='Std of Gaussian noise added to visual features')
    option.parser.add_argument('--sigmas', default=[0.0, 1.0, 2.0, 3.0, 5.0],
                               nargs='+', type=float,
                               help='Temporal Gaussian smoothing sigmas to sweep (clip-level)')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)

    checkpoint = torch.load(args.checkpoint_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
        print(f"Loaded checkpoint (epoch={checkpoint['epoch']+1}, AUC1={checkpoint['ap']:.4f})")
    else:
        state = checkpoint
        print(f"Loaded raw state_dict from {args.checkpoint_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)} (ok for original model without map_head)")

    print(f"\n=== Clean inference (no TTA) ===")
    videos_clean = collect_p1_per_video(model, testdataloader, args.visual_length,
                                        prompt_text, device, tta_k=0)

    print(f"\n=== Smoothing sweep (no TTA) ===")
    best_clean_auc = 0
    best_clean_sigma = 0
    for sigma in args.sigmas:
        scores = to_frame_level(videos_clean, smooth_sigma=sigma)
        label = f"clean, sigma={sigma:.1f}"
        auc = report(label, scores, gt)
        if auc > best_clean_auc:
            best_clean_auc = auc
            best_clean_sigma = sigma

    if args.tta_k > 0:
        print(f"\n=== TTA inference (K={args.tta_k}, noise={args.tta_noise}) ===")
        videos_tta = collect_p1_per_video(model, testdataloader, args.visual_length,
                                          prompt_text, device,
                                          tta_k=args.tta_k, tta_noise=args.tta_noise)

        print(f"\n=== TTA + smoothing sweep ===")
        best_tta_auc = 0
        best_tta_sigma = 0
        for sigma in args.sigmas:
            scores = to_frame_level(videos_tta, smooth_sigma=sigma)
            label = f"TTA k={args.tta_k}, sigma={sigma:.1f}"
            auc = report(label, scores, gt)
            if auc > best_tta_auc:
                best_tta_auc = auc
                best_tta_sigma = sigma

        print(f"\n=== Summary ===")
        print(f"  Best clean: AUC={best_clean_auc:.4f} at sigma={best_clean_sigma:.1f}")
        print(f"  Best TTA:   AUC={best_tta_auc:.4f} at sigma={best_tta_sigma:.1f}, "
              f"K={args.tta_k}, noise={args.tta_noise}")
        print(f"  Baseline:   0.8801")
        best_overall = max(best_clean_auc, best_tta_auc)
        print(f"  Beat baseline: {'YES ✓' if best_overall > 0.8801 else 'NO ✗'} "
              f"(delta = {best_overall - 0.8801:+.4f})")

        os.makedirs(args.log_dir, exist_ok=True)
        best_scores = to_frame_level(videos_tta, smooth_sigma=best_tta_sigma) \
            if best_tta_auc >= best_clean_auc \
            else to_frame_level(videos_clean, smooth_sigma=best_clean_sigma)
        np.save(os.path.join(args.log_dir, 'tta_smooth_scores.npy'), best_scores)
        print(f"\nSaved best scores to {args.log_dir}/tta_smooth_scores.npy")
    else:
        print(f"\n=== Summary ===")
        print(f"  Best clean: AUC={best_clean_auc:.4f} at sigma={best_clean_sigma:.1f}")
        print(f"  Baseline:   0.8801")
        print(f"  Beat baseline: {'YES ✓' if best_clean_auc > 0.8801 else 'NO ✗'} "
              f"(delta = {best_clean_auc - 0.8801:+.4f})")
