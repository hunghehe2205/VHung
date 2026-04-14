"""Per-video AUC breakdown for logits1 (p1).

Diagnoses whether the 0.8736 vs 0.8801 gap is uniform across test videos or
concentrated in a few. Outputs CSV + prints worst anomaly videos.

Usage:
    python src/diagnose_per_video.py --model-path final_model/model_ucf.pth
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from test import LABEL_MAP
import option


def run_per_video(model, testdataset, testdataloader, maxlen, prompt_text, gt, device):
    model.to(device)
    model.eval()

    per_video_p1 = []
    offset = 0
    results = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(testdataloader, desc='Per-video eval')):
            visual = item[0].squeeze(0)
            label = item[1][0]
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
            prob1_expanded = np.repeat(prob1, 16)

            per_video_p1.append(prob1_expanded)

            n_frames = prob1_expanded.shape[0]
            gt_slice = gt[offset:offset + n_frames]
            offset += n_frames

            path = testdataset.df.loc[i]['path']
            video_name = testdataset._extract_video_name(path)

            pos_ratio = float(gt_slice.mean())
            mean_score = float(prob1_expanded.mean())
            max_score = float(prob1_expanded.max())

            if gt_slice.min() < gt_slice.max():
                video_auc = float(roc_auc_score(gt_slice, prob1_expanded))
            else:
                video_auc = None

            # anomaly-frame vs normal-frame score gap (only for anomaly videos)
            if video_auc is not None:
                anom_mean = float(prob1_expanded[gt_slice == 1].mean())
                norm_mean = float(prob1_expanded[gt_slice == 0].mean())
                gap = anom_mean - norm_mean
            else:
                anom_mean = None
                norm_mean = float(prob1_expanded.mean())
                gap = None

            results.append({
                'idx': i,
                'video_name': video_name,
                'class': label,
                'n_frames': n_frames,
                'pos_ratio': pos_ratio,
                'video_auc': video_auc,
                'mean_score': mean_score,
                'max_score': max_score,
                'anom_mean': anom_mean,
                'norm_mean': norm_mean,
                'gap': gap,
            })

    all_p1 = np.concatenate(per_video_p1)
    assert all_p1.shape[0] == gt.shape[0], f"p1 {all_p1.shape} vs gt {gt.shape}"
    global_auc = float(roc_auc_score(gt, all_p1))
    return results, global_auc, all_p1


def summarize(results, global_auc, out_csv):
    anomaly_results = [r for r in results if r['video_auc'] is not None]
    normal_results = [r for r in results if r['video_auc'] is None]

    mean_per_video_auc = float(np.mean([r['video_auc'] for r in anomaly_results]))
    median_per_video_auc = float(np.median([r['video_auc'] for r in anomaly_results]))

    print(f"\n=== Global AUC (frame-pooled, all test frames) ===")
    print(f"  AUC1 global: {global_auc:.4f}  (target ≈ 0.8736, paper 0.8801)")
    print(f"\n=== Per-video stats (anomaly videos only, n={len(anomaly_results)}) ===")
    print(f"  Mean per-video AUC:   {mean_per_video_auc:.4f}")
    print(f"  Median per-video AUC: {median_per_video_auc:.4f}")
    print(f"\n=== Normal videos (n={len(normal_results)}, gt all-zero, AUC undefined) ===")
    norm_score_means = [r['mean_score'] for r in normal_results]
    norm_score_maxes = [r['max_score'] for r in normal_results]
    print(f"  Mean score per video: mean={np.mean(norm_score_means):.3f}, "
          f"max across videos={np.max(norm_score_means):.3f}")
    print(f"  Max score per video:  mean={np.mean(norm_score_maxes):.3f}")

    # Worst and best anomaly videos
    anomaly_sorted = sorted(anomaly_results, key=lambda r: r['video_auc'])
    print(f"\n=== Top-20 WORST anomaly videos by AUC ===")
    print(f"  {'idx':>4} {'class':>16} {'AUC':>6} {'gap':>6} {'pos%':>6} {'name':>32}")
    for r in anomaly_sorted[:20]:
        print(f"  {r['idx']:>4} {r['class']:>16} "
              f"{r['video_auc']:>6.3f} {r['gap']:>6.3f} "
              f"{r['pos_ratio']*100:>5.1f}% {r['video_name']:>32}")

    print(f"\n=== Top-20 BEST anomaly videos by AUC ===")
    for r in anomaly_sorted[-20:][::-1]:
        print(f"  {r['idx']:>4} {r['class']:>16} "
              f"{r['video_auc']:>6.3f} {r['gap']:>6.3f} "
              f"{r['pos_ratio']*100:>5.1f}% {r['video_name']:>32}")

    # Per-class stats
    classes = sorted(set(r['class'] for r in anomaly_results))
    print(f"\n=== Per-class AUC stats (anomaly only) ===")
    print(f"  {'class':>16} {'n':>3} {'mean':>6} {'median':>6} {'min':>6} {'max':>6}")
    for c in classes:
        aucs = [r['video_auc'] for r in anomaly_results if r['class'] == c]
        print(f"  {c:>16} {len(aucs):>3} "
              f"{np.mean(aucs):>6.3f} {np.median(aucs):>6.3f} "
              f"{np.min(aucs):>6.3f} {np.max(aucs):>6.3f}")

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nCSV written to {out_csv}")


if __name__ == '__main__':
    option.parser.add_argument('--out-csv', default='logs/per_video_auc.csv')
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

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    results, global_auc, all_p1 = run_per_video(
        model, testdataset, testdataloader, args.visual_length, prompt_text, gt, device
    )
    summarize(results, global_auc, args.out_csv)
