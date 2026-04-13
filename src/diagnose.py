"""Diagnostic analysis of anomaly map behavior from checkpoint.

Usage:
    python src/diagnose.py --checkpoint final_model/checkpoint_phaseB.pth
    python src/diagnose.py --checkpoint final_model/checkpoint.pth --output diagnose_phaseA.txt
"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from collections import defaultdict

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def run_diagnostic(model, testloader, maxlen, prompt_text, gt, device):
    model.to(device)
    model.eval()

    # Per-video storage
    video_results = []
    all_prob1 = []
    all_prob2 = []
    all_prob3 = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(testloader, desc='Diagnosing')):
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

            _, logits1, logits2, logits3 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])
            logits3 = logits3.reshape(-1, logits3.shape[2])

            p1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()
            p2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0]).cpu().numpy()
            p3 = torch.sigmoid(logits3[0:len_cur].squeeze(-1)).cpu().numpy()
            raw3 = logits3[0:len_cur].squeeze(-1).cpu().numpy()  # raw logits before sigmoid

            all_prob1.append(p1)
            all_prob2.append(p2)
            all_prob3.append(p3)

            video_results.append({
                'idx': i,
                'label': label,
                'length': len_cur,
                'p1_mean': p1.mean(), 'p1_std': p1.std(), 'p1_max': p1.max(), 'p1_min': p1.min(),
                'p2_mean': p2.mean(), 'p2_std': p2.std(), 'p2_max': p2.max(), 'p2_min': p2.min(),
                'p3_mean': p3.mean(), 'p3_std': p3.std(), 'p3_max': p3.max(), 'p3_min': p3.min(),
                'raw3_mean': raw3.mean(), 'raw3_std': raw3.std(), 'raw3_max': raw3.max(), 'raw3_min': raw3.min(),
            })

    # Frame-level metrics
    gt_frames = gt
    pred1 = np.concatenate([np.repeat(p, 16) for p in all_prob1])
    pred2 = np.concatenate([np.repeat(p, 16) for p in all_prob2])
    pred3 = np.concatenate([np.repeat(p, 16) for p in all_prob3])

    # Ensemble variations
    pred_13 = 0.5 * pred1 + 0.5 * pred3
    pred_123 = (pred1 + pred2 + pred3) / 3
    pred_1w3 = 0.7 * pred1 + 0.3 * pred3

    return video_results, {
        'AUC1': roc_auc_score(gt_frames, pred1),
        'AP1': average_precision_score(gt_frames, pred1),
        'AUC2': roc_auc_score(gt_frames, pred2),
        'AP2': average_precision_score(gt_frames, pred2),
        'AUC3': roc_auc_score(gt_frames, pred3),
        'AP3': average_precision_score(gt_frames, pred3),
        'AUC_avg13': roc_auc_score(gt_frames, pred_13),
        'AUC_avg123': roc_auc_score(gt_frames, pred_123),
        'AUC_w13_7_3': roc_auc_score(gt_frames, pred_1w3),
    }


def print_report(video_results, metrics, output_file=None):
    lines = []

    def p(s=""):
        lines.append(s)

    p("=" * 80)
    p("ANOMALY MAP DIAGNOSTIC REPORT")
    p("=" * 80)

    # 1. Overall AUC
    p("\n--- 1. OVERALL METRICS ---")
    p(f"  AUC1 (logits1):       {metrics['AUC1']:.4f}   AP1: {metrics['AP1']:.4f}")
    p(f"  AUC2 (logits2):       {metrics['AUC2']:.4f}   AP2: {metrics['AP2']:.4f}")
    p(f"  AUC3 (logits3):       {metrics['AUC3']:.4f}   AP3: {metrics['AP3']:.4f}")
    p(f"  AUC ensemble(1+3)/2:  {metrics['AUC_avg13']:.4f}")
    p(f"  AUC ensemble(1+2+3)/3:{metrics['AUC_avg123']:.4f}")
    p(f"  AUC 0.7*p1+0.3*p3:   {metrics['AUC_w13_7_3']:.4f}")

    # 2. Score distributions by category
    p("\n--- 2. SCORE DISTRIBUTIONS ---")
    normal_vids = [v for v in video_results if v['label'] == 'Normal']
    anomaly_vids = [v for v in video_results if v['label'] != 'Normal']

    for head, prefix in [('p1', 'logits1'), ('p2', 'logits2'), ('p3', 'logits3')]:
        p(f"\n  [{prefix}]")
        nm = [v[f'{head}_mean'] for v in normal_vids]
        am = [v[f'{head}_mean'] for v in anomaly_vids]
        p(f"    Normal  ({len(nm):3d} vids): mean_of_means={np.mean(nm):.4f} std={np.std(nm):.4f} "
          f"range=[{np.min(nm):.4f}, {np.max(nm):.4f}]")
        p(f"    Anomaly ({len(am):3d} vids): mean_of_means={np.mean(am):.4f} std={np.std(am):.4f} "
          f"range=[{np.min(am):.4f}, {np.max(am):.4f}]")
        p(f"    Gap (anomaly - normal): {np.mean(am) - np.mean(nm):.4f}")
        # Overlap: how many normal videos have mean score > min anomaly mean
        overlap = sum(1 for n in nm if n > np.percentile(am, 10))
        p(f"    Overlap: {overlap}/{len(nm)} normal vids have mean > anomaly p10")

    # 3. Raw logits3 distribution (before sigmoid)
    p("\n--- 3. RAW LOGITS3 (before sigmoid) ---")
    nr = [v['raw3_mean'] for v in normal_vids]
    ar = [v['raw3_mean'] for v in anomaly_vids]
    p(f"  Normal:  mean={np.mean(nr):.4f} std={np.std(nr):.4f} range=[{np.min(nr):.4f}, {np.max(nr):.4f}]")
    p(f"  Anomaly: mean={np.mean(ar):.4f} std={np.std(ar):.4f} range=[{np.min(ar):.4f}, {np.max(ar):.4f}]")
    p(f"  Gap: {np.mean(ar) - np.mean(nr):.4f}")
    p(f"  → If gap is large in logit space, Platt scaling can calibrate effectively")

    # 4. Per-class breakdown
    p("\n--- 4. PER-CLASS AUC3 MEAN SCORES ---")
    class_stats = defaultdict(list)
    for v in video_results:
        class_stats[v['label']].append(v['p3_mean'])

    p(f"  {'Class':<20s} {'Count':>5s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    for cls in sorted(class_stats.keys()):
        vals = class_stats[cls]
        p(f"  {cls:<20s} {len(vals):5d} {np.mean(vals):8.4f} {np.std(vals):8.4f} "
          f"{np.min(vals):8.4f} {np.max(vals):8.4f}")

    # 5. Worst and best videos
    p("\n--- 5. INTERESTING VIDEOS ---")
    # Highest score normal videos (potential false positives)
    normal_by_score = sorted(normal_vids, key=lambda v: v['p3_mean'], reverse=True)
    p("\n  Top 5 normal videos with HIGHEST logits3 score (false positive risk):")
    for v in normal_by_score[:5]:
        p(f"    video_{v['idx']:04d} mean={v['p3_mean']:.4f} max={v['p3_max']:.4f}")

    # Lowest score anomaly videos (potential misses)
    anomaly_by_score = sorted(anomaly_vids, key=lambda v: v['p3_mean'])
    p("\n  Top 5 anomaly videos with LOWEST logits3 score (miss risk):")
    for v in anomaly_by_score[:5]:
        p(f"    video_{v['idx']:04d} ({v['label']}) mean={v['p3_mean']:.4f} max={v['p3_max']:.4f}")

    # 6. Correlation between heads
    p("\n--- 6. HEAD CORRELATION ---")
    p1_means = [v['p1_mean'] for v in video_results]
    p3_means = [v['p3_mean'] for v in video_results]
    p2_means = [v['p2_mean'] for v in video_results]
    corr_13 = np.corrcoef(p1_means, p3_means)[0, 1]
    corr_12 = np.corrcoef(p1_means, p2_means)[0, 1]
    corr_23 = np.corrcoef(p2_means, p3_means)[0, 1]
    p(f"  Correlation p1 vs p3: {corr_13:.4f}")
    p(f"  Correlation p1 vs p2: {corr_12:.4f}")
    p(f"  Correlation p2 vs p3: {corr_23:.4f}")
    p(f"  → Low correlation = heads capture different info = ensemble potential")
    p(f"  → High correlation = heads redundant = ensemble less useful")

    # 7. Platt scaling simulation
    p("\n--- 7. PLATT SCALING SIMULATION ---")
    p("  Testing if temperature scaling on logits3 could improve calibration:")
    all_raw_normal = [v['raw3_mean'] for v in normal_vids]
    all_raw_anomaly = [v['raw3_mean'] for v in anomaly_vids]
    raw_gap = np.mean(all_raw_anomaly) - np.mean(all_raw_normal)
    p(f"  Raw logit gap: {raw_gap:.4f}")
    if raw_gap > 0.3:
        p(f"  → Gap > 0.3 in logit space: Platt scaling LIKELY effective")
    elif raw_gap > 0.1:
        p(f"  → Gap 0.1-0.3 in logit space: Platt scaling MAYBE effective")
    else:
        p(f"  → Gap < 0.1 in logit space: Platt scaling UNLIKELY to help much")

    # Simulate different temperatures
    for temp in [0.5, 0.3, 0.1]:
        bias = (np.mean(all_raw_normal) + np.mean(all_raw_anomaly)) / 2
        cal_normal = 1 / (1 + np.exp(-(np.array(all_raw_normal) - bias) / temp))
        cal_anomaly = 1 / (1 + np.exp(-(np.array(all_raw_anomaly) - bias) / temp))
        p(f"  temp={temp}: normal_mean={cal_normal.mean():.4f} anomaly_mean={cal_anomaly.mean():.4f} "
          f"gap={cal_anomaly.mean() - cal_normal.mean():.4f}")

    p("\n" + "=" * 80)

    report = "\n".join(lines)
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='diagnose_report.txt')
    args_diag = parser.parse_args()

    # Load model args
    model_args = option.parser.parse_args([])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    testdataset = UCFDataset(model_args.visual_length, model_args.test_list, True, LABEL_MAP)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(model_args.gt_path)

    model = CLIPVAD(model_args.classes_num, model_args.embed_dim, model_args.visual_length,
                    model_args.visual_width, model_args.visual_head, model_args.visual_layers,
                    model_args.attn_window, model_args.prompt_prefix, model_args.prompt_postfix, device)

    checkpoint = torch.load(args_diag.checkpoint, weights_only=False, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: epoch={checkpoint.get('epoch', '?')} best_auc={checkpoint.get('ap', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state dict directly")

    video_results, metrics = run_diagnostic(model, testloader, model_args.visual_length,
                                            prompt_text, gt, device)
    print_report(video_results, metrics, args_diag.output)


if __name__ == '__main__':
    main()
