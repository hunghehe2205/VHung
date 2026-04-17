"""Abnormal-video localization diagnostic for CLIPVAD.

Per anomaly video, computes:
- gt_coverage, pred_coverage, over-prediction ratio
- frame precision/recall at thresh={0.3,0.5,0.7}
- best-IoU of adaptive-threshold proposals vs any GT segment
- peak location vs GT midpoint (is peak inside any GT?)
- multi-peak count (count of disconnected high-prob regions)

Aggregates to identify dominant failure mode (over-pred / wrong-loc / multi-peak).
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

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


def _upsample_repeat(v, factor=16):
    return np.repeat(v, factor)


def _adaptive_thresh_proposals(prob, thr_ratio=0.6):
    if prob.max() == prob.min():
        return []
    thr = prob.max() - (prob.max() - prob.min()) * thr_ratio
    mask = np.concatenate([[0.0], (prob > thr).astype(np.float32), [0.0]])
    diff = mask[1:] - mask[:-1]
    starts = np.where(diff == 1)[0].tolist()
    ends = np.where(diff == -1)[0].tolist()
    out = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            out.append((s, e, float(prob[s:e].max())))
    return out


def _iou_1d(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0 else 0.0


def _gt_frame_mask(n_frames, segments, labels):
    """Binary mask [n_frames] = 1 inside any non-Normal GT segment."""
    m = np.zeros(n_frames, dtype=np.float32)
    if segments is None or len(segments) == 0:
        return m
    for seg, lab in zip(segments, labels):
        if lab == 'A' or seg is None or len(seg) < 2:
            continue
        s = max(0, int(seg[0]))
        e = min(n_frames, int(seg[1]))
        if e > s:
            m[s:e] = 1.0
    return m


def _get_probs(model, loader, maxlen, prompt_text, device):
    probs, labels, paths = [], [], []
    df = loader.dataset.df
    with torch.no_grad():
        for i, item in enumerate(loader):
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

            _, logits1, _, _, _ = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1],
                                      logits1.shape[2])
            prob = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()
            probs.append(prob)
            labels.append(df.loc[i]['label'])
            paths.append(df.loc[i]['path'])
    return probs, labels, paths


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(LABEL_MAP)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device)
    model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                     map_location=device), strict=False)
    model.to(device).eval()

    probs, labels, paths = _get_probs(
        model, testloader, args.visual_length, prompt_text, device)

    anomaly_idx = [i for i, l in enumerate(labels) if str(l).lower() != 'normal']
    n_total = len(anomaly_idx)

    print('=' * 78)
    print(f'Model: {args.model_path}')
    print(f'Anomaly videos analyzed: {n_total}')
    print('=' * 78)

    # Per-video stats (frames are snippets, upsample ×16 to match GT which is in frames)
    over_ratios, precisions, recalls = [], [], []
    best_ious, peak_in_gt, n_peaks_list = [], [], []
    gt_covs, pred_covs = [], []
    per_video_info = []

    for i in anomaly_idx:
        prob_snip = probs[i]
        prob_frame = _upsample_repeat(prob_snip, 16)
        n_frames = len(prob_frame)
        gt_mask = _gt_frame_mask(n_frames, gtsegments[i], gtlabels[i])

        gt_cov = gt_mask.mean()
        pred_cov = (prob_frame > 0.5).mean()
        gt_covs.append(gt_cov)
        pred_covs.append(pred_cov)

        if gt_mask.sum() > 0:
            # Precision: of predicted-high frames, how many are GT
            pred_high = prob_frame > 0.5
            if pred_high.sum() > 0:
                prec = (pred_high & (gt_mask > 0.5)).sum() / pred_high.sum()
            else:
                prec = 0.0
            rec = ((prob_frame > 0.5) & (gt_mask > 0.5)).sum() / gt_mask.sum()
            precisions.append(prec)
            recalls.append(rec)
            over_ratios.append(pred_cov / max(gt_cov, 1e-6))
        else:
            precisions.append(np.nan)
            recalls.append(np.nan)
            over_ratios.append(np.nan)

        # Peak location analysis — on snippet-res prob
        peak_idx = int(np.argmax(prob_snip))
        peak_frame = peak_idx * 16 + 8
        peak_in_gt.append(1.0 if (peak_frame < n_frames and gt_mask[peak_frame] > 0.5) else 0.0)

        # Best IoU of adaptive proposals (snippet-res) vs GT (frame-res)
        props = _adaptive_thresh_proposals(prob_snip, thr_ratio=0.6)
        max_iou = 0.0
        for (s, e, _) in props:
            sf, ef = s * 16, e * 16
            for seg, lab in zip(gtsegments[i], gtlabels[i]):
                if lab == 'A' or seg is None or len(seg) < 2:
                    continue
                iou = _iou_1d(sf, ef, int(seg[0]), int(seg[1]))
                if iou > max_iou:
                    max_iou = iou
        best_ious.append(max_iou)

        # Count high-prob regions (disconnected) at thr=0.5 — multi-peak proxy
        mask = np.concatenate([[0.0], (prob_snip > 0.5).astype(np.float32), [0.0]])
        diff = mask[1:] - mask[:-1]
        n_peaks = int((diff == 1).sum())
        n_peaks_list.append(n_peaks)

        per_video_info.append({
            'idx': i, 'path': paths[i],
            'over': pred_cov / max(gt_cov, 1e-6) if gt_cov > 0 else np.nan,
            'prec': prec if gt_mask.sum() > 0 else np.nan,
            'rec': rec if gt_mask.sum() > 0 else np.nan,
            'peak_in_gt': peak_in_gt[-1],
            'best_iou': max_iou, 'n_peaks': n_peaks,
            'gt_cov': gt_cov, 'pred_cov': pred_cov,
        })

    gt_covs = np.array(gt_covs)
    pred_covs = np.array(pred_covs)
    over = np.array(over_ratios, dtype=np.float32)
    prec = np.array(precisions, dtype=np.float32)
    rec = np.array(recalls, dtype=np.float32)
    best_ious = np.array(best_ious)
    peak_in_gt = np.array(peak_in_gt)
    n_peaks_arr = np.array(n_peaks_list)

    def _d(name, arr):
        a = arr[np.isfinite(arr)]
        if len(a) == 0:
            print(f'  {name:32s}  n=0'); return
        print(f'  {name:32s}  median={np.median(a):.3f}  '
              f'mean={a.mean():.3f}  p10={np.percentile(a,10):.3f}  '
              f'p90={np.percentile(a,90):.3f}')

    print('\n[coverage distributions across anomaly videos]')
    _d('GT coverage (frac frames)', gt_covs)
    _d('Pred coverage @thr=0.5', pred_covs)
    _d('Pred/GT over-ratio', over)
    print('  → ratio=1 perfect, <1 under-predict, >1 over-predict')

    print('\n[frame-level precision/recall @thr=0.5 on anomaly videos]')
    _d('Precision (of pred, GT)', prec)
    _d('Recall    (of GT, pred)', rec)

    print('\n[best IoU of adaptive-threshold proposals vs any GT segment]')
    _d('Best IoU per video', best_ious)
    for th in [0.1, 0.3, 0.5, 0.7]:
        n = int((best_ious >= th).sum())
        print(f'  best_IoU >= {th:.1f}   {n:3d}/{n_total}  ({100.0*n/n_total:5.1f}%)')

    print('\n[peak-location check]')
    pct = 100.0 * peak_in_gt.mean()
    print(f'  argmax(prob) lies inside a GT segment:  {int(peak_in_gt.sum())}/{n_total}  ({pct:.1f}%)')

    print('\n[multi-peak count — high-prob regions @thr=0.5]')
    _d('Disconnected peaks per video', n_peaks_arr)
    for k in [1, 2, 3, 5]:
        n = int((n_peaks_arr >= k).sum())
        print(f'  #peaks >= {k}   {n:3d}/{n_total}  ({100.0*n/n_total:5.1f}%)')

    print('\n[top-5 worst by over-prediction ratio]')
    valid = [v for v in per_video_info if np.isfinite(v['over'])]
    valid.sort(key=lambda v: -v['over'])
    for v in valid[:5]:
        vid = os.path.basename(v['path']).replace('.npy', '')
        print(f"  {vid:42s}  over={v['over']:.1f}x  "
              f"gt_cov={v['gt_cov']:.2f}  pred_cov={v['pred_cov']:.2f}  "
              f"best_iou={v['best_iou']:.2f}  peaks={v['n_peaks']}")

    print('\n[top-5 best by IoU]')
    valid2 = sorted(per_video_info, key=lambda v: -v['best_iou'])
    for v in valid2[:5]:
        vid = os.path.basename(v['path']).replace('.npy', '')
        print(f"  {vid:42s}  best_iou={v['best_iou']:.2f}  "
              f"over={v['over']:.1f}x  peak_in_gt={int(v['peak_in_gt'])}")

    print('\n[top-5 worst — peak outside GT AND best_iou < 0.1]')
    wrong = [v for v in per_video_info
             if v['peak_in_gt'] == 0 and v['best_iou'] < 0.1]
    wrong.sort(key=lambda v: -v['pred_cov'])
    print(f'  n={len(wrong)}')
    for v in wrong[:5]:
        vid = os.path.basename(v['path']).replace('.npy', '')
        print(f"  {vid:42s}  pred_cov={v['pred_cov']:.2f}  "
              f"gt_cov={v['gt_cov']:.2f}  peaks={v['n_peaks']}")

    # Failure mode summary
    print('\n[dominant failure mode]')
    # Define rough buckets on the n_total videos
    over_heavy = np.sum(over > 2.0)  # pred ≥ 2× GT coverage
    peak_wrong = int(n_total - peak_in_gt.sum())
    multi_peak = int((n_peaks_arr >= 2).sum())
    tight_ok = int((best_ious >= 0.5).sum())
    print(f'  over-predict (pred/gt > 2×):  {over_heavy}/{n_total}  ({100.0*over_heavy/n_total:.1f}%)')
    print(f'  peak outside any GT segment:  {peak_wrong}/{n_total}  ({100.0*peak_wrong/n_total:.1f}%)')
    print(f'  multi-peak (≥2 high regions): {multi_peak}/{n_total}  ({100.0*multi_peak/n_total:.1f}%)')
    print(f'  tight localization (IoU≥0.5): {tight_ok}/{n_total}  ({100.0*tight_ok/n_total:.1f}%)')
    print('=' * 78)


if __name__ == '__main__':
    main()
