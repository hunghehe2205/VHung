"""Per-category diagnostic: which GT segments are missed and why.

Identifies the 33% GT segments with zero coverage, breaks down by category,
and analyzes actionness profile per category.
"""
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
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

# Reverse: csv label → GT label mapping
CSV_TO_GT = {}
for csv_lab in LABEL_MAP:
    if csv_lab != 'Normal':
        CSV_TO_GT[csv_lab] = csv_lab


def _upsample_repeat(v):
    return np.repeat(v, 16)


def _compute_iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = (pe - ps) + (ge - gs) - inter
    return inter / union if union > 0 else 0.0


def _threshold_proposals(prob, thr_ratio=0.6):
    if prob.max() == prob.min():
        return []
    threshold = prob.max() - (prob.max() - prob.min()) * thr_ratio
    vid_pred = np.concatenate([np.zeros(1), (prob > threshold).astype('float32'), np.zeros(1)])
    vid_pred_diff = [vid_pred[t] - vid_pred[t - 1] for t in range(1, len(vid_pred))]
    s = [k for k, item in enumerate(vid_pred_diff) if item == 1]
    e = [k for k, item in enumerate(vid_pred_diff) if item == -1]
    proposals = []
    for j in range(len(s)):
        if e[j] - s[j] >= 2:
            score = float(np.max(prob[s[j]:e[j]]))
            proposals.append((s[j], e[j], score))
    proposals.sort(key=lambda x: -x[2])
    return proposals


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
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device)
    model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                     map_location=device), strict=False)
    model.to(device).eval()

    maxlen = args.visual_length
    labels_csv = list(testdataset.df['label'])

    # Inference
    all_probs = []
    with torch.no_grad():
        for i, item in enumerate(testloader):
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
            logits1 = logits1.reshape(-1, logits1.shape[2])
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()
            all_probs.append(_upsample_repeat(prob1))

    # ========== Per-category analysis ==========
    # Collect per-GT-segment stats
    seg_info = []  # list of dicts
    for i in range(len(all_probs)):
        is_normal = str(labels_csv[i]).lower() == 'normal'
        if is_normal:
            continue
        category = str(labels_csv[i])
        prob = all_probs[i]
        n_frames = len(prob)
        proposals = _threshold_proposals(prob)

        for j in range(len(gtsegments[i])):
            if gtlabels[i][j] == 'A':
                continue
            gs = float(gtsegments[i][j][0])
            ge = float(gtsegments[i][j][1])
            gt_len = ge - gs

            # Best IoU with any proposal
            best_iou = 0.0
            best_prop = None
            for ps, pe, sc in proposals:
                iou = _compute_iou(ps, pe, gs, ge)
                if iou > best_iou:
                    best_iou = iou
                    best_prop = (ps, pe, sc)

            # Actionness inside this GT segment
            gs_i = max(0, int(gs))
            ge_i = min(n_frames, int(ge))
            if ge_i > gs_i:
                inside_mean = float(prob[gs_i:ge_i].mean())
                inside_max = float(prob[gs_i:ge_i].max())
            else:
                inside_mean = 0.0
                inside_max = 0.0

            # Video-level max prob
            video_max = float(prob.max())

            seg_info.append({
                'video_idx': i,
                'category': category,
                'gt_start': gs,
                'gt_end': ge,
                'gt_len': gt_len,
                'best_iou': best_iou,
                'best_prop': best_prop,
                'inside_mean': inside_mean,
                'inside_max': inside_max,
                'video_max': video_max,
                'n_proposals': len(proposals),
            })

    # ========== Print results ==========
    print('=' * 70)
    print('PER-CATEGORY GT SEGMENT ANALYSIS')
    print('=' * 70)
    print(f'Total GT segments (anomaly): {len(seg_info)}')
    print()

    # Group by category
    by_cat = defaultdict(list)
    for s in seg_info:
        by_cat[s['category']].append(s)

    # Per-category table
    print(f'{"Category":<16} {"#GT":>4} {"Covered":>8} {"Miss":>5} '
          f'{"avgIoU":>7} {"IoU>0.5":>8} {"actInside":>10} {"actMax":>7}')
    print('-' * 70)

    missed_segments = []
    for cat in sorted(by_cat.keys()):
        segs = by_cat[cat]
        n = len(segs)
        covered = sum(1 for s in segs if s['best_iou'] > 0.1)
        missed = n - covered
        avg_iou = np.mean([s['best_iou'] for s in segs])
        iou_gt_05 = sum(1 for s in segs if s['best_iou'] > 0.5)
        act_inside = np.mean([s['inside_mean'] for s in segs])
        act_max = np.mean([s['inside_max'] for s in segs])

        print(f'{cat:<16} {n:>4} {covered:>5}/{n:<3} {missed:>4} '
              f'{avg_iou:>7.3f} {iou_gt_05:>5}/{n:<3} '
              f'{act_inside:>10.3f} {act_max:>7.3f}')

        for s in segs:
            if s['best_iou'] <= 0.1:
                missed_segments.append(s)

    # ========== Missed segments detail ==========
    print()
    print('=' * 70)
    print(f'MISSED GT SEGMENTS (IoU <= 0.1): {len(missed_segments)}')
    print('=' * 70)

    # Group missed by category
    missed_by_cat = defaultdict(list)
    for s in missed_segments:
        missed_by_cat[s['category']].append(s)

    print(f'\n{"Category":<16} {"#Miss":>6} {"avgActInside":>13} {"avgActMax":>10} {"avgGTlen":>9}')
    print('-' * 60)
    for cat in sorted(missed_by_cat.keys()):
        segs = missed_by_cat[cat]
        n = len(segs)
        act_in = np.mean([s['inside_mean'] for s in segs])
        act_mx = np.mean([s['inside_max'] for s in segs])
        gt_len = np.mean([s['gt_len'] for s in segs])
        print(f'{cat:<16} {n:>6} {act_in:>13.3f} {act_mx:>10.3f} {gt_len:>9.0f}')

    # Print worst missed (lowest actionness inside GT)
    print(f'\nTop-10 worst missed (lowest inside_mean):')
    missed_segments.sort(key=lambda x: x['inside_mean'])
    for rank, s in enumerate(missed_segments[:10]):
        print(f'  #{rank+1} [{s["category"]}] video={s["video_idx"]} '
              f'GT=[{s["gt_start"]:.0f},{s["gt_end"]:.0f}] len={s["gt_len"]:.0f} '
              f'actInside={s["inside_mean"]:.4f} actMax={s["inside_max"]:.4f} '
              f'videoMax={s["video_max"]:.4f} props={s["n_proposals"]}')

    # ========== Short vs Long segments ==========
    print()
    print('=' * 70)
    print('COVERAGE BY GT SEGMENT LENGTH')
    print('=' * 70)
    lengths = np.array([s['gt_len'] for s in seg_info])
    percentiles = [0, 25, 50, 75, 100]
    bins = np.percentile(lengths, percentiles)
    print(f'GT length distribution: min={lengths.min():.0f} '
          f'p25={np.percentile(lengths,25):.0f} '
          f'median={np.median(lengths):.0f} '
          f'p75={np.percentile(lengths,75):.0f} '
          f'max={lengths.max():.0f}')
    print()
    for lo, hi, label in [(0, np.percentile(lengths, 25), 'short (Q1)'),
                           (np.percentile(lengths, 25), np.median(lengths), 'medium-short (Q2)'),
                           (np.median(lengths), np.percentile(lengths, 75), 'medium-long (Q3)'),
                           (np.percentile(lengths, 75), lengths.max() + 1, 'long (Q4)')]:
        bucket = [s for s in seg_info if lo <= s['gt_len'] < hi]
        if not bucket:
            continue
        n = len(bucket)
        covered = sum(1 for s in bucket if s['best_iou'] > 0.1)
        avg_iou = np.mean([s['best_iou'] for s in bucket])
        iou_05 = sum(1 for s in bucket if s['best_iou'] > 0.5)
        print(f'  {label:<20} n={n:>3}  covered={covered}/{n}  '
              f'avgIoU={avg_iou:.3f}  IoU>0.5={iou_05}/{n}')


if __name__ == '__main__':
    main()
