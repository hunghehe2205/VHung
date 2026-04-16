"""Diagnostic analysis for trained CLIPVAD models.

Runs 3 analyses:
1. Abnormal-only mAP (exclude Normal videos entirely)
2. Proposal quality on abnormal videos (IoU distribution, Recall@K)
3. Actionness profile on abnormal videos (precision of spike location)
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


def _upsample_repeat(v):
    return np.repeat(v, 16)


def _threshold_proposals(prob, thr_ratio=0.6):
    """Generate proposals using adaptive threshold. Returns list of (start, end, score)."""
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


def _compute_iou(pred_start, pred_end, gt_start, gt_end):
    inter_s = max(pred_start, gt_start)
    inter_e = min(pred_end, gt_end)
    inter = max(0, inter_e - inter_s)
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _loc_map_abnormal_only(predictions, gtsegments, gtlabels, th):
    """mAP computed only on abnormal videos."""
    segment_predict = []
    for i in range(len(predictions)):
        # Skip normal videos
        if not any(l != 'A' for l in gtlabels[i]):
            continue
        tmp = predictions[i]
        if tmp.max() == tmp.min():
            continue
        threshold = tmp.max() - (tmp.max() - tmp.min()) * 0.6
        c_s = float(np.mean(np.sort(tmp)[::-1][:max(1, len(tmp) // 16)]))
        vid_pred = np.concatenate([np.zeros(1), (tmp > threshold).astype('float32'), np.zeros(1)])
        vid_pred_diff = [vid_pred[t] - vid_pred[t - 1] for t in range(1, len(vid_pred))]
        s = [k for k, item in enumerate(vid_pred_diff) if item == 1]
        e = [k for k, item in enumerate(vid_pred_diff) if item == -1]
        for j in range(len(s)):
            if e[j] - s[j] >= 2:
                score = float(np.max(tmp[s[j]:e[j]])) + 0.7 * c_s
                segment_predict.append([i, s[j], e[j], score])

    if not segment_predict:
        return 0.0
    segment_predict = np.array(segment_predict)
    segment_predict = segment_predict[np.argsort(-segment_predict[:, 3])]

    segment_gt = [[i, gtsegments[i][j][0], gtsegments[i][j][1]]
                  for i in range(len(gtsegments))
                  for j in range(len(gtsegments[i]))
                  if gtlabels[i][j] != 'A']
    gtpos = len(segment_gt)
    if gtpos == 0:
        return 0.0

    tp, fp = [], []
    for i in range(len(segment_predict)):
        flag = 0.0
        best_iou = 0.0
        best_j = -1
        for j in range(len(segment_gt)):
            if segment_predict[i][0] == segment_gt[j][0]:
                gt = range(int(segment_gt[j][1]), int(segment_gt[j][2]))
                p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                inter = len(set(gt).intersection(set(p)))
                union = len(set(gt).union(set(p)))
                if union == 0:
                    continue
                IoU = float(inter) / float(union)
                if IoU >= th and IoU > best_iou:
                    flag = 1.0
                    best_iou = IoU
                    best_j = j
        if flag > 0 and best_j >= 0:
            del segment_gt[best_j]
        tp.append(flag)
        fp.append(1.0 - flag)
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    if sum(tp) == 0:
        return 0.0
    prc = np.sum((tp_c / (fp_c + tp_c)) * np.array(tp)) / gtpos
    return 100.0 * prc


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

    # Collect per-video data
    all_probs = []  # frame-level probs per video
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
            prob_frame = _upsample_repeat(prob1)
            all_probs.append(prob_frame)

    # ========== 1. Abnormal-only mAP ==========
    print('=' * 60)
    print('1. ABNORMAL-ONLY mAP')
    print('=' * 60)

    iou_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    dmap_abn = [_loc_map_abnormal_only(all_probs, gtsegments, gtlabels, iou)
                for iou in iou_list]
    avg_abn = float(np.mean(dmap_abn))
    abn_str = '/'.join(f'{v:.2f}' for v in dmap_abn)
    print(f'  Abnormal-only mAP: AVG={avg_abn:.2f} [{abn_str}]')

    # Also compute all-videos mAP for comparison
    from utils.detection_map import getDetectionMAP_agnostic
    dmap_all, _ = getDetectionMAP_agnostic(all_probs, gtsegments, gtlabels)
    avg_all = float(np.mean(dmap_all))
    all_str = '/'.join(f'{v:.2f}' for v in dmap_all)
    print(f'  All-videos mAP:    AVG={avg_all:.2f} [{all_str}]')
    print(f'  Delta (abn-all):   {avg_abn - avg_all:+.2f}')

    # ========== 2. Proposal Quality on Abnormal Videos ==========
    print()
    print('=' * 60)
    print('2. PROPOSAL QUALITY ON ABNORMAL VIDEOS')
    print('=' * 60)

    all_ious = []  # all proposal-GT IoU values
    recall_at_k = {1: 0, 3: 0, 5: 0, 10: 0}
    n_anomaly = 0
    n_gt_total = 0
    proposals_per_video = []

    for i in range(len(all_probs)):
        is_normal = str(labels_csv[i]).lower() == 'normal'
        if is_normal:
            continue
        n_anomaly += 1

        gt_segs = [(float(gtsegments[i][j][0]), float(gtsegments[i][j][1]))
                    for j in range(len(gtsegments[i]))
                    if gtlabels[i][j] != 'A']
        n_gt_total += len(gt_segs)

        proposals = _threshold_proposals(all_probs[i])
        proposals_per_video.append(len(proposals))

        # Compute IoU for each proposal against best-matching GT
        for ps, pe, _ in proposals:
            best_iou = 0.0
            for gs, ge in gt_segs:
                iou = _compute_iou(ps, pe, gs, ge)
                best_iou = max(best_iou, iou)
            all_ious.append(best_iou)

        # Recall@K: fraction of GT segments covered by top-K proposals
        for K in recall_at_k:
            top_k = proposals[:K]
            covered = 0
            for gs, ge in gt_segs:
                for ps, pe, _ in top_k:
                    if _compute_iou(ps, pe, gs, ge) >= 0.1:
                        covered += 1
                        break
            recall_at_k[K] += covered

    all_ious = np.array(all_ious)
    print(f'  Anomaly videos: {n_anomaly}, GT segments: {n_gt_total}')
    print(f'  Proposals/video: mean={np.mean(proposals_per_video):.1f} '
          f'median={np.median(proposals_per_video):.1f} '
          f'min={np.min(proposals_per_video)} max={np.max(proposals_per_video)}')
    print()

    if len(all_ious) > 0:
        print(f'  IoU distribution (all proposals vs best GT):')
        print(f'    mean={all_ious.mean():.3f}  median={np.median(all_ious):.3f}')
        for thr in [0.0, 0.1, 0.3, 0.5, 0.7]:
            pct = (all_ious > thr).mean() * 100
            print(f'    IoU > {thr:.1f}: {pct:.1f}% ({(all_ious > thr).sum()}/{len(all_ious)})')
        print()

    print(f'  Recall@K (IoU>0.1, over {n_gt_total} GT segments):')
    for K in sorted(recall_at_k):
        r = recall_at_k[K] / max(1, n_gt_total) * 100
        print(f'    Recall@{K}: {r:.1f}% ({recall_at_k[K]}/{n_gt_total})')

    # ========== 3. Actionness Profile on Abnormal Videos ==========
    print()
    print('=' * 60)
    print('3. ACTIONNESS PROFILE ON ABNORMAL VIDEOS')
    print('=' * 60)

    inside_means = []
    outside_means = []
    inside_maxs = []
    precision_at_05 = []  # fraction of frames with prob>0.5 that are inside GT

    for i in range(len(all_probs)):
        is_normal = str(labels_csv[i]).lower() == 'normal'
        if is_normal:
            continue

        prob = all_probs[i]
        n_frames = len(prob)

        # Build GT mask
        gt_mask = np.zeros(n_frames, dtype=bool)
        for j in range(len(gtsegments[i])):
            if gtlabels[i][j] == 'A':
                continue
            s = max(0, int(gtsegments[i][j][0]))
            e = min(n_frames, int(gtsegments[i][j][1]))
            gt_mask[s:e] = True

        if not gt_mask.any():
            continue

        inside = prob[gt_mask]
        outside = prob[~gt_mask]

        inside_means.append(float(inside.mean()))
        inside_maxs.append(float(inside.max()))
        if len(outside) > 0:
            outside_means.append(float(outside.mean()))

        # Temporal precision: of frames with prob > 0.5, how many are inside GT?
        high_prob = prob > 0.5
        if high_prob.any():
            tp_frames = (high_prob & gt_mask).sum()
            precision_at_05.append(float(tp_frames) / float(high_prob.sum()))

    print(f'  Actionness inside GT:')
    print(f'    mean: {np.mean(inside_means):.4f} (std={np.std(inside_means):.4f})')
    print(f'    max:  {np.mean(inside_maxs):.4f}')
    print()
    print(f'  Actionness outside GT:')
    print(f'    mean: {np.mean(outside_means):.4f} (std={np.std(outside_means):.4f})')
    print()
    gap = np.mean(inside_means) - np.mean(outside_means)
    print(f'  Gap (inside - outside): {gap:.4f}')
    print()
    if precision_at_05:
        print(f'  Temporal precision (prob>0.5):')
        print(f'    mean: {np.mean(precision_at_05):.3f}  median: {np.median(precision_at_05):.3f}')
        print(f'    (fraction of high-prob frames that are inside GT)')
        n_low = sum(1 for p in precision_at_05 if p < 0.5)
        print(f'    videos with precision < 0.5: {n_low}/{len(precision_at_05)}')
    else:
        print(f'  No videos with prob > 0.5')


if __name__ == '__main__':
    main()
