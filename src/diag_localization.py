"""Localization diagnostic: D1 score dist, D3 peak loc, D4 IoU hist,
D5 coverage quality. Runs on a single checkpoint, dumps JSON.

See docs/superpowers/specs/2026-04-18-localization-diagnostics-design.md
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.detection_map import nms
import option


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _upsample_linear_1d(x, factor=16):
    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    if T == 0:
        return x
    centers = np.arange(T) * factor + (factor - 1) / 2.0
    frame_idx = np.arange(T * factor)
    return np.interp(frame_idx, centers, x)


# ------------------ D1: Score distribution ------------------

def _local_maxima(prob, min_dist=8):
    """Indices of strict local maxima, deduped within min_dist."""
    n = len(prob)
    peaks = []
    for t in range(n):
        ok = True
        for k in range(1, min_dist + 1):
            if t - k >= 0 and prob[t - k] >= prob[t]:
                ok = False
                break
            if t + k < n and prob[t + k] > prob[t]:
                ok = False
                break
        if ok:
            peaks.append(t)
    dedup = []
    for p in peaks:
        if not dedup or p - dedup[-1] >= min_dist:
            dedup.append(p)
    return dedup


def compute_D1(prob):
    peaks = _local_maxima(prob)
    n_abs = sum(1 for p in peaks if prob[p] > 0.5)
    rel_thr = 0.8 * prob.max() if prob.max() > 0 else 1.0
    n_rel = sum(1 for p in peaks if prob[p] > rel_thr)
    p90 = float(np.percentile(prob, 90)) if len(prob) > 0 else 0.0
    n_p90 = sum(1 for p in peaks if prob[p] > p90)
    return {
        'max': float(prob.max()),
        'mean': float(prob.mean()),
        'std': float(prob.std()),
        'median': float(np.median(prob)),
        'n_peaks_abs_0.5': int(n_abs),
        'n_peaks_rel_0.8': int(n_rel),
        'n_peaks_p90': int(n_p90),
        'peak_pos': float(np.argmax(prob) / max(1, len(prob))),
    }


# ------------------ GT helpers ------------------

def _anomaly_gt_list(gt_segs_i, gt_labels_i):
    """List of (start_frame, end_frame) for non-Normal segments."""
    out = []
    for j in range(len(gt_segs_i)):
        if gt_labels_i[j] != 'A':
            out.append((int(gt_segs_i[j][0]), int(gt_segs_i[j][1])))
    return out


def _longest_segment(gt_list):
    return max(gt_list, key=lambda se: se[1] - se[0]) if gt_list else None


def _in_any_segment(frame, gt_list):
    return any(s <= frame < e for s, e in gt_list)


def _in_middle_50(frame, gt_list):
    seg = _longest_segment(gt_list)
    if seg is None:
        return False
    s, e = seg
    q1 = s + 0.25 * (e - s)
    q3 = s + 0.75 * (e - s)
    return q1 <= frame < q3


def _peak_to_gt(frame, gt_list):
    if not gt_list or _in_any_segment(frame, gt_list):
        return 0
    dist = min(min(abs(frame - s), abs(frame - (e - 1))) for s, e in gt_list)
    return int(dist)


# ------------------ D3: Peak location ------------------

def compute_D3(prob, gt_list):
    peak = int(np.argmax(prob))
    return {
        'peak_in_gt': bool(_in_any_segment(peak, gt_list)),
        'peak_in_middle_50pct': bool(_in_middle_50(peak, gt_list)),
        'peak_to_gt_frames': _peak_to_gt(peak, gt_list),
    }


# ------------------ D4: IoU histogram ------------------

def _gen_proposals(prob, rel=0.6):
    """Adaptive threshold + NMS + top_2 (matches inference)."""
    if prob.max() == prob.min():
        return []
    pp = np.sort(prob)[::-1]
    c_s = float(np.mean(pp[:max(1, int(len(pp) / 16))]))
    thr = prob.max() - (prob.max() - prob.min()) * rel
    vid = np.concatenate([np.zeros(1),
                          (prob > thr).astype(np.float32),
                          np.zeros(1)], axis=0)
    diff = vid[1:] - vid[:-1]
    starts = np.where(diff == 1)[0].tolist()
    ends = np.where(diff == -1)[0].tolist()
    per_video = []
    for s, e in zip(starts, ends):
        if e - s >= 2:
            score = float(prob[s:e].max()) + 0.7 * c_s
            per_video.append([s, e, score])
    if not per_video:
        return []
    arr = np.array(per_video)
    arr = arr[np.argsort(-arr[:, -1])]
    _, keep = nms(arr[:, :2], 0.6)
    keep = keep[:2]
    return [(int(arr[k, 0]), int(arr[k, 1])) for k in keep]


def _iou_pair(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def compute_D4(prob, gt_list, video_len):
    props = _gen_proposals(prob)
    gt_frames = sum(e - s for s, e in gt_list)
    gt_frac = gt_frames / max(1, video_len)
    if not props or not gt_list:
        return {'best_iou': 0.0, 'gt_duration_frac': float(gt_frac)}
    best = 0.0
    for p in props:
        for g in gt_list:
            iou = _iou_pair(p, g)
            if iou > best:
                best = iou
    return {'best_iou': float(best), 'gt_duration_frac': float(gt_frac)}


# ------------------ D5: Coverage quality ------------------

def compute_D5(prob, gt_list, video_len, k=8):
    if not gt_list:
        return None
    n = video_len
    gt_mask = np.zeros(n, dtype=bool)
    for s, e in gt_list:
        gt_mask[s:min(e, n)] = True
    inside_idx = np.where(gt_mask)[0]
    outside_idx = np.where(~gt_mask)[0]
    if len(inside_idx) == 0:
        return None

    cov_inside = float(np.mean(prob[inside_idx] > 0.5))
    inside_mean = float(np.mean(prob[inside_idx]))

    window_mask = np.zeros(n, dtype=bool)
    for s, e in gt_list:
        seg_len = e - s
        ws = max(0, s - seg_len)
        we = min(n, e + seg_len)
        window_mask[ws:we] = True
    window_outside = window_mask & ~gt_mask
    if inside_mean < 1e-9 or not np.any(window_outside):
        sp_win = 0.0
    else:
        sp_win = float(np.mean(prob[window_outside]) / inside_mean)

    if inside_mean < 1e-9 or len(outside_idx) == 0:
        sp_gl = 0.0
    else:
        sp_gl = float(np.mean(prob[outside_idx]) / inside_mean)

    grads = []
    for s, e in gt_list:
        if s - k >= 0:
            grads.append(abs(float(prob[s] - prob[s - k])))
        if e - 1 + k < n:
            grads.append(abs(float(prob[e - 1] - prob[min(n - 1, e - 1 + k)])))
    sharpness = float(np.mean(grads)) if grads else 0.0

    if prob.max() == prob.min():
        over_cov = 0.0
    else:
        thr = prob.max() - (prob.max() - prob.min()) * 0.6
        pred_len = int(np.sum(prob > thr))
        gt_len = int(np.sum(gt_mask))
        over_cov = float(pred_len / max(1, gt_len))

    return {
        'coverage_inside': cov_inside,
        'spillover_ratio_window': sp_win,
        'spillover_ratio_global': sp_gl,
        'boundary_sharpness': sharpness,
        'over_coverage_ratio': over_cov,
    }


# ------------------ Aggregation ------------------

def _median_safe(xs):
    return float(np.median(xs)) if len(xs) else 0.0


def _mean_safe(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def _aggregate(per_video_dicts, keys):
    """Transpose list-of-dicts into dict-of-lists, dropping None entries."""
    return {k: [d[k] for d in per_video_dicts if d is not None] for k in keys}


def _iou_histogram(best_ious, bins=(0, 0.1, 0.3, 0.5, 0.7, 1.0 + 1e-9)):
    counts, _ = np.histogram(best_ious, bins=bins)
    return counts.tolist()


# ------------------ Main inference + diag loop ------------------

def run_diagnostics(model, testdataloader, maxlen, prompt_text,
                    gt_segments, gt_labels, device):
    model.to(device).eval()

    D1_all = {'normal': [], 'anomaly': []}
    D3_all = []         # per anomaly video
    D4_all = []         # per anomaly video
    D5_all = []         # per anomaly video

    with torch.no_grad():
        for idx, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            remaining = length
            for j in range(int(length / maxlen) + 1):
                if j == 0 and remaining < maxlen:
                    lengths[j] = remaining
                elif j == 0 and remaining > maxlen:
                    lengths[j] = maxlen
                    remaining -= maxlen
                elif remaining > maxlen:
                    lengths[j] = maxlen
                    remaining -= maxlen
                else:
                    lengths[j] = remaining
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            _, logits1, _, _, _ = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1],
                                      logits1.shape[2])
            prob_snippet = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()
            prob = _upsample_linear_1d(prob_snippet, factor=16)

            gt_list = _anomaly_gt_list(gt_segments[idx], gt_labels[idx])
            is_anom = len(gt_list) > 0

            d1 = compute_D1(prob)
            if is_anom:
                D1_all['anomaly'].append(d1)
                D3_all.append(compute_D3(prob, gt_list))
                D4_all.append(compute_D4(prob, gt_list, len(prob)))
                D5_all.append(compute_D5(prob, gt_list, len(prob)))
            else:
                D1_all['normal'].append(d1)

            if (idx + 1) % 20 == 0:
                print(f'  [{idx + 1} videos done]')

    D1_keys = ['max', 'mean', 'std', 'median',
               'n_peaks_abs_0.5', 'n_peaks_rel_0.8', 'n_peaks_p90', 'peak_pos']
    D1_out = {
        'normal':  _aggregate(D1_all['normal'], D1_keys),
        'anomaly': _aggregate(D1_all['anomaly'], D1_keys),
    }

    D3_keys = ['peak_in_gt', 'peak_in_middle_50pct', 'peak_to_gt_frames']
    D3_raw = _aggregate(D3_all, D3_keys)
    D3_out = {
        **D3_raw,
        'peak_in_gt_ratio':     _mean_safe([float(x) for x in D3_raw['peak_in_gt']]),
        'peak_in_middle_ratio': _mean_safe([float(x) for x in D3_raw['peak_in_middle_50pct']]),
        'mean_peak_to_gt':      _mean_safe(D3_raw['peak_to_gt_frames']),
    }

    best_ious = [d['best_iou'] for d in D4_all]
    gt_fracs = [d['gt_duration_frac'] for d in D4_all]
    short = [b for b, f in zip(best_ious, gt_fracs) if f < 0.20]
    long_ = [b for b, f in zip(best_ious, gt_fracs) if f >= 0.20]
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0 + 1e-9]
    D4_out = {
        'best_iou_per_video': best_ious,
        'gt_duration_frac':   gt_fracs,
        'iou_hist_bins':      [0, 0.1, 0.3, 0.5, 0.7, 1.0],
        'iou_hist_all':       _iou_histogram(best_ious, bins),
        'iou_hist_short':     _iou_histogram(short, bins),
        'iou_hist_long':      _iou_histogram(long_, bins),
        'median_iou':         _median_safe(best_ious),
        'pct_iou_ge_0.5':     _mean_safe([1.0 if b >= 0.5 else 0.0 for b in best_ious]),
        'pct_iou_lt_0.1':     _mean_safe([1.0 if b < 0.1 else 0.0 for b in best_ious]),
    }

    D5_keys = ['coverage_inside', 'spillover_ratio_window', 'spillover_ratio_global',
               'boundary_sharpness', 'over_coverage_ratio']
    D5_raw = _aggregate(D5_all, D5_keys)
    summary = {k: {'mean': _mean_safe(D5_raw[k]), 'median': _median_safe(D5_raw[k])}
               for k in D5_keys}
    # Stratified medians by short/long GT
    short_idx = [i for i, f in enumerate(gt_fracs) if f < 0.20]
    long_idx = [i for i, f in enumerate(gt_fracs) if f >= 0.20]

    def _strat(idx_list):
        if not idx_list:
            return {k + '_median': 0.0 for k in
                    ['coverage_inside', 'spillover_window',
                     'boundary_sharp', 'over_cov']}
        return {
            'coverage_inside_median':  _median_safe([D5_raw['coverage_inside'][i] for i in idx_list]),
            'spillover_window_median': _median_safe([D5_raw['spillover_ratio_window'][i] for i in idx_list]),
            'boundary_sharp_median':   _median_safe([D5_raw['boundary_sharpness'][i] for i in idx_list]),
            'over_cov_median':         _median_safe([D5_raw['over_coverage_ratio'][i] for i in idx_list]),
        }

    D5_out = {
        **D5_raw,
        'summary':    summary,
        'stratified': {'short': _strat(short_idx), 'long': _strat(long_idx)},
    }

    return {
        'n_videos': {'normal':  len(D1_all['normal']),
                     'anomaly': len(D1_all['anomaly'])},
        'D1': D1_out,
        'D3': D3_out,
        'D4': D4_out,
        'D5': D5_out,
    }


def main():
    parser = option.parser
    parser.add_argument('--out-json', required=True,
                        help='Output JSON path (e.g. docs/diag/exp12.json)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = args.device if args.device != 'cuda' or torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    print(f'Checkpoint:   {args.model_path}')
    print(f'Test list:    {args.test_list}')

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt_segments = np.load(args.gt_segment_path, allow_pickle=True)
    gt_labels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device)

    state = torch.load(args.model_path, weights_only=False, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys (first 5): {missing[:5]}')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}')

    result = run_diagnostics(model, testdataloader, args.visual_length,
                             prompt_text, gt_segments, gt_labels, device)
    result['model_path'] = args.model_path

    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {args.out_json}')
    print(f"  peak_in_gt_ratio:        {result['D3']['peak_in_gt_ratio']:.3f}")
    print(f"  peak_in_middle_ratio:    {result['D3']['peak_in_middle_ratio']:.3f}")
    print(f"  coverage_inside median:  {result['D5']['summary']['coverage_inside']['median']:.3f}")
    print(f"  spillover_window median: {result['D5']['summary']['spillover_ratio_window']['median']:.3f}")
    print(f"  boundary_sharp  median:  {result['D5']['summary']['boundary_sharpness']['median']:.3f}")
    print(f"  over_coverage   median:  {result['D5']['summary']['over_coverage_ratio']['median']:.3f}")
    print(f"  median IoU:              {result['D4']['median_iou']:.3f}")


if __name__ == '__main__':
    main()
