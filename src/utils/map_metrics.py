"""Map-quality metrics for logits3 anomaly distribution.

Designed to evaluate whether s_t ∈ [0,1] is suitable for the
CDF-cumsum sampler (see inference_viz/sampling.py).

All functions operate on NumPy arrays (CPU) and are side-effect free.
"""
from __future__ import annotations

import math
import numpy as np


def separation_stats(scores: np.ndarray, mask: np.ndarray) -> dict:
    """Gap and per-side means.

    Args:
        scores: (T,) float array, s_t ∈ [0,1].
        mask:   (T,) float/bool array, 1 inside events.

    Returns:
        dict with keys: in_mean, out_mean, gap.
        If no in-event frames exist (normal video), in_mean is NaN and gap is NaN.
        If no out-of-event frames exist (fully anomalous), out_mean is NaN.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    in_frames = scores[mask]
    out_frames = scores[~mask]
    in_mean = float(in_frames.mean()) if in_frames.size > 0 else float('nan')
    out_mean = float(out_frames.mean()) if out_frames.size > 0 else float('nan')
    gap = in_mean - out_mean
    return {'in_mean': in_mean, 'out_mean': out_mean, 'gap': gap}


def mass_stats(scores: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> dict:
    """Event-Mass Ratio (EMR), Event-Time Ratio (ETR), and Mass Concentration Lift (MCL).

    EMR = Σ s_t(in-event) / Σ s_t(all)
    ETR = |in-event| / |all|
    MCL = EMR / ETR  (NaN if no events)
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    total_mass = scores.sum()
    in_mass = scores[mask].sum()
    emr = float(in_mass / (total_mass + eps)) if total_mass > 0 else 0.0
    etr = float(mask.sum() / mask.size) if mask.size > 0 else 0.0
    mcl = float(emr / etr) if etr > 0 else float('nan')
    return {'emr': emr, 'etr': etr, 'mcl': mcl, 'mass_lift': emr - etr}


def density_stats(scores: np.ndarray, mask: np.ndarray,
                  cov_thresholds: tuple = (0.3, 0.5),
                  top_fraction: float = 0.1,
                  eps: float = 1e-8) -> dict:
    """Metrics for in-event density and anti-spike.

    in_event_cov@τ: fraction of event frames with s_t > τ.
    peak_concentration: mass of top X% event frames / total event mass.
    in_event_entropy: normalized entropy H(p̂) / log(L_event), where
        p̂[t] = s_t / Σ_event s_t.  1 = uniform (dense), 0 = single spike.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    event_scores = scores[mask]
    L = event_scores.size
    out = {}
    for tau in cov_thresholds:
        key = f'in_event_cov_{int(round(tau * 10)):02d}'
        out[key] = float((event_scores > tau).mean()) if L > 0 else float('nan')

    if L == 0:
        out['peak_concentration'] = float('nan')
        out['in_event_entropy'] = float('nan')
        return out

    total_event_mass = event_scores.sum()
    if total_event_mass <= eps:
        out['peak_concentration'] = float('nan')
        out['in_event_entropy'] = 0.0
        return out

    k = max(1, int(math.ceil(L * top_fraction)))
    topk = np.sort(event_scores)[-k:]
    out['peak_concentration'] = float(topk.sum() / total_event_mass)

    p = event_scores / total_event_mass
    p_safe = np.clip(p, eps, 1.0)
    H = float(-(p * np.log(p_safe)).sum())
    out['in_event_entropy'] = float(H / math.log(L)) if L > 1 else 0.0
    return out


from .detection_map import nms as _nms  # reuse existing NMS


def _extract_segments(score: np.ndarray, thresholds=(0.3, 0.5, 0.7), min_length: int = 2):
    """Threshold a 1-D score array at several levels; produce candidate segments.

    Returns list of [start, end, score].
    """
    segments = []
    for thr in thresholds:
        mask = (score > thr).astype(np.int8)
        padded = np.concatenate([[0], mask, [0]])
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if e - s >= min_length:
                seg_score = float(score[s:e].max())
                segments.append([int(s), int(e), seg_score])
    return segments


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return 0.0
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    precision = tp_c / (tp_c + fp_c + 1e-12)
    # VOC-style: Σ (precision_at_tp) / n_gt
    ap = float((precision * tp).sum() / n_gt)
    return ap


def binary_detection_map(predictions: list,
                         gt_segments: list,
                         iou_thresholds: tuple = (0.1, 0.2, 0.3, 0.4, 0.5),
                         nms_thresh: float = 0.6) -> dict:
    """Class-agnostic temporal detection mAP over a dataset of videos.

    Args:
        predictions: list length V of 1-D numpy arrays (frame-level or snippet-level scores).
        gt_segments: list length V. gt_segments[i] is iterable of [start, end] pairs
                     (same frame domain as predictions[i]). Empty list means no events.
        iou_thresholds: IoU levels at which to compute AP.
        nms_thresh: IoU threshold for NMS between multi-threshold candidates.

    Returns:
        dict with 'map_at_iou_{XX}' for each threshold plus 'map_avg'.
    """
    # Collect all candidate segments across videos.
    all_preds = []  # [video_idx, start, end, score]
    for i, score in enumerate(predictions):
        score = np.asarray(score, dtype=np.float64)
        segs = _extract_segments(score)
        if len(segs) == 0:
            continue
        segs_arr = np.array(segs)
        segs_arr = segs_arr[np.argsort(-segs_arr[:, -1])]
        _, keep = _nms(segs_arr[:, :2], thresh=nms_thresh)
        for k in keep:
            all_preds.append([i, int(segs_arr[k, 0]), int(segs_arr[k, 1]), float(segs_arr[k, 2])])

    n_gt_total = sum(len(g) for g in gt_segments)
    result = {}
    if n_gt_total == 0:
        for thr in iou_thresholds:
            result[f'map_at_iou_{int(thr*10):02d}'] = float('nan')
        result['map_avg'] = float('nan')
        return result

    if len(all_preds) == 0:
        for thr in iou_thresholds:
            result[f'map_at_iou_{int(thr*10):02d}'] = 0.0
        result['map_avg'] = 0.0
        return result

    all_preds.sort(key=lambda x: -x[3])

    aps = []
    for iou_thr in iou_thresholds:
        # Track matched GT per video
        remaining_gt = [list(map(list, g)) for g in gt_segments]
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))
        for idx, (vi, ps, pe, _score) in enumerate(all_preds):
            best_iou = 0.0
            best_gi = -1
            for gi, (gs, ge) in enumerate(remaining_gt[vi]):
                inter = max(0, min(pe, ge) - max(ps, gs))
                union = max(pe, ge) - min(ps, gs)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thr and best_gi >= 0:
                tp[idx] = 1.0
                remaining_gt[vi].pop(best_gi)
            else:
                fp[idx] = 1.0
        ap = _average_precision(tp, fp, n_gt_total)
        aps.append(ap)
        result[f'map_at_iou_{int(iou_thr*10):02d}'] = ap

    result['map_avg'] = float(np.mean(aps))
    return result
