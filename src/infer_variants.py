"""Inference-only ablation on proposal generation — no retrain needed.

Runs Exp 12 model once, caches per-video probs, then tries multiple proposal
strategies and reports abnormal-only mAP for each.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.detection_map import _iou_matching_ap, nms
import option


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _get_probs(model, loader, maxlen, prompt_text, device):
    probs = []
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
            probs.append(torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy())
    return probs


# --------- Proposal generators (snippet-resolution, upsample ×16 for IoU) ---------

def _runs_above(prob, threshold, min_len=2, clip_len=16):
    """Contiguous runs where prob > threshold. Returns [(s_frame, e_frame, max_prob)]."""
    if prob.max() <= threshold:
        return []
    mask = np.concatenate([[0.0], (prob > threshold).astype(np.float32), [0.0]])
    diff = mask[1:] - mask[:-1]
    s_snip = np.where(diff == 1)[0].tolist()
    e_snip = np.where(diff == -1)[0].tolist()
    out = []
    for s, e in zip(s_snip, e_snip):
        if e - s >= min_len:
            out.append((s * clip_len, e * clip_len, float(prob[s:e].max())))
    return out


def proposals_adaptive(prob, thr_ratio=0.6, clip_len=16):
    """Baseline: threshold = max - thr_ratio*(max-min)."""
    if prob.max() == prob.min():
        return []
    thr = prob.max() - (prob.max() - prob.min()) * thr_ratio
    return _runs_above(prob, thr, min_len=2, clip_len=clip_len)


def proposals_absolute(prob, abs_thr=0.5, clip_len=16):
    """A2: absolute threshold. Empty if max < abs_thr."""
    if prob.max() < abs_thr:
        return []
    return _runs_above(prob, abs_thr, min_len=2, clip_len=clip_len)


def proposals_hybrid(prob, thr_ratio=0.6, abs_floor=0.3, clip_len=16):
    """A1+A2: threshold = max(adaptive, abs_floor). Empty if max < abs_floor."""
    if prob.max() < abs_floor:
        return []
    if prob.max() == prob.min():
        return []
    adaptive = prob.max() - (prob.max() - prob.min()) * thr_ratio
    thr = max(adaptive, abs_floor)
    return _runs_above(prob, thr, min_len=2, clip_len=clip_len)


def proposals_peak_based(prob, abs_floor=0.3, prominence=0.2, clip_len=16):
    """A1: find local peaks above abs_floor, expand to half-height on each side.
    A peak at frame i expands left/right while prob >= max(abs_floor, peak*0.5).
    NMS via wider half-height extents removed by proposals_nms in caller.
    """
    if prob.max() < abs_floor:
        return []
    n = len(prob)
    # Find all snippets that are local maxima (strictly > neighbors)
    peaks = []
    for t in range(n):
        if prob[t] < abs_floor:
            continue
        left = prob[t - 1] if t > 0 else -1
        right = prob[t + 1] if t + 1 < n else -1
        if prob[t] >= left and prob[t] >= right:
            peaks.append(t)
    out = []
    for pk in peaks:
        half = max(abs_floor, prob[pk] * 0.5)
        s = pk
        while s > 0 and prob[s - 1] >= half:
            s -= 1
        e = pk + 1
        while e < n and prob[e] >= half:
            e += 1
        if e - s >= 2:
            out.append((s * clip_len, e * clip_len, float(prob[pk])))
    return out


def _nms_score(props, iou_thr=0.5):
    """Standard NMS on (start, end, score). Returns list of kept tuples."""
    if not props:
        return []
    arr = np.array(props)
    arr = arr[np.argsort(-arr[:, 2])]
    _, keep = nms(arr[:, :2], iou_thr)
    return [tuple(arr[k]) for k in keep]


def build_segments(probs, proposal_fn, use_nms=True, top_k=None):
    """Applies proposal_fn per video, adds c_score bonus, returns flat list of
    [video_idx, s, e, score] for mAP computation.
    """
    out = []
    for i, p in enumerate(probs):
        pp = np.sort(p)[::-1]
        c_s = float(np.mean(pp[:max(1, int(len(pp) / 16))]))
        props = proposal_fn(p)
        if not props:
            continue
        if use_nms:
            props = _nms_score(props, 0.6)
        # Re-score with c_s bonus (match existing mAP convention)
        scored = [(s, e, sc + 0.7 * c_s) for (s, e, sc) in props]
        scored.sort(key=lambda x: -x[2])
        if top_k is not None:
            scored = scored[:top_k]
        for s, e, sc in scored:
            out.append([i, s, e, sc])
    return out


def filter_abnormal(segment_predict, gtlabels):
    """Keep only proposals from videos with at least one non-Normal GT label."""
    return [sp for sp in segment_predict
            if any(l != 'A' for l in gtlabels[int(sp[0])])]


def eval_strategy(name, probs, proposal_fn, gtsegments, gtlabels,
                  use_nms=True, top_k=None):
    segs = build_segments(probs, proposal_fn, use_nms=use_nms, top_k=top_k)
    segs_abn = filter_abnormal(segs, gtlabels)
    ious = [0.1, 0.2, 0.3, 0.4, 0.5]
    aps = [_iou_matching_ap(segs_abn, gtsegments, gtlabels, iou) for iou in ious]
    avg = float(np.mean(aps))
    per = '/'.join(f'{v:5.2f}' for v in aps)
    n_props = len(segs_abn)
    print(f'  {name:44s}  AVG={avg:5.2f}  [{per}]  n_props={n_props}')
    return avg, aps


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testloader = DataLoader(testdataset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(LABEL_MAP)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    cache_path = args.model_path.replace('.pth', '.probs.npy')
    if os.path.exists(cache_path):
        print(f'[loading cached probs from {cache_path}]')
        probs = list(np.load(cache_path, allow_pickle=True))
    else:
        print(f'[running model inference, will cache to {cache_path}]')
        model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                        args.visual_width, args.visual_head, args.visual_layers,
                        args.attn_window, args.prompt_prefix, args.prompt_postfix,
                        device)
        model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                         map_location=device), strict=False)
        model.to(device).eval()
        probs = _get_probs(model, testloader, args.visual_length, prompt_text, device)
        np.save(cache_path, np.array(probs, dtype=object), allow_pickle=True)

    print('=' * 82)
    print(f'Abnormal-only mAP — proposal strategy ablation on {args.model_path}')
    print('=' * 82)

    # Baseline (replicate existing)
    print('\n[baseline — current adaptive thresh max-0.6*(max-min), NMS 0.6]')
    eval_strategy('adaptive_0.6 (baseline)',
                  probs, lambda p: proposals_adaptive(p, 0.6),
                  gtsegments, gtlabels)

    print('\n[A2 — pure absolute threshold]')
    for a in [0.3, 0.4, 0.5, 0.6, 0.7]:
        eval_strategy(f'abs_{a}',
                      probs, lambda p, a=a: proposals_absolute(p, a),
                      gtsegments, gtlabels)

    print('\n[A1+A2 — hybrid: max(adaptive, abs_floor), skip video if max < floor]')
    for af in [0.2, 0.3, 0.4, 0.5]:
        for tr in [0.5, 0.6, 0.7]:
            eval_strategy(f'hybrid thr_ratio={tr} abs_floor={af}',
                          probs,
                          lambda p, tr=tr, af=af: proposals_hybrid(p, tr, af),
                          gtsegments, gtlabels)

    print('\n[A1 — peak-based, expand to half-height, NMS 0.6]')
    for af in [0.3, 0.5]:
        eval_strategy(f'peak_half abs_floor={af}',
                      probs,
                      lambda p, af=af: proposals_peak_based(p, af),
                      gtsegments, gtlabels)

    print('\n[top-K cap (keep best-scoring K proposals per video)]')
    for k in [1, 2, 3, 5]:
        eval_strategy(f'adaptive_0.6 + top_{k}',
                      probs, lambda p: proposals_adaptive(p, 0.6),
                      gtsegments, gtlabels, top_k=k)

    print('=' * 82)


if __name__ == '__main__':
    main()
