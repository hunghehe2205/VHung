import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.detection_map import (getDetectionMAP_abnormal_only,
                                 getDetectionMAP_agnostic)
from utils.tools import get_batch_mask, get_prompt_text
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _upsample(x, factor=16):
    return np.repeat(np.asarray(x, dtype=np.float32), factor)


def _frame_metrics(scores, gt, thr=0.5):
    pred = (scores >= thr).astype(np.int32)
    gt_i = gt.astype(np.int32)
    tp = int(((pred == 1) & (gt_i == 1)).sum())
    fp = int(((pred == 1) & (gt_i == 0)).sum())
    fn = int(((pred == 0) & (gt_i == 1)).sum())
    tn = int(((pred == 0) & (gt_i == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    FPR = fp / max(fp + tn, 1)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=P, R=R, F1=F1, FPR=FPR)


def _boundary_sharpness(prob, inside_mask, k_frame=48):
    """Per-video boundary sharpness. Max |Δprob| within a ±k_frame window
    around any GT boundary — captures the sharpest single-step transition.
    Using mean would dilute a real step jump by 95 smooth frames in a
    96-frame window (dev_reframe spec: MAX, not mean)."""
    gt_diff = np.abs(np.diff(inside_mask.astype(np.float32), prepend=0))
    edges = np.where(gt_diff > 0.5)[0]
    if len(edges) == 0:
        return 0.0
    diff = np.abs(np.diff(prob, prepend=prob[0]))
    mask = np.zeros_like(prob, dtype=bool)
    for ed in edges:
        lo, hi = max(0, ed - k_frame), min(len(prob), ed + k_frame)
        mask[lo:hi] = True
    return float(diff[mask].max()) if mask.any() else 0.0


def _compute_diag(primary, gt_full, gtsegments, gtlabels):
    """Produces sep/cov/bsh/norm stats on the primary per-video scores."""
    per_frame_lens = [len(_upsample(p)) for p in primary]
    offsets = np.cumsum([0] + per_frame_lens)

    n_anom = n_norm = 0
    inside_sum = outside_sum = 0.0
    peak_in_gt = peak_total = 0
    over_cov = []
    cov_inside = []
    spillover = []
    bsh = []
    norm_max_med_list = []
    norm_max_mean_list = []
    norm_hot = 0
    norm_frac_hi_count = 0
    norm_total_frames = 0
    pos_count_all = 0
    total_frames_all = 0

    for i, p_snip in enumerate(primary):
        p_frame = _upsample(p_snip)
        gt_slice = gt_full[offsets[i]:offsets[i] + len(p_frame)]
        total_frames_all += len(p_frame)
        pos_count_all += int((p_frame > 0.5).sum())
        is_anom = any(l != 'A' for l in gtlabels[i])

        if is_anom:
            n_anom += 1
            inside_mask = gt_slice > 0.5
            outside_mask = ~inside_mask
            if inside_mask.sum() > 0:
                inside_sum += float(p_frame[inside_mask].mean())
                outside_sum += (float(p_frame[outside_mask].mean())
                                if outside_mask.sum() > 0 else 0.0)
                pk = int(np.argmax(p_frame))
                peak_in_gt += int(inside_mask[pk])
                peak_total += 1
                thr = 0.5 * p_frame.max() if p_frame.max() > 0 else 1.0
                high_mask = p_frame >= thr
                cov_inside.append(float((high_mask & inside_mask).sum()
                                         / max(int(inside_mask.sum()), 1)))
                spillover.append(float((high_mask & outside_mask).sum()
                                        / max(int(high_mask.sum()), 1)))
                over_cov.append(float(int(high_mask.sum())
                                       / max(int(inside_mask.sum()), 1)))
                bsh.append(_boundary_sharpness(p_frame, inside_mask))
        else:
            n_norm += 1
            norm_max_mean_list.append(float(p_frame.mean()))
            norm_max_med_list.append(float(np.median(p_frame)))
            norm_hot += int(p_frame.max() > 0.9)
            norm_frac_hi_count += int((p_frame > 0.5).sum())
            norm_total_frames += len(p_frame)

    inside_mean = inside_sum / max(n_anom, 1)
    outside_mean = outside_sum / max(n_anom, 1)
    delta = inside_mean - outside_mean

    return dict(
        n_anom=n_anom, n_norm=n_norm,
        inside_mean=inside_mean, outside_mean=outside_mean, delta=delta,
        peak_in_gt=(peak_in_gt / max(peak_total, 1)),
        over_cov_mean=float(np.mean(over_cov)) if over_cov else 0.0,
        over_cov_med=float(np.median(over_cov)) if over_cov else 0.0,
        cov_inside=float(np.mean(cov_inside)) if cov_inside else 0.0,
        spillover=float(np.mean(spillover)) if spillover else 0.0,
        bsh_mean=float(np.mean(bsh)) if bsh else 0.0,
        bsh_med=float(np.median(bsh)) if bsh else 0.0,
        norm_max_med=float(np.median(norm_max_med_list)) if norm_max_med_list else 0.0,
        norm_max_mean=float(np.mean(norm_max_mean_list)) if norm_max_mean_list else 0.0,
        n_norm_hot=norm_hot,
        norm_frac_hi=(norm_frac_hi_count / max(norm_total_frames, 1)),
        frac_tcn_high=(pos_count_all / max(total_frames_all, 1)),
    )


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         device, quiet=False, return_diag=False, eval_head='tcn'):
    """Eval. Primary localization = sigmoid(tcn_logits) when eval_head='tcn'."""
    model.to(device)
    model.eval()

    tcn_per_video = []
    wsv_per_video = []

    with torch.no_grad():
        iterator = testdataloader if quiet else tqdm(
            testdataloader, desc='Testing', disable=not sys.stderr.isatty())
        for i, item in enumerate(iterator):
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

            _, logits1, _, tcn_logits = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[-1])
            tcn_logits = tcn_logits.reshape(-1, tcn_logits.shape[-1])

            tcn_per_video.append(
                torch.sigmoid(tcn_logits[0:len_cur, 0]).cpu().numpy())
            wsv_per_video.append(
                torch.sigmoid(logits1[0:len_cur, 0]).cpu().numpy())

    primary = tcn_per_video if eval_head == 'tcn' else wsv_per_video
    primary_frame = np.concatenate([_upsample(v) for v in primary])
    wsv_frame = np.concatenate([_upsample(v) for v in wsv_per_video])
    tcn_frame = np.concatenate([_upsample(v) for v in tcn_per_video])

    AUC_tcn = roc_auc_score(gt, tcn_frame)
    AUC_wsv = roc_auc_score(gt, wsv_frame)
    AP_tcn = average_precision_score(gt, tcn_frame)
    AP_wsv = average_precision_score(gt, wsv_frame)

    agnostic_stack = [_upsample(v) for v in primary]
    dmap_abn, _ = getDetectionMAP_abnormal_only(agnostic_stack, gtsegments, gtlabels)
    avg_abn = float(np.mean(dmap_abn))
    dmap_all, _ = getDetectionMAP_agnostic(agnostic_stack, gtsegments, gtlabels)
    avg_all = float(np.mean(dmap_all))

    diag = _compute_diag(primary, gt, gtsegments, gtlabels)
    diag['mAP_all'] = avg_all
    diag['dmap_all'] = dmap_all

    if not quiet:
        print(f'[TCN  ] AUC={AUC_tcn:.4f} AP={AP_tcn:.4f}')
        print(f'[WSVAD] logits1 AUC={AUC_wsv:.4f} AP={AP_wsv:.4f}')
        abn_str = '/'.join(f'{v:.2f}' for v in dmap_abn[:5])
        print(f'[abn @rel=0.60] AVG={avg_abn:.2f} [{abn_str}]')
        all_str = '/'.join(f'{v:.2f}' for v in dmap_all[:5])
        print(f'[all @rel=0.60] AVG={avg_all:.2f} [{all_str}]')

        fm = _frame_metrics(primary_frame, gt, thr=0.5)
        print(f'[frame @thr=0.50] TP={fm["TP"]} FP={fm["FP"]} FN={fm["FN"]} TN={fm["TN"]}'
              f' | P={fm["P"]:.3f} R={fm["R"]:.3f} F1={fm["F1"]:.3f} FPR={fm["FPR"]:.4f}')
        print(f'[sep  anom={diag["n_anom"]}] inside={diag["inside_mean"]:.3f} '
              f'outside={diag["outside_mean"]:.3f} Δ=+{diag["delta"]:.3f} '
              f"peak_in_gt={diag['peak_in_gt']:.3f}")
        print(f'[cov  anom={diag["n_anom"]}] over_cov(mean)={diag["over_cov_mean"]:.2f}x '
              f'over_cov(med)={diag["over_cov_med"]:.2f}x '
              f'coverage_inside={diag["cov_inside"]:.3f} '
              f'spillover={diag["spillover"]:.3f}')
        print(f'[bsh  anom={diag["n_anom"]}] boundary_sharp={diag["bsh_mean"]:.4f} '
              f'med={diag["bsh_med"]:.4f} k_frame=±48')
        print(f'[norm n={diag["n_norm"]}]    max(med)={diag["norm_max_med"]:.3f} '
              f'max(mean)={diag["norm_max_mean"]:.3f} '
              f'frac>0.5={diag["norm_frac_hi"]:.4f} n_max>0.9={diag["n_norm_hot"]}')

    if return_diag:
        return AUC_wsv, avg_abn, dmap_abn, AUC_tcn, diag
    return AUC_wsv, avg_abn, dmap_abn, AUC_tcn, None


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix, device,
                    tcn_dilations=tuple(args.tcn_dilations),
                    tcn_input=args.tcn_input,
                    use_a_branch=bool(args.use_a_branch))
    model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                     map_location=device), strict=False)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments,
         gtlabels, device, eval_head=args.eval_head)
