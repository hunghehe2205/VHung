import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.map_metrics import binary_detection_map
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _segments_to_mask(seg_list: list, length: int) -> np.ndarray:
    """Convert [[start, end], ...] segments to a binary mask of given length."""
    mask = np.zeros(length, dtype=bool)
    for s, e in seg_list:
        s = max(0, int(s))
        e = min(length, int(e))
        if e > s:
            mask[s:e] = True
    return mask


def _per_video_diagnostics(score_clip: np.ndarray, seg_list_clip: list) -> dict:
    """Per-video diagnostic stats, computed at clip resolution.

    - margin = soft_min(s_in) - soft_max(s_out) (NaN if no event or no normal)
    - intra_event_variances = list of Var(s_t) per contiguous event span (length >= 2)

    seg_list_clip MUST already be in clip resolution (not raw frames).
    """
    score_clip = np.asarray(score_clip, dtype=np.float64)
    mask = _segments_to_mask(seg_list_clip, score_clip.shape[0])

    out = {'margin': float('nan'), 'intra_event_vars': []}

    s_in = score_clip[mask]
    s_out = score_clip[~mask]
    if s_in.size > 0 and s_out.size > 0:
        out['margin'] = float(s_in.min() - s_out.max())

    # Find contiguous event spans for variance computation.
    if mask.any():
        padded = np.concatenate([[False], mask, [False]])
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for a, b in zip(starts, ends):
            if b - a >= 2:
                out['intra_event_vars'].append(float(score_clip[a:b].var()))
    return out


def _ano_auc(gt_all: np.ndarray, prob_all: np.ndarray, video_lens: list, video_labels: list) -> float:
    """ROC-AUC restricted to frames belonging to anomaly videos.

    A video is "anomaly" if its label is not 'Normal'. For mixed-frame AUC
    on this subset, normal frames inside an anomaly video still count as
    negatives (their gt is 0 wherever they're not in events).
    """
    assert len(video_lens) == len(video_labels)
    cursor = 0
    anomaly_gt_chunks = []
    anomaly_prob_chunks = []
    for L, lab in zip(video_lens, video_labels):
        seg_gt = gt_all[cursor: cursor + L]
        seg_p = prob_all[cursor: cursor + L]
        cursor += L
        if lab != 'Normal':
            anomaly_gt_chunks.append(seg_gt)
            anomaly_prob_chunks.append(seg_p)
    if not anomaly_gt_chunks:
        return float('nan')
    g = np.concatenate(anomaly_gt_chunks)
    p = np.concatenate(anomaly_prob_chunks)
    if g.sum() == 0 or g.sum() == len(g):
        return float('nan')
    return float(roc_auc_score(g, p))


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         device, logger=None, score_source='dbranch'):
    """Evaluate model on UCF test set.

    Reports AUC, Ano-AUC, AP at frame level (16x repeat to match raw frames)
    and Binary mAP@IoU class-agnostic at segment level.

    Returns a dict with keys: auc, ano_auc, ap, map_avg, map_at_iou_0X (for X in 1..5).
    """
    model.to(device)
    model.eval()

    binary_preds = []           # list of frame-level prob arrays per video (clip resolution)
    binary_gt_segs = []         # list of [[start, end], ...] per video
    video_clip_lens = []        # length in clips (matches binary_preds[i].shape[0])
    video_raw_lens = []         # length × 16 to slice gt
    video_labels = []           # per-video label string

    all_concat_pred = []        # frame-resolution (×16) concatenated for AUC/AP

    per_video_margins = []      # diagnostic: per-video min(s_in) - max(s_out)
    all_intra_event_vars = []   # diagnostic: variance of s_t inside each event span

    with torch.no_grad():
        for i, item in enumerate(tqdm(testdataloader, desc='Testing')):
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

            _, logits1, logits2, s_t = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[-1])
            logits2 = logits2.reshape(-1, logits2.shape[-1])
            s_t = s_t.reshape(-1)

            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0])
            prob_d = s_t[0:len_cur]

            source = {'prob1': prob1, 'prob2': prob2, 'dbranch': prob_d}[score_source]
            score_clip = source.detach().cpu().numpy().astype(np.float64)
            score_raw = np.repeat(score_clip, 16)

            binary_preds.append(score_clip)
            seg_list = gtsegments[i] if gtsegments[i] is not None else []
            seg_list_clip = [list(map(int, s)) for s in seg_list]
            binary_gt_segs.append(seg_list_clip)
            video_clip_lens.append(score_clip.shape[0])
            video_raw_lens.append(score_raw.shape[0])
            video_labels.append(label)
            all_concat_pred.append(score_raw)

            diag = _per_video_diagnostics(score_clip, seg_list_clip)
            if not np.isnan(diag['margin']):
                per_video_margins.append(diag['margin'])
            all_intra_event_vars.extend(diag['intra_event_vars'])

    pred_all = np.concatenate(all_concat_pred)
    # gt is the canonical raw-frame ground truth array.
    gt_all = np.asarray(gt[: pred_all.shape[0]], dtype=np.float32)
    if pred_all.shape[0] != gt.shape[0]:
        # The split path may produce a slightly different total length when a
        # video's clip count isn't a multiple of maxlen. Trim to common length
        # so AUC/AP remain well-defined.
        n = min(pred_all.shape[0], gt.shape[0])
        pred_all = pred_all[:n]
        gt_all = np.asarray(gt[:n], dtype=np.float32)

    auc = float(roc_auc_score(gt_all, pred_all))
    ap = float(average_precision_score(gt_all, pred_all))
    ano_auc = _ano_auc(gt_all, pred_all, video_raw_lens, video_labels)

    bmap = binary_detection_map(binary_preds, binary_gt_segs)

    mean_margin = float(np.mean(per_video_margins)) if per_video_margins else float('nan')
    mean_intra_var = float(np.mean(all_intra_event_vars)) if all_intra_event_vars else float('nan')

    result = {
        'auc': auc,
        'ano_auc': ano_auc,
        'ap': ap,
        'map_avg': bmap['map_avg'],
        'map_at_iou_01': bmap['map_at_iou_01'],
        'map_at_iou_03': bmap['map_at_iou_03'],
        'map_at_iou_05': bmap['map_at_iou_05'],
        'map_at_iou_07': bmap['map_at_iou_07'],
        'mean_margin': mean_margin,
        'mean_intra_event_variance': mean_intra_var,
        'score_source': score_source,
    }

    print(f"  [Eval source={score_source}]")
    print(f"  AUC      = {auc:.4f}")
    print(f"  Ano-AUC  = {ano_auc:.4f}")
    print(f"  AP       = {ap:.4f}")
    print(f"  mAP AVG  = {bmap['map_avg']:.4f}  "
          f"(@0.1={bmap['map_at_iou_01']:.4f} @0.3={bmap['map_at_iou_03']:.4f} "
          f"@0.5={bmap['map_at_iou_05']:.4f} @0.7={bmap['map_at_iou_07']:.4f})")
    print(f"  [Diag] mean_margin={mean_margin:.4f}  mean_intra_event_var={mean_intra_var:.4f}")

    if logger:
        logger.info(
            f"[Eval src={score_source}] AUC={auc:.4f} AnoAUC={ano_auc:.4f} AP={ap:.4f} | "
            f"mAP_avg={bmap['map_avg']:.4f} "
            f"@.1={bmap['map_at_iou_01']:.4f} @.3={bmap['map_at_iou_03']:.4f} "
            f"@.5={bmap['map_at_iou_05']:.4f} @.7={bmap['map_at_iou_07']:.4f} | "
            f"margin={mean_margin:.4f} intra_var={mean_intra_var:.4f}"
        )

    return result


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = option.parser.parse_args()

    testdataset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)
    # strict=False so stock VadCLIP checkpoints (no d_branch) load with random head;
    # pair with --score-source prob1 to reproduce the baseline AUC/Ano-AUC/mAP.
    missing, unexpected = model.load_state_dict(
        torch.load(args.model_path, weights_only=False, map_location=device), strict=False)
    if missing:
        print(f"[load] missing keys (random init): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"[load] unexpected keys (ignored): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")

    import logging
    logger = logging.getLogger('dbranch')
    if not logger.handlers:
        os.makedirs(args.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(args.log_dir, 'test.log'), mode='a')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    test(model, testdataloader, args.visual_length, prompt_text,
         gt, gtsegments, gtlabels, device, logger,
         score_source=args.score_source)
