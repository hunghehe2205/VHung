import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.detection_map import getDetectionMAP as dmAP
from utils.map_metrics import (
    compute_per_video_metrics,
    binary_detection_map,
    composite_map_score,
)
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         device, logger=None):
    model.to(device)
    model.eval()

    element_logits2_stack = []
    score_maps = {}

    per_video_metrics = []
    binary_preds = []      # list of frame-level prob3 arrays
    binary_gt_segs = []    # list of [[start, end], ...]
    video_start = 0        # running index into `gt` array

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

            _, logits1, logits2, logits3 = model(visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            logits3 = logits3.reshape(logits3.shape[0] * logits3.shape[1], logits3.shape[2])

            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            prob3 = torch.sigmoid(logits3[0:len_cur].squeeze(-1))

            if i == 0:
                ap1 = prob1
                ap2 = prob2
                ap3 = prob3
            else:
                ap1 = torch.cat([ap1, prob1], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)
                ap3 = torch.cat([ap3, prob3], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

            score_map_np = np.repeat(prob3.cpu().numpy(), 16)
            score_maps[f"video_{i:04d}_{label}"] = score_map_np

            # Per-video map metrics (uses frame-level gt slice + gt_segments[i]).
            video_len_frames = score_map_np.shape[0]
            gt_slice = np.asarray(gt[video_start: video_start + video_len_frames], dtype=np.float32)
            video_start += video_len_frames

            metrics_i = compute_per_video_metrics(score_map_np, gt_slice)
            per_video_metrics.append(metrics_i)
            binary_preds.append(score_map_np)
            seg_list = gtsegments[i] if gtsegments[i] is not None else []
            binary_gt_segs.append([list(map(int, s)) for s in seg_list])

    ap1 = ap1.cpu().numpy().tolist()
    ap2 = ap2.cpu().numpy().tolist()
    ap3 = ap3.cpu().numpy().tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))
    ROC3 = roc_auc_score(gt, np.repeat(ap3, 16))
    AP3 = average_precision_score(gt, np.repeat(ap3, 16))

    print(f"  AUC1: {ROC1:.4f}  AP1: {AP1:.4f}  (logits1 - baseline)")
    print(f"  AUC3: {ROC3:.4f}  AP3: {AP3:.4f}  (logits3 - map head)")
    print(f"  AUC2: {ROC2:.4f}  AP2: {AP2:.4f}  (logits2 - text align)")

    # Aggregate per-video metrics. Use nan-aware mean for metrics defined only on anomaly videos.
    def _nanmean_key(key):
        vals = [m[key] for m in per_video_metrics if not (isinstance(m[key], float) and np.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float('nan')

    gap_mean = _nanmean_key('gap')
    mcl_mean = _nanmean_key('mcl')
    pc_mean = _nanmean_key('peak_concentration')
    cov05_mean = _nanmean_key('in_event_cov_05')
    cov03_mean = _nanmean_key('in_event_cov_03')
    entropy_mean = _nanmean_key('in_event_entropy')

    # Normal-mean: averaged on videos with no events (gtsegments[i] empty).
    normal_means = []
    for i, m in enumerate(per_video_metrics):
        if len(binary_gt_segs[i]) == 0:
            normal_means.append(m['out_mean'])
    normal_mean = float(np.mean(normal_means)) if normal_means else float('nan')

    binary_map = binary_detection_map(binary_preds, binary_gt_segs)

    aggregated = {
        'gap': gap_mean, 'mcl': mcl_mean, 'peak_concentration': pc_mean,
        'map_avg': binary_map['map_avg'],
    }
    map_score = composite_map_score(aggregated)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for k in range(5):
        print('  mAP@{0:.1f} ={1:.2f}%'.format(iou[k], dmap[k]))
        averageMAP += dmap[k]
    averageMAP = averageMAP / 5
    print('  average MAP: {:.2f}'.format(averageMAP))

    print("  --- Map Quality Metrics ---")
    print(f"  [Separation] Gap={gap_mean:.3f}  Normal-mean={normal_mean:.3f}")
    print(f"  [Mass]       MCL={mcl_mean:.3f}")
    print(f"  [Localize]   mAP@IoU avg={binary_map['map_avg']:.3f}  "
          f"@0.1={binary_map['map_at_iou_01']:.3f}  @0.5={binary_map['map_at_iou_05']:.3f}")
    print(f"  [Density]    InEventCov@0.5={cov05_mean:.3f}  InEventCov@0.3={cov03_mean:.3f}  "
          f"PeakConc={pc_mean:.3f}  Entropy={entropy_mean:.3f}")
    print(f"  [Composite]  MapScore={map_score:.4f}")

    if logger:
        logger.info(
            f"[Eval] AUC1={ROC1:.4f} AP1={AP1:.4f} | "
            f"AUC3={ROC3:.4f} AP3={AP3:.4f} | "
            f"AUC2={ROC2:.4f} AP2={AP2:.4f} | "
            f"avgMAP={averageMAP:.2f}")
        logger.info(
            f"[MapEval] Gap={gap_mean:.3f} NormMean={normal_mean:.3f} MCL={mcl_mean:.3f} "
            f"mAPavg={binary_map['map_avg']:.3f} InCov05={cov05_mean:.3f} "
            f"PC={pc_mean:.3f} Entropy={entropy_mean:.3f} MapScore={map_score:.4f}"
        )

    return ROC1, AP1, ROC3, AP3, score_maps, map_score


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
    model.load_state_dict(torch.load(args.model_path, weights_only=False))

    import logging
    logger = logging.getLogger('logits3')
    if not logger.handlers:
        os.makedirs(args.log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(args.log_dir, 'train.log'), mode='a')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    _, _, _, _, score_maps, _ = test(model, testdataloader, args.visual_length, prompt_text,
                                     gt, gtsegments, gtlabels, device, logger)

    maps_dir = os.path.join(args.log_dir, 'score_maps')
    os.makedirs(maps_dir, exist_ok=True)
    for name, smap in score_maps.items():
        np.save(os.path.join(maps_dir, f"{name}.npy"), smap)
