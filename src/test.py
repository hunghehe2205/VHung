import sys
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
import option

LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism'
}


def _upsample_linear_1d(x, factor):
    """Linear temporal upsample of 1D array [T] -> [T*factor]. T-step i is
    treated as a sample at frame i*factor + (factor-1)/2; edges hold endpoint.
    """
    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    if T == 0:
        return x
    centers = np.arange(T) * factor + (factor - 1) / 2.0
    frame_idx = np.arange(T * factor)
    return np.interp(frame_idx, centers, x)


def _upsample_linear_2d(x, factor):
    """Axis-0 linear upsample of 2D array [T, C] -> [T*factor, C]."""
    x = np.asarray(x, dtype=np.float32)
    T, C = x.shape
    out = np.empty((T * factor, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = _upsample_linear_1d(x[:, c], factor)
    return out


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device,
         quiet=False, upsample='repeat', inference='threshold',
         bsn_start_thresh=0.5, bsn_end_thresh=0.5, bsn_max_dur=2048):
    model.to(device)
    model.eval()

    if upsample == 'linear':
        up1d = lambda v: _upsample_linear_1d(v, 16)
        up2d = lambda v: _upsample_linear_2d(v, 16)
    else:
        up1d = lambda v: np.repeat(v, 16)
        up2d = lambda v: np.repeat(v, 16, 0)

    element_logits2_stack = []

    with torch.no_grad():
        ap1_per_video = []
        ap2_per_video = []
        start_cls_per_video = []
        start_off_per_video = []
        end_cls_per_video = []
        end_off_per_video = []
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

            _, logits1, logits2, s_logits, e_logits = model(
                visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            # s_logits, e_logits: [chunks, T, 2] → [chunks*T, 2]
            s_logits = s_logits.reshape(s_logits.shape[0] * s_logits.shape[1], s_logits.shape[2])
            e_logits = e_logits.reshape(e_logits.shape[0] * e_logits.shape[1], e_logits.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            ap1_per_video.append(prob1.cpu().numpy())
            ap2_per_video.append(prob2.cpu().numpy())
            # D=2: channel 0 = cls, channel 1 = offset
            start_cls_per_video.append(
                torch.sigmoid(s_logits[0:len_cur, 0]).cpu().numpy())
            start_off_per_video.append(
                torch.sigmoid(s_logits[0:len_cur, 1]).cpu().numpy())
            end_cls_per_video.append(
                torch.sigmoid(e_logits[0:len_cur, 0]).cpu().numpy())
            end_off_per_video.append(
                torch.sigmoid(e_logits[0:len_cur, 1]).cpu().numpy())

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = up2d(element_logits2)
            element_logits2_stack.append(element_logits2)

    ap1_frame = np.concatenate([up1d(v) for v in ap1_per_video])
    ap2_frame = np.concatenate([up1d(v) for v in ap2_per_video])

    ROC1 = roc_auc_score(gt, ap1_frame)
    AP1 = average_precision_score(gt, ap1_frame)
    ROC2 = roc_auc_score(gt, ap2_frame)
    AP2 = average_precision_score(gt, ap2_frame)

    from utils.detection_map import getDetectionMAP_agnostic, getDetectionMAP_agnostic_bsn

    # Per-class (legacy)
    dmap_pc, iou = dmAP(element_logits2_stack, gtsegments, gtlabels,
                       excludeNormal=False)
    averageMAP_pc = float(np.mean(dmap_pc[:5]))

    # Dual eval: threshold mAP always (model selection), BSN diagnostic
    agnostic_stack = [up1d(fs) for fs in ap1_per_video]
    dmap_thr, _ = getDetectionMAP_agnostic(agnostic_stack, gtsegments, gtlabels)
    avg_thr = float(np.mean(dmap_thr))

    bsn_stats = None
    dmap_bsn = None
    if inference == 'bsn':
        # Snippet-resolution peak picking + offset refinement
        dmap_bsn, _, bsn_stats = getDetectionMAP_agnostic_bsn(
            ap1_per_video, start_cls_per_video, end_cls_per_video,
            gtsegments, gtlabels,
            start_offs=start_off_per_video, end_offs=end_off_per_video,
            start_thr=bsn_start_thresh, end_thr=bsn_end_thresh,
            max_dur=bsn_max_dur)

    if not quiet:
        print(f"AUC1={ROC1:.4f} AP1={AP1:.4f} | AUC2={ROC2:.4f} AP2={AP2:.4f}")
        pc_str = '/'.join(f'{v:.2f}' for v in dmap_pc[:5])
        thr_str = '/'.join(f'{v:.2f}' for v in dmap_thr[:5])
        print(f"[per-class] AVG={averageMAP_pc:.2f} [{pc_str}]")
        print(f"[threshold] AVG={avg_thr:.2f} [{thr_str}]")
        if dmap_bsn is not None:
            bsn_str = '/'.join(f'{v:.2f}' for v in dmap_bsn[:5])
            avg_bsn = float(np.mean(dmap_bsn))
            print(f"[BSN     ] AVG={avg_bsn:.2f} [{bsn_str}]")
        if bsn_stats:
            st = bsn_stats
            avg_prop = st['total_nms_proposals'] / max(1, st['n_with_proposals'])
            print(f"[BSN] {st['n_with_proposals']}/{st['n_videos']} videos with proposals, "
                  f"{st['n_skipped']} skipped | "
                  f"peaks: {st['total_starts']}s/{st['total_ends']}e | "
                  f"proposals: {st['total_raw_proposals']} raw -> {st['total_nms_proposals']} nms "
                  f"({avg_prop:.1f}/video) | "
                  f"A:{st['n_anomaly_with_prop']}v/{st['n_anomaly_proposals']}p "
                  f"N:{st['n_normal_with_prop']}v/{st['n_normal_proposals']}p")

    return ROC1, avg_thr, dmap_thr, dmap_bsn, bsn_stats


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
    model.load_state_dict(torch.load(args.model_path, weights_only=False, map_location=device),
                          strict=False)

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device,
         upsample=args.upsample, inference=args.inference,
         bsn_start_thresh=args.bsn_start_thresh, bsn_end_thresh=args.bsn_end_thresh,
         bsn_max_dur=args.bsn_max_dur)
