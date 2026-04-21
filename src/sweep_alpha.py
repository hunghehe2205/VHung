"""Fusion-alpha sweep: fused = α·sigmoid(logits1) + (1-α)·sigmoid(tcn_logits).

One forward pass per checkpoint, then loop α ∈ {0.0 … 1.0} step 0.1 and
compute 7 metrics per α. Sanity rows:
  2D α=0 → mAP_all≈19.16, bsh≈0.199, peak≈0.429
  2D α=1 → AUC≈0.8577
  4B α=0 → mAP_all≈19.24, mAP_abn≈21.02
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CLIPVAD
from test import LABEL_MAP, _compute_diag, _upsample
from utils.dataset import UCFDataset
from utils.detection_map import (getDetectionMAP_abnormal_only,
                                 getDetectionMAP_agnostic)
from utils.tools import get_batch_mask, get_prompt_text
import option


def collect_per_video(model, loader, maxlen, prompt_text, device):
    """Forward once; return per-video sigmoid arrays for both heads."""
    model.to(device)
    model.eval()
    tcn_pv, wsv_pv = [], []
    with torch.no_grad():
        for item in tqdm(loader, desc='Forward',
                         disable=not sys.stderr.isatty()):
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

            _, logits1, _, tcn_logits = model(
                visual, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[-1])
            tcn_logits = tcn_logits.reshape(-1, tcn_logits.shape[-1])
            tcn_pv.append(
                torch.sigmoid(tcn_logits[0:len_cur, 0]).cpu().numpy())
            wsv_pv.append(
                torch.sigmoid(logits1[0:len_cur, 0]).cpu().numpy())
    return tcn_pv, wsv_pv


def peak_top2_in_gt(snip_scores, gt_slice, factor=16, W=16):
    """Snippet-level top-2 NMS (window=±W snippets). Returns mean of two
    indicators (each peak's frame span intersects GT). ∈ {0.0, 0.5, 1.0}."""
    s = np.asarray(snip_scores, dtype=np.float32).copy()
    T = len(s)
    if T == 0:
        return 0.0
    p1 = int(np.argmax(s))
    s_sup = s.copy()
    lo, hi = max(0, p1 - W), min(T, p1 + W + 1)
    s_sup[lo:hi] = -np.inf
    if not np.isfinite(s_sup).any():
        p2 = p1
    else:
        p2 = int(np.argmax(s_sup))

    def in_gt(p):
        fl = p * factor
        fh = min(len(gt_slice), (p + 1) * factor)
        if fh <= fl:
            return 0.0
        return float(gt_slice[fl:fh].any())
    return 0.5 * (in_gt(p1) + in_gt(p2))


def slice_gt(gt_full, per_video):
    lens = [len(_upsample(p)) for p in per_video]
    offsets = np.cumsum([0] + lens)
    return [gt_full[offsets[i]:offsets[i] + lens[i]]
            for i in range(len(per_video))]


def compute_row(fused_pv, gt, gtsegments, gtlabels, gt_slices):
    diag = _compute_diag(fused_pv, gt, gtsegments, gtlabels)
    frame = np.concatenate([_upsample(v) for v in fused_pv])
    auc = roc_auc_score(gt, frame)
    stack = [_upsample(v) for v in fused_pv]
    dmap_abn, _ = getDetectionMAP_abnormal_only(stack, gtsegments, gtlabels)
    dmap_all, _ = getDetectionMAP_agnostic(stack, gtsegments, gtlabels)

    pt2_vals = []
    for i, vid in enumerate(fused_pv):
        is_anom = any(l != 'A' for l in gtlabels[i])
        if is_anom:
            pt2_vals.append(peak_top2_in_gt(vid, gt_slices[i]))
    peak_top2 = float(np.mean(pt2_vals)) if pt2_vals else 0.0

    return dict(
        mAP_all=float(np.mean(dmap_all)),
        mAP_abn=float(np.mean(dmap_abn)),
        AUC=float(auc),
        bsh_med=diag['bsh_med'],
        peak_in_gt=diag['peak_in_gt'],
        peak_top2=peak_top2,
        over_cov_med=diag['over_cov_med'],
    )


def print_table(name, rows):
    print(f'\n=== Checkpoint: {name} ===')
    header = (f'{"α":<5}{"mAP_all":>9}{"mAP_abn":>9}{"AUC":>8}'
              f'{"bsh":>8}{"peak":>8}{"peak_t2":>9}{"over_cov":>10}')
    print(header)
    print('-' * len(header))
    for alpha, m in rows:
        tag = ''
        if abs(alpha) < 1e-6:
            tag = '  (TCN only)'
        elif abs(alpha - 1.0) < 1e-6:
            tag = '  (WSV only)'
        print(f'{alpha:<5.1f}{m["mAP_all"]:>9.2f}{m["mAP_abn"]:>9.2f}'
              f'{m["AUC"]:>8.4f}{m["bsh_med"]:>8.3f}{m["peak_in_gt"]:>8.3f}'
              f'{m["peak_top2"]:>9.3f}{m["over_cov_med"]:>9.2f}x{tag}')


def sweep_one(ckpt_path, name, args, alphas, device):
    testset = UCFDataset(args.visual_length, args.test_list, True, LABEL_MAP)
    loader = DataLoader(testset, batch_size=1, shuffle=False)
    prompt_text = get_prompt_text(LABEL_MAP)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length,
                    args.visual_width, args.visual_head, args.visual_layers,
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device, tcn_dilations=tuple(args.tcn_dilations),
                    tcn_input=args.tcn_input)
    print(f'\nLoading {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, weights_only=False,
                                      map_location=device), strict=False)

    tcn_pv, wsv_pv = collect_per_video(
        model, loader, args.visual_length, prompt_text, device)
    gt_slices = slice_gt(gt, tcn_pv)

    rows = []
    for alpha in alphas:
        fused = [alpha * w + (1.0 - alpha) * t
                 for t, w in zip(tcn_pv, wsv_pv)]
        rows.append((alpha, compute_row(
            fused, gt, gtsegments, gtlabels, gt_slices)))
    print_table(name, rows)
    return rows


def main():
    parser = option.parser
    parser.add_argument('--ckpt-2d',
                        default='final_model/model_exp2d_tcn.pth')
    parser.add_argument('--ckpt-4b',
                        default='final_model/model_exp4b_tcn.pth')
    parser.add_argument('--alphas',
                        default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    alphas = [float(a) for a in args.alphas.split(',')]

    for ckpt, name in [(args.ckpt_2d, '2D'), (args.ckpt_4b, '4B')]:
        if not ckpt:
            continue
        if not os.path.isfile(ckpt):
            print(f'[skip] {name} ckpt not found: {ckpt}')
            continue
        sweep_one(ckpt, name, args, alphas, device)


if __name__ == '__main__':
    main()
