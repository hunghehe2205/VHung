"""Per-video anomaly-map visualizer for the CLIPVAD model.

Loads a trained CLIPVAD checkpoint, runs inference over the UCF test split,
and for every video writes a PNG at
    docs/anomaly_maps/<video_id>/anomaly_map.png
that plots the linearly upsampled frame-level anomaly score along with the
GT segment(s) and class label.
"""
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from torch.utils.data import DataLoader

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

_VIDEO_NAME_RE = re.compile(r'(.*?)__\d+$')


def _parse_video_id(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    m = _VIDEO_NAME_RE.match(base)
    return m.group(1) if m else base


def _upsample_linear_1d(x, factor=16):
    x = np.asarray(x, dtype=np.float32)
    T = len(x)
    if T == 0:
        return x
    centers = np.arange(T) * factor + (factor - 1) / 2.0
    frame_idx = np.arange(T * factor)
    return np.interp(frame_idx, centers, x)


def _plot_one(out_path, prob_frame, gt_segments, class_name, is_normal,
              normalize=False):
    n_frames = len(prob_frame)
    disp = prob_frame
    ymin, ymax = 0.0, 1.0
    ylabel = 'P(anomaly)'
    if normalize:
        lo, hi = float(disp.min()), float(disp.max())
        if hi - lo > 1e-6:
            disp = (disp - lo) / (hi - lo)
        ylabel = f'P scaled  (raw {lo:.2f}–{hi:.2f})'
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=120)

    # Shade GT anomaly segments — only for anomaly videos.
    first_seg = None
    if not is_normal and gt_segments is not None and len(gt_segments) > 0:
        for seg in gt_segments:
            if seg is None or len(seg) < 2:
                continue
            start, end = float(seg[0]), float(seg[1])
            if end < 0 or start < 0:
                continue
            start = max(0.0, start)
            end = min(float(n_frames - 1), end)
            if end <= start:
                continue
            ax.axvspan(start, end, color='pink', alpha=0.5, linewidth=0)
            if first_seg is None:
                first_seg = (start, end)

    x = np.arange(n_frames)
    ax.plot(x, disp, color='#1f77b4', linewidth=1.2)

    if first_seg is not None and class_name:
        cx = 0.5 * (first_seg[0] + first_seg[1])
        ax.text(cx, 0.1, class_name, fontsize=12, color='#c91b5c',
                fontweight='bold', ha='center', va='center', clip_on=True)

    ax.set_xlim(0, max(1, n_frames - 1))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Frame', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis='both', labelsize=8, length=3)
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.25))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=6, integer=True))
    ax.grid(True, axis='y', linestyle=':', linewidth=0.4, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if is_normal:
        ax.set_title('Normal', fontsize=10, color='#1f77b4', pad=4)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


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
                    args.attn_window, args.prompt_prefix, args.prompt_postfix,
                    device)
    model.load_state_dict(torch.load(args.model_path, weights_only=False,
                                     map_location=device), strict=False)
    model.to(device).eval()

    out_root = args.out_dir
    maxlen = args.visual_length

    paths = list(testdataset.df['path'])
    labels_csv = list(testdataset.df['label'])
    n_total = len(paths)

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
            logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1],
                                      logits1.shape[2])
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()

            prob_frame = _upsample_linear_1d(prob1, factor=16)

            video_id = _parse_video_id(paths[i])
            segs = gtsegments[i] if i < len(gtsegments) else []
            labs = gtlabels[i] if i < len(gtlabels) else []
            csv_label = labels_csv[i] if i < len(labels_csv) else ''
            is_normal = str(csv_label).lower() == 'normal'
            class_name = csv_label if not is_normal else ''

            out_path = os.path.join(out_root, video_id, 'anomaly_map.png')
            _plot_one(out_path, prob_frame, segs, class_name, is_normal,
                      normalize=args.normalize)

            if (i + 1) % 30 == 0:
                print(f'[visualize] {i + 1}/{n_total}', flush=True)

    print(f'[visualize] done — wrote {n_total} videos to {out_root}')


if __name__ == '__main__':
    main()
