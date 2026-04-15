import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def parse_ucf_gt(txt_path):
    """Parse UCF-Crime Temporal_Anomaly_Annotation_for_Testing_Videos.txt.

    Returns dict keyed by video id without extension:
      {'class': str, 'seg1': [start, end] | None, 'seg2': [start, end] | None}
    Frame indices are as given in the annotation file (1-indexed upstream).
    """
    out = {}
    with open(txt_path, 'r') as fp:
        for line in fp:
            parts = line.split()
            if not parts:
                continue
            mp4_positions = [i for i, tok in enumerate(parts) if tok.endswith('.mp4')]
            if not mp4_positions:
                continue
            i0 = mp4_positions[0]
            try:
                video_name = parts[i0]
                cls = parts[i0 + 1]
                a, b, c, d = (int(parts[i0 + j]) for j in range(2, 6))
            except (IndexError, ValueError):
                continue
            seg1 = [a, b] if a > 0 and b > 0 else None
            seg2 = [c, d] if c > 0 and d > 0 else None
            key = video_name.rsplit('.mp4', 1)[0]
            out[key] = {'class': cls, 'seg1': seg1, 'seg2': seg2}
    return out


def load_frames(video_path, frame_indices):
    """Read RGB frames at given indices. Returns (frames, total_frames, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    for idx in frame_indices:
        idx = max(0, min(total - 1, int(idx)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames, total, fps


def compose_viz(video_id, class_label, frames, anomaly_scores,
                sampled_snippet_idxs, gt_snippet_ranges, out_path,
                figsize=(18, 7)):
    """Combined single-image visualization: frame strip over anomaly curve."""
    K = max(len(frames), 1)
    T = len(anomaly_scores)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, K, height_ratios=[1.1, 1.6], hspace=0.08, wspace=0.04,
    )

    for i, img in enumerate(frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_curve = fig.add_subplot(gs[1, :])
    x = np.arange(T)
    ax_curve.plot(x, anomaly_scores, color='purple', linewidth=1.6)

    for rng in gt_snippet_ranges:
        if rng is None:
            continue
        s, e = rng
        ax_curve.axvspan(s, e, color='pink', alpha=0.45, linewidth=0)

    for idx in sampled_snippet_idxs:
        ax_curve.axvline(idx, color='red', linewidth=1.0)

    ax_curve.set_xlim(0, max(T - 1, 1))
    ax_curve.set_ylim(0, 1.0)
    ax_curve.set_xlabel('snippet')
    ax_curve.set_ylabel('anomaly score')

    fig.suptitle(f"{video_id}  ({class_label})", fontsize=14)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
