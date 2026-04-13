"""Visualize anomaly score maps from logits3.

Usage:
    python src/visualize_maps.py --maps-dir logs_phaseB/score_maps --gt-path list/gt_ucf.npy --test-list list/ucf_CLIP_rgbtest.csv
    python src/visualize_maps.py --maps-dir logs_phaseB/score_maps --video-indices 0,5,10  # specific videos
    python src/visualize_maps.py --maps-dir logs_phaseB/score_maps --video-names Burglary,Stealing  # filter by class
"""
import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def load_score_maps(maps_dir):
    """Load all score maps, return sorted list of (name, scores)."""
    files = sorted(glob.glob(os.path.join(maps_dir, "*.npy")))
    maps = []
    for f in files:
        name = os.path.basename(f).replace(".npy", "")
        scores = np.load(f)
        maps.append((name, scores))
    return maps


def load_gt(gt_path, test_list):
    """Load frame-level GT and video boundaries from test list."""
    gt = np.load(gt_path)

    # Reconstruct per-video boundaries from test CSV
    import pandas as pd
    df = pd.read_csv(test_list)
    video_lengths = []
    for _, row in df.iterrows():
        feat = np.load(row['path'])
        video_lengths.append(feat.shape[0] * 16)  # clip_length * 16 = frame_length

    # Build per-video GT slices
    boundaries = []
    offset = 0
    for vl in video_lengths:
        boundaries.append((offset, offset + vl))
        offset += vl

    return gt, boundaries


def plot_single(ax, name, scores, gt_slice=None):
    """Plot a single video's anomaly map."""
    frames = np.arange(len(scores))
    ax.plot(frames, scores, color='purple', linewidth=0.8, alpha=0.8)
    ax.fill_between(frames, scores, alpha=0.3, color='purple')

    if gt_slice is not None:
        # Highlight GT anomaly regions
        anomaly_frames = np.where(gt_slice > 0)[0]
        if len(anomaly_frames) > 0:
            # Find contiguous regions
            diffs = np.diff(anomaly_frames)
            splits = np.where(diffs > 1)[0] + 1
            regions = np.split(anomaly_frames, splits)
            for region in regions:
                if len(region) > 0:
                    ax.axvspan(region[0], region[-1], alpha=0.15, color='red')

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(name, fontsize=10)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps-dir', required=True, help='Directory containing score map .npy files')
    parser.add_argument('--gt-path', default=None, help='GT .npy file for overlay (optional)')
    parser.add_argument('--test-list', default=None, help='Test CSV for per-video GT boundaries (optional)')
    parser.add_argument('--video-indices', default=None, help='Comma-separated video indices to plot (e.g. 0,5,10)')
    parser.add_argument('--video-names', default=None, help='Filter by class name substring (e.g. Burglary,Stealing)')
    parser.add_argument('--max-plots', default=12, type=int, help='Max number of plots')
    parser.add_argument('--output', default=None, help='Save to file instead of showing (e.g. maps.png)')
    args = parser.parse_args()

    maps = load_score_maps(args.maps_dir)
    print(f"Loaded {len(maps)} score maps from {args.maps_dir}")

    # Load GT if available
    gt = None
    boundaries = None
    if args.gt_path and args.test_list:
        gt, boundaries = load_gt(args.gt_path, args.test_list)
        print(f"Loaded GT: {len(gt)} frames, {len(boundaries)} videos")

    # Filter maps
    if args.video_indices:
        indices = [int(x) for x in args.video_indices.split(',')]
        maps = [maps[i] for i in indices if i < len(maps)]
    elif args.video_names:
        names = args.video_names.split(',')
        maps = [(n, s) for n, s in maps if any(name in n for name in names)]

    maps = maps[:args.max_plots]

    if not maps:
        print("No maps to plot!")
        return

    # Compute stats
    print(f"\nPlotting {len(maps)} videos:")
    for name, scores in maps:
        print(f"  {name}: len={len(scores)} mean={scores.mean():.3f} "
              f"max={scores.max():.3f} min={scores.min():.3f} "
              f">0.5={( scores > 0.5).mean():.1%}")

    # Plot
    n_plots = len(maps)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), squeeze=False)

    for idx, (name, scores) in enumerate(maps):
        ax = axes[idx, 0]
        # Extract video index from name (video_NNNN_label)
        gt_slice = None
        if gt is not None and boundaries is not None:
            parts = name.split('_')
            if len(parts) >= 2:
                try:
                    vid_idx = int(parts[1])
                    if vid_idx < len(boundaries):
                        start, end = boundaries[vid_idx]
                        gt_slice = gt[start:end]
                        # Trim to match score length
                        gt_slice = gt_slice[:len(scores)]
                except (ValueError, IndexError):
                    pass
        plot_single(ax, name, scores, gt_slice)

    axes[-1, 0].set_xlabel('Frame')
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
