"""
Infer anomaly scores directly from a video file or pre-extracted features.

Usage:
    python infer_video.py --mode video
    python infer_video.py --mode feature
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel

from src.crop_intern import video_crop, build_transform
from src.utils.tools import process_split, pad
from intern_vad import VadInternVL
import src.ucf_option as ucf_option

# ============ CONFIG ============
# Mode: video — extract features on the fly (requires InternVL)
VIDEO_PATH = '/home/emogenai4e/emo/Hung_data/UCF_Crime/Abuse/Abuse028_x264.mp4'
INTERNVL_PATH = 'ppxin321/HolmesVAU-2B'

# Mode: feature — use pre-extracted .npy features (fast, no InternVL needed)
FEATURE_PATH = '/home/emogenai4e/emo/Hung_data/ucf_internvl_test_feature/Abuse/Abuse028_x264__5.npy'

MODEL_PATH = '/home/emogenai4e/emo/VHung/model/model_ucf_intern.pth'
OUTPUT_DIR = 'output_infer'
OUTPUT_PATH = None  # None = auto from video/feature name
# ================================

NUM_FRAMES = 16
BATCH_SIZE = 32
VISUAL_LENGTH = 256


def load_all_frames(video_path):
    """Load frames uniformly sampled every NUM_FRAMES from video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total <= 0:
        raise ValueError(f"Cannot read video: {video_path}")

    indices = np.arange(0, total, NUM_FRAMES)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return np.array(frames), fps, total


def extract_features(internvl_model, transform, frames, device):
    """Extract InternVL CLS features from frames (center crop)."""
    cropped = video_crop(frames, 0)  # center crop
    features = []
    with torch.no_grad():
        for i in range(0, cropped.shape[0], BATCH_SIZE):
            batch = cropped[i:i + BATCH_SIZE]
            imgs = torch.stack([transform(Image.fromarray(f)) for f in batch])
            imgs = imgs.to(torch.bfloat16).to(device)
            vit_embeds = internvl_model.vision_model(
                pixel_values=imgs,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            cls_token = vit_embeds[:, 0, :]
            features.append(cls_token.cpu().float())
    return torch.cat(features, dim=0).numpy()  # [T, 1024]


def infer(vad_model, features, device):
    """Run VAD model on features, return per-snippet anomaly scores."""
    num_snippets = features.shape[0]

    if num_snippets <= VISUAL_LENGTH:
        feat = pad(features, VISUAL_LENGTH)
        feat = torch.tensor(feat).unsqueeze(0).to(device)
        lengths = torch.tensor([num_snippets]).to(int)
        with torch.no_grad():
            logits = vad_model(feat, lengths)
        scores = torch.sigmoid(logits).squeeze(-1).squeeze(0)[:num_snippets]
    else:
        feat, _ = process_split(features, VISUAL_LENGTH)
        feat = torch.tensor(feat).to(device)
        num_splits = feat.shape[0]
        lengths = torch.zeros(num_splits).to(int)
        remaining = num_snippets
        for j in range(num_splits):
            if remaining >= VISUAL_LENGTH:
                lengths[j] = VISUAL_LENGTH
                remaining -= VISUAL_LENGTH
            else:
                lengths[j] = remaining
        with torch.no_grad():
            logits = vad_model(feat, lengths)
        logits = logits.reshape(-1, 1)
        scores = torch.sigmoid(logits[:num_snippets].squeeze(-1))

    return scores.cpu().numpy()


def sample_keyframes(video_path, num_keyframes=8):
    """Sample evenly spaced keyframes for display."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_keyframes, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((120, 160, 3), dtype=np.uint8))
    cap.release()
    return frames, indices


GT_TXT = 'data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'


def load_gt_segments(video_path_or_name):
    """Load ground truth anomaly segments for a video from annotation file."""
    video_name = os.path.basename(video_path_or_name)
    if not os.path.exists(GT_TXT):
        return []
    segments = []
    for line in open(GT_TXT):
        if video_name in line:
            parts = line.strip().split('  ')
            # parts: [filename, label, start1, end1, start2, end2]
            nums = [int(x) for x in parts[2:]]
            for i in range(0, len(nums), 2):
                start, end = nums[i], nums[i + 1]
                if start != -1 and end != -1:
                    segments.append((start, end))
            break
    return segments


def plot_result(scores, video_path, output_path):
    """Plot anomaly scores with keyframes on top and GT shading."""
    keyframes, kf_indices = sample_keyframes(video_path, num_keyframes=8)
    gt_segments = load_gt_segments(video_path)

    fig, (ax_frames, ax_scores) = plt.subplots(
        2, 1, figsize=(14, 5),
        gridspec_kw={'height_ratios': [1, 2.5]},
    )
    fig.subplots_adjust(hspace=0.05)

    # Top: keyframes
    ax_frames.set_xlim(0, len(scores))
    ax_frames.set_ylim(0, 1)
    ax_frames.axis('off')

    frame_width = len(scores) / len(keyframes)
    for i, (frame, idx) in enumerate(zip(keyframes, kf_indices)):
        snippet_pos = idx / NUM_FRAMES
        x_center = np.clip(snippet_pos, frame_width / 2, len(scores) - frame_width / 2)
        extent = [x_center - frame_width / 2, x_center + frame_width / 2, 0.05, 0.95]
        ax_frames.imshow(frame, extent=extent, aspect='auto')

    # Bottom: GT shading + anomaly score curve
    for start_frame, end_frame in gt_segments:
        s = start_frame / NUM_FRAMES
        e = end_frame / NUM_FRAMES
        ax_scores.axvspan(s, e, alpha=0.25, color='#FF6B6B', label='GT Anomaly')

    x = np.arange(len(scores))
    ax_scores.plot(x, scores, color='#4ECDC4', linewidth=1.5)
    ax_scores.set_xlim(0, len(scores))
    ax_scores.set_ylim(0, 1)
    ax_scores.set_ylabel('Anomaly Score', fontsize=12)
    ax_scores.set_xlabel('Snippet Index', fontsize=12)
    ax_scores.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Deduplicate legend
    handles, labels = ax_scores.get_legend_handles_labels()
    if handles:
        ax_scores.legend([handles[0]], [labels[0]], loc='upper right', fontsize=10)

    video_name = os.path.basename(video_path)
    fig.suptitle(video_name, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_result_from_feature(scores, video_name, output_path):
    """Plot anomaly scores with GT shading (no keyframes)."""
    gt_segments = load_gt_segments(video_name)

    fig, ax = plt.subplots(figsize=(14, 4))

    # GT shading
    for start_frame, end_frame in gt_segments:
        s = start_frame / NUM_FRAMES
        e = end_frame / NUM_FRAMES
        ax.axvspan(s, e, alpha=0.25, color='#FF6B6B', label='GT Anomaly')

    # Score curve
    x = np.arange(len(scores))
    ax.plot(x, scores, color='#4ECDC4', linewidth=1.5)
    ax.set_xlim(0, len(scores))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_xlabel('Snippet Index', fontsize=12)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend([handles[0]], [labels[0]], loc='upper right', fontsize=10)

    fig.suptitle(video_name, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer anomaly scores')
    parser.add_argument('--mode', type=str, required=True, choices=['video', 'feature'])
    args = parser.parse_args()

    output_dir = os.path.join(OUTPUT_DIR, args.mode)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vad_args = ucf_option.parser.parse_args([])

    # Load VAD model
    print("Loading VAD model...")
    vad_model = VadInternVL(
        vad_args.visual_length, vad_args.visual_width, vad_args.visual_head,
        vad_args.visual_layers, vad_args.attn_window, device
    ).to(device).eval()
    vad_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    if args.mode == 'video':
        base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
        output = OUTPUT_PATH or os.path.join(output_dir, f"{base_name}_anomaly_video.png")

        # Load InternVL
        print("Loading InternVL...")
        internvl_model = AutoModel.from_pretrained(
            INTERNVL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device).eval()
        transform = build_transform(input_size=448)

        # Extract features
        print("Extracting features from video...")
        frames, fps, total_frames = load_all_frames(VIDEO_PATH)
        print(f"  {len(frames)} snippets from {total_frames} frames ({fps:.1f} fps)")
        features = extract_features(internvl_model, transform, frames, device)

        # Infer
        print("Running anomaly detection...")
        scores = infer(vad_model, features, device)
        print(f"  Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        plot_result(scores, VIDEO_PATH, output)

    elif args.mode == 'feature':
        feat_name = os.path.splitext(os.path.basename(FEATURE_PATH))[0]
        base_name = feat_name.split('__')[0] if '__' in feat_name else feat_name
        output = OUTPUT_PATH or os.path.join(output_dir, f"{base_name}_anomaly_feature.png")

        print(f"Loading features from {FEATURE_PATH}...")
        features = np.load(FEATURE_PATH)
        print(f"  {features.shape[0]} snippets, dim={features.shape[1]}")

        print("Running anomaly detection...")
        scores = infer(vad_model, features, device)
        print(f"  Scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        video_name = base_name + '.mp4'
        plot_result_from_feature(scores, video_name, output)
