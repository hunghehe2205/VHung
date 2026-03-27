"""
Extract InternVL CLS token features cho toàn bộ UCF-Crime dataset.

Usage:
    python src/extract_features.py --mode train
    python src/extract_features.py --mode test

Videos:  /home/emogenai4e/emo/Hung_data/UCF_Crime/{Label}/video.avi
Output:  /home/emogenai4e/emo/Hung_data/internvl_feature/{Label}/video__0.npy ~ __9.npy
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crop_intern import video_crop, build_transform

VIDEO_DIR = '/home/emogenai4e/emo/Hung_data/UCF_Crime'
OUTPUT_DIR = '/home/emogenai4e/emo/Hung_data/internvl_feature'
BATCH_SIZE = 8
TRAIN_CROPS = list(range(10))  # 0-9
TEST_CROPS = [5]               # center crop only


def load_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    return np.array(frames)


def extract_video_features(model, transform, frames, crop_id, device):
    cropped = video_crop(frames, crop_id)  # [T, 448, 448, 3] RGB
    features = []
    with torch.no_grad():
        for i in range(0, cropped.shape[0], BATCH_SIZE):
            batch = cropped[i:i + BATCH_SIZE]
            imgs = torch.stack([transform(Image.fromarray(f)) for f in batch])
            imgs = imgs.to(torch.bfloat16).to(device)
            vit_embeds = model.vision_model(
                pixel_values=imgs,
                output_hidden_states=False,
                return_dict=True,
            ).last_hidden_state
            cls_token = vit_embeds[:, 0, :]  # [B, 1024]
            features.append(cls_token.cpu().float())
    return torch.cat(features, dim=0).numpy()  # [T, 1024]


def get_video_list(mode):
    """Collect all video paths grouped by label."""
    videos = []
    for label in sorted(os.listdir(VIDEO_DIR)):
        label_dir = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        is_normal = 'Normal' in label
        # Train: anomaly folders + Training_Normal_Videos_Anomaly
        # Test: Testing_Normal_Videos_Anomaly + anomaly test videos
        if mode == 'train':
            if label == 'Testing_Normal_Videos_Anomaly':
                continue
        else:
            if label == 'Training_Normal_Videos_Anomaly':
                continue

        out_label = 'Normal' if is_normal else label

        for f in sorted(os.listdir(label_dir)):
            if f.endswith(('.avi', '.mp4')):
                videos.append((os.path.join(label_dir, f), out_label))

    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading InternVL model...")
    model = AutoModel.from_pretrained(
        "ppxin321/HolmesVAU-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    transform = build_transform(input_size=448)
    crops = TRAIN_CROPS if args.mode == 'train' else TEST_CROPS
    videos = get_video_list(args.mode)

    print(f"Mode: {args.mode} | Videos: {len(videos)} | Crops per video: {len(crops)}")

    for idx, (video_path, label) in enumerate(videos):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_subdir, exist_ok=True)

        # Check if all crops already exist
        all_exist = all(
            os.path.exists(os.path.join(out_subdir, f"{video_name}__{c}.npy"))
            for c in crops
        )
        if all_exist:
            print(f"[{idx + 1}/{len(videos)}] SKIP {video_name} (already done)")
            continue

        frames = load_all_frames(video_path)
        if frames is None:
            print(f"[{idx + 1}/{len(videos)}] FAIL {video_path} (cannot read)")
            continue

        for crop_id in crops:
            out_path = os.path.join(out_subdir, f"{video_name}__{crop_id}.npy")
            if os.path.exists(out_path):
                continue
            feat = extract_video_features(model, transform, frames, crop_id, device)
            np.save(out_path, feat)

        print(f"[{idx + 1}/{len(videos)}] {video_name} | {label} | frames={frames.shape[0]}")

    print("Done.")


if __name__ == '__main__':
    main()
