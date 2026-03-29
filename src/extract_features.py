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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from crop_intern import video_crop, build_transform

VIDEO_DIR = '/home/emogenai4e/emo/Hung_data/UCF_Crime'
TRAIN_TXT = '/home/emogenai4e/emo/Hung_data/Anomaly_Train.txt'
TEST_TXT = '/home/emogenai4e/emo/Hung_data/Anomaly_Test.txt'
TRAIN_OUTPUT_DIR = '/home/emogenai4e/emo/Hung_data/ucf_internvl_train_feature'
TEST_OUTPUT_DIR  = '/home/emogenai4e/emo/Hung_data/ucf_internvl_test_feature'
BATCH_SIZE = 32
TRAIN_CROPS = list(range(10))  # 0-9
TEST_CROPS = [0]               # center crop only
NUM_FRAMES = 16


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    """Uniform sampling num_frames từ video, trả về numpy [T, H, W, 3] BGR."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
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
    """Load video list from Anomaly_Train.txt or Anomaly_Test.txt."""
    txt_path = TRAIN_TXT if mode == 'train' else TEST_TXT
    files = [line.strip() for line in open(txt_path) if line.strip()]
    videos = []
    for entry in files:
        label = entry.split('/')[0]
        video_path = os.path.join(VIDEO_DIR, entry)
        out_label = 'Normal' if 'Normal' in label else label
        videos.append((video_path, out_label))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA not available! This script requires GPU."
    device = "cuda"
    print(f"[GPU] {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print("[1/3] Loading InternVL model (ppxin321/HolmesVAU-2B)...")
    model = AutoModel.from_pretrained(
        "ppxin321/HolmesVAU-2B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[1/3] Model loaded. GPU memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    transform = build_transform(input_size=448)

    print("[2/3] Scanning video list...")
    crops = TRAIN_CROPS if args.mode == 'train' else TEST_CROPS
    output_dir = TRAIN_OUTPUT_DIR if args.mode == 'train' else TEST_OUTPUT_DIR

    videos = get_video_list(args.mode)
    print(f"[2/3] Found {len(videos)} videos | Mode: {args.mode} | Crops per video: {len(crops)}")

    print("[3/3] Extracting features...")

    pbar = tqdm(videos, desc=f"Extracting [{args.mode}]")
    for video_path, label in pbar:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_subdir = os.path.join(output_dir, label)
        os.makedirs(out_subdir, exist_ok=True)

        # Check if all crops already exist
        all_exist = all(
            os.path.exists(os.path.join(out_subdir, f"{video_name}__{c}.npy"))
            for c in crops
        )
        if all_exist:
            pbar.set_postfix(video=video_name, status="SKIP")
            continue

        frames = load_video_frames(video_path)
        if frames is None:
            pbar.set_postfix(video=video_name, status="FAIL")
            continue

        for crop_id in crops:
            out_path = os.path.join(out_subdir, f"{video_name}__{crop_id}.npy")
            if os.path.exists(out_path):
                continue
            feat = extract_video_features(model, transform, frames, crop_id, device)
            np.save(out_path, feat)

        pbar.set_postfix(video=video_name, label=label, frames=frames.shape[0])

    print("Done.")


if __name__ == '__main__':
    main()
