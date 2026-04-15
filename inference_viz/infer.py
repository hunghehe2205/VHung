import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import CLIPVAD
from utils.tools import get_batch_mask, get_prompt_text, pad, process_split

from inference_viz.sampling import density_aware_sample
from inference_viz.viz import compose_viz, load_frames, parse_ucf_gt


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism',
}

CLIP_LEN = 16


# ---------------------------------------------------------------------------
# Hardcoded paths — edit here before running.
# Everything else (CLIPVAD architecture + sampling knobs) stays on the CLI.
# ---------------------------------------------------------------------------
MODEL_PATH = 'model/model_ucf.pth'
FEATURE_ROOT = '/home/emogenai4e/emo/VHung/UCFClipFeatures'
RAW_VIDEO_ROOT = '/home/emogenai4e/emo/Hung_data/UCF_Crime'
TEST_LIST = 'list/ucf_CLIP_rgbtest.csv'
GT_PATH = 'data/Temporal_Anomaly_Annotation_for_Testing_Videos.txt'
OUTPUT_DIR = 'ucf_infer_results'


def parse_args():
    p = argparse.ArgumentParser(description='UCF-Crime inference + visualization')
    p.add_argument('--video', default=None,
                   help='Single video id to run, e.g. Arrest001_x264. Omit = full test set.')
    p.add_argument('--select-frames', type=int, default=16)
    p.add_argument('--tau', type=float, default=0.1)

    # CLIPVAD construction (mirrors src/option.py defaults)
    p.add_argument('--embed-dim', type=int, default=512)
    p.add_argument('--visual-length', type=int, default=256)
    p.add_argument('--visual-width', type=int, default=512)
    p.add_argument('--visual-head', type=int, default=1)
    p.add_argument('--visual-layers', type=int, default=2)
    p.add_argument('--attn-window', type=int, default=8)
    p.add_argument('--prompt-prefix', type=int, default=10)
    p.add_argument('--prompt-postfix', type=int, default=10)
    p.add_argument('--classes-num', type=int, default=14)
    return p.parse_args()


def prepare_visual(feat, maxlen):
    """Pack a feature array of shape [T, D] into batched model input.

    Returns (visual_np, clip_length) where visual_np is [num_chunks, maxlen, D].
    """
    clip_length = feat.shape[0]
    if clip_length < maxlen:
        visual_np = pad(feat, maxlen)[None, ...]  # [1, maxlen, D]
    else:
        split_feat, _ = process_split(feat, maxlen)  # [N, maxlen, D]
        visual_np = split_feat
    return visual_np, clip_length


def build_lengths(clip_length, maxlen):
    num_chunks = int(clip_length / maxlen) + 1
    lengths = torch.zeros(num_chunks, dtype=torch.int)
    remaining = clip_length
    for j in range(num_chunks):
        if remaining > maxlen:
            lengths[j] = maxlen
            remaining -= maxlen
        else:
            lengths[j] = remaining
    return lengths


def run_model(model, feat, maxlen, prompt_text, device):
    visual_np, clip_length = prepare_visual(feat, maxlen)
    visual = torch.tensor(visual_np).to(device)
    lengths = build_lengths(clip_length, maxlen)
    padding_mask = get_batch_mask(lengths, maxlen).to(device)

    _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
    logits1 = logits1.reshape(-1, logits1.shape[-1])
    logits2 = logits2.reshape(-1, logits2.shape[-1])

    prob1 = torch.sigmoid(logits1[:clip_length].squeeze(-1)).detach().cpu().numpy()
    prob2 = (1 - logits2[:clip_length].softmax(dim=-1)[:, 0]).detach().cpu().numpy()
    return prob1, prob2


def gt_to_snippet_ranges(gt, num_snippets):
    ranges = []
    for key in ('seg1', 'seg2'):
        seg = gt.get(key)
        if seg is None:
            continue
        s = max(0, seg[0] // CLIP_LEN)
        e = min(num_snippets - 1, seg[1] // CLIP_LEN)
        if e >= s:
            ranges.append((int(s), int(e)))
    return ranges


def process_video(model, row, args, prompt_text, gt_map, device):
    label = row['label']
    feature_name = os.path.basename(row['path'])
    video_id = feature_name.split('__')[0]
    feature_path = os.path.join(FEATURE_ROOT, label, feature_name)

    feat = np.load(feature_path)
    with torch.no_grad():
        prob1, prob2 = run_model(model, feat, args.visual_length, prompt_text, device)

    sampled_snippet_idxs = density_aware_sample(prob1, args.select_frames, args.tau)
    frame_indices = [idx * CLIP_LEN + CLIP_LEN // 2 for idx in sampled_snippet_idxs]

    video_path = os.path.join(RAW_VIDEO_ROOT, label, f"{video_id}.mp4")
    frames, total_frames, fps = load_frames(video_path, frame_indices)
    frame_indices = [min(total_frames - 1, max(0, f)) for f in frame_indices]

    gt = gt_map.get(video_id, {'class': label, 'seg1': None, 'seg2': None})
    num_snippets = len(prob1)
    gt_snippet_ranges = gt_to_snippet_ranges(gt, num_snippets)

    out_dir = os.path.join(OUTPUT_DIR, video_id)
    compose_viz(
        video_id, label, frames, prob1,
        sampled_snippet_idxs, gt_snippet_ranges,
        os.path.join(out_dir, 'viz.png'),
    )

    metadata = {
        'video_id': video_id,
        'class': label,
        'video_path': video_path,
        'total_frames': int(total_frames),
        'fps': float(fps),
        'num_snippets': int(num_snippets),
        'anomaly_scores_binary': [float(x) for x in prob1],
        'anomaly_scores_multiclass': [float(x) for x in prob2],
        'sampled_snippet_indices': [int(x) for x in sampled_snippet_idxs],
        'sampled_frame_indices': [int(x) for x in frame_indices],
        'gt': {'seg1': gt.get('seg1'), 'seg2': gt.get('seg2')},
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp, indent=2)

    return out_dir


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, device,
    )
    state = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    prompt_text = get_prompt_text(LABEL_MAP)
    df = pd.read_csv(TEST_LIST)
    gt_map = parse_ucf_gt(GT_PATH)

    if args.video:
        df = df[df['path'].str.contains(f"/{args.video}__")].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"Video '{args.video}' not found in {TEST_LIST}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i in tqdm(range(len(df)), desc='Inference'):
        try:
            out_dir = process_video(model, df.iloc[i], args, prompt_text, gt_map, device)
            tqdm.write(f"[{i+1}/{len(df)}] -> {out_dir}")
        except Exception as e:
            tqdm.write(f"[{i+1}/{len(df)}] FAILED {df.iloc[i]['path']}: {e}")


if __name__ == '__main__':
    main()
