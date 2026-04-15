"""Precompute scalar pos_weight for L_frame_bce on the training set.

Usage:
    python src/utils/compute_pos_weight.py \
        --train-list list/ucf_CLIP_rgb.csv \
        --train-json HIVAU-70k-NEW/ucf_database_train_filtered.json \
        --clip-dim 256 \
        --out list/pos_weight_bin.npy
"""
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import UCFDataset


LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv')
    p.add_argument('--train-json',
                   default='HIVAU-70k-NEW/ucf_database_train_filtered.json')
    p.add_argument('--clip-dim', type=int, default=256)
    p.add_argument('--out', default='list/pos_weight_bin.npy')
    args = p.parse_args()

    # Iterate over ALL training samples (both normal + anomaly partitions of CSV)
    all_ds = [
        UCFDataset(args.clip_dim, args.train_list, test_mode=False,
                   label_map=LABEL_MAP, normal=True,
                   json_path=args.train_json),
        UCFDataset(args.clip_dim, args.train_list, test_mode=False,
                   label_map=LABEL_MAP, normal=False,
                   json_path=args.train_json),
    ]

    n_pos = 0
    n_neg = 0
    for ds in all_ds:
        for i in tqdm(range(len(ds)), desc='scan'):
            _, _, y_bin, length = ds[i]
            # Only count valid frames (within length)
            valid = y_bin[:length]
            n_pos += int((valid > 0.5).sum().item())
            n_neg += int((valid <= 0.5).sum().item())

    pos_weight = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    print(f'n_pos={n_pos}  n_neg={n_neg}  pos_weight={pos_weight:.4f}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, np.array([pos_weight], dtype=np.float32))
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
