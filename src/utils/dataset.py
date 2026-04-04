import json
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torch.utils.data import DataLoader

from src.utils.tools import process_feat, process_split


def build_frame_gt(events, n_frames, fps, feat_length, target_length,
                   normal_target=0.1, is_normal=False, sigma=3.0):
    """Build snippet-level GT from HIVAU temporal events with Gaussian boundary smoothing.

    After uniform_extract, output always has target_length snippets.
    Snippet i corresponds to time i/target_length * duration.
    So map seconds -> index using target_length directly (not raw feat_length).
    For padded videos (raw < target), zero out padding region.
    """
    frame_gt = torch.full((target_length,), normal_target if is_normal else 0.0)

    if is_normal or not events:
        return frame_gt

    duration = n_frames / fps
    indices = torch.arange(target_length, dtype=torch.float32)

    for start_sec, end_sec in events:
        # Map directly to extracted space (target_length snippets)
        start_idx = (start_sec / duration) * target_length
        end_idx = (end_sec / duration) * target_length

        ramp_up = torch.sigmoid((indices - start_idx) / sigma)
        ramp_down = torch.sigmoid((end_idx - indices) / sigma)
        event_gt = ramp_up * ramp_down

        frame_gt = torch.max(frame_gt, event_gt)

    # Zero out padding region for short videos
    if feat_length < target_length:
        frame_gt[feat_length:] = 0.0

    return frame_gt


class UCFDataset(data.Dataset):
    def __init__(self, feat_dim: int, file_path: str, test_mode: bool,
                 normal: bool = False, hivau_path: str = None,
                 normal_target: float = 0.1, sigma: float = 3.0):
        self.df = pd.read_csv(file_path)
        self.feat_dim = feat_dim
        self.test_mode = test_mode
        self.normal = normal
        self.normal_target = normal_target
        self.sigma = sigma

        # Load HIVAU annotation
        self.hivau = None
        if hivau_path and not test_mode and os.path.exists(hivau_path):
            with open(hivau_path) as f:
                self.hivau = json.load(f)

        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

    def _get_video_name(self, path):
        """Extract video name from feature path. e.g. .../Abuse001_x264__0.npy -> Abuse001_x264"""
        basename = os.path.basename(path)  # Abuse001_x264__0.npy
        return basename.split('__')[0]     # Abuse001_x264

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        feature = np.load(self.df.loc[index]['path'])
        raw_length = feature.shape[0]

        if not self.test_mode:
            feature, feat_length = process_feat(feature, self.feat_dim)
        else:
            feature, feat_length = process_split(feature, self.feat_dim)

        feature = torch.tensor(feature)
        label = self.df.loc[index]['label']

        # Build frame-level GT from HIVAU (anomaly videos only)
        frame_gt = torch.full((self.feat_dim,), -1.0)  # -1 = no annotation
        if self.hivau is not None:
            video_name = self._get_video_name(self.df.loc[index]['path'])
            is_normal = (label == 'Normal')

            if is_normal:
                # Normal video: all frames target = 0
                frame_gt = torch.zeros(self.feat_dim)
            elif video_name in self.hivau and self.hivau[video_name].get('events'):
                # Anomaly video with HIVAU events
                info = self.hivau[video_name]
                frame_gt = build_frame_gt(
                    events=info['events'],
                    n_frames=info['n_frames'],
                    fps=info['fps'],
                    feat_length=raw_length,
                    target_length=self.feat_dim,
                    normal_target=0.0,
                    is_normal=False,
                    sigma=self.sigma
                )

        return feature, label, feat_length, frame_gt


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from torch.utils.data import DataLoader

    train_list = 'list/ucf_intern_rgb.csv'
    test_list = 'list/ucf_intern_rgbtest.csv'
    hivau_path = 'HIVAU-70k-NEW/ucf_database_train.json'
    feat_dim = 256

    # 1. Train dataset - Normal
    normal_ds = UCFDataset(feat_dim, train_list, test_mode=False, normal=True, hivau_path=hivau_path)
    print(f'[Train Normal] samples: {len(normal_ds)}')
    feat, label, length, frame_gt = normal_ds[0]
    print(f'  feat: {feat.shape}, label: {label}, length: {length}')
    print(f'  frame_gt: {frame_gt.shape}, min={frame_gt.min():.2f}, max={frame_gt.max():.2f}')

    # 2. Train dataset - Anomaly
    anomaly_ds = UCFDataset(feat_dim, train_list, test_mode=False, normal=False, hivau_path=hivau_path)
    print(f'[Train Anomaly] samples: {len(anomaly_ds)}')
    feat, label, length, frame_gt = anomaly_ds[0]
    print(f'  feat: {feat.shape}, label: {label}, length: {length}')
    print(f'  frame_gt: {frame_gt.shape}, min={frame_gt.min():.2f}, max={frame_gt.max():.2f}')
    print(f'  anomaly_ratio: {(frame_gt == 1.0).float().mean():.2%}')

    # 3. Test dataset
    test_ds = UCFDataset(feat_dim, test_list, test_mode=True)
    print(f'[Test] samples: {len(test_ds)}')
    feat, label, length, frame_gt = test_ds[0]
    print(f'  feat: {feat.shape}, label: {label}, length: {length}')
    print(f'  frame_gt: {frame_gt.shape}, min={frame_gt.min():.2f} (should be -1, no annotation)')

    # 4. DataLoader batching (train)
    normal_loader = DataLoader(normal_ds, batch_size=4, shuffle=True, drop_last=True)
    batch_feat, batch_label, batch_length, batch_gt = next(iter(normal_loader))
    print(f'[Train Batch] feat: {batch_feat.shape}, labels: {batch_label}, lengths: {batch_length}')
    print(f'  batch_gt: {batch_gt.shape}')

    print('\nAll tests passed!')
