import json
import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools

CLIP_LEN = 16


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False, hivau_json: str = None,
                 boundary_margin: int = 3, label_smooth: float = 0.05):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.boundary_margin = boundary_margin
        self.label_smooth = label_smooth
        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        # Load HIVAU annotations
        self.hivau = None
        if hivau_json and os.path.exists(hivau_json):
            with open(hivau_json) as f:
                self.hivau = json.load(f)

    def _get_frame_gt(self, path, feat_len):
        """Convert HIVAU temporal events to feature-level soft GT.

        Boundary frames get smoothed scores (0→1 ramp-up, 1→0 ramp-down)
        to account for annotation noise at event edges.
        """
        if self.hivau is None:
            return np.zeros(feat_len, dtype=np.float32)

        video_name = os.path.basename(path).rsplit('__', 1)[0]

        if video_name not in self.hivau:
            return np.zeros(feat_len, dtype=np.float32)

        info = self.hivau[video_name]
        fps = info['fps']
        events = info['events']

        gt = np.zeros(feat_len, dtype=np.float32)
        margin = self.boundary_margin  # smooth boundary width in feature indices
        for start_sec, end_sec in events:
            start_idx = int(start_sec * fps / CLIP_LEN)
            end_idx = int(end_sec * fps / CLIP_LEN) + 1
            start_idx = max(0, min(start_idx, feat_len))
            end_idx = max(0, min(end_idx, feat_len))
            # Core region: full confidence
            gt[start_idx:end_idx] = 1.0
            # Ramp-up before start
            for m in range(1, margin + 1):
                idx = start_idx - m
                if 0 <= idx < feat_len:
                    gt[idx] = max(gt[idx], 1.0 - m / (margin + 1))
            # Ramp-down after end
            for m in range(margin):
                idx = end_idx + m
                if 0 <= idx < feat_len:
                    gt[idx] = max(gt[idx], 1.0 - (m + 1) / (margin + 1))
        # Label smoothing: clip to [smooth_val, 1 - smooth_val]
        smooth = self.label_smooth
        gt = gt * (1 - 2 * smooth) + smooth
        return gt

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        clip_feature = np.load(path)
        raw_len = clip_feature.shape[0]

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        # Frame-level GT (only for training with HIVAU)
        if not self.test_mode and self.hivau is not None:
            frame_gt = self._get_frame_gt(path, raw_len)
            # Align to clip_dim (same as process_feat: pad or uniform_extract)
            if raw_len > self.clip_dim:
                frame_gt = tools.uniform_extract(frame_gt.reshape(-1, 1), self.clip_dim, avg=True).reshape(-1)
                frame_gt = (frame_gt > 0.5).astype(np.float32)
            else:
                frame_gt = np.pad(frame_gt, (0, self.clip_dim - raw_len), mode='constant')
            frame_gt = torch.tensor(frame_gt)
        else:
            frame_gt = torch.zeros(self.clip_dim)

        return clip_feature, clip_label, clip_length, frame_gt
