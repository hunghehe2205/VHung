import os
import re
import json
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False, hivau_json_path: str = None, smooth_sigma: float = 1.5):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        self.smooth_sigma = smooth_sigma
        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        self.hivau = {}
        if hivau_json_path and os.path.exists(hivau_json_path):
            with open(hivau_json_path) as f:
                self.hivau = json.load(f)

    def _extract_video_name(self, path):
        basename = os.path.basename(path)
        return re.sub(r'__\d+\.npy$', '', basename)

    def _create_mask(self, path, num_clips):
        video_name = self._extract_video_name(path)
        if video_name in self.hivau:
            entry = self.hivau[video_name]
            raw_mask = tools.events_to_clip_mask(
                entry['events'], entry['n_frames'], entry['fps'], num_clips
            )
        else:
            raw_mask = np.zeros(num_clips, dtype=np.float32)

        processed = tools.process_mask(raw_mask, self.clip_dim)
        soft_mask = tools.smooth_mask(processed, sigma=self.smooth_sigma)
        return soft_mask, processed

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        clip_feature = np.load(path)
        num_clips_raw = clip_feature.shape[0]

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            soft_mask, raw_mask = self._create_mask(path, num_clips_raw)
            clip_feature = torch.tensor(clip_feature)
            soft_mask = torch.tensor(soft_mask, dtype=torch.float32)
            raw_mask = torch.tensor(raw_mask, dtype=torch.float32)
            clip_label = self.df.loc[index]['label']
            return clip_feature, clip_label, clip_length, soft_mask, raw_mask
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            clip_feature = torch.tensor(clip_feature)
            clip_label = self.df.loc[index]['label']
            return clip_feature, clip_label, clip_length
