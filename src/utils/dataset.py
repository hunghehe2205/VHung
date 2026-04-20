import os
import json
import re
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools


_VIDEO_NAME_RE = re.compile(r'(.*?)__\d+$')


def _parse_video_name(path: str) -> str:
    """Turn '.../Abuse001_x264__3.npy' -> 'Abuse001_x264'."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = _VIDEO_NAME_RE.match(base)
    return m.group(1) if m else base


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool,
                 label_map: dict, normal: bool = False,
                 json_path: str = None):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal

        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

        self._events = {}
        if json_path is not None and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self._events = json.load(f)

    def __len__(self):
        return self.df.shape[0]

    def _lookup_events(self, video_name):
        """Return (events_sec, fps) or ([], 30.0) if not present."""
        entry = self._events.get(video_name)
        if entry is None:
            return [], 30.0
        return list(entry.get('events', [])), float(entry.get('fps', 30.0))

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        clip_feature = np.load(path)
        n_features_raw = clip_feature.shape[0]

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        if not self.test_mode:
            video_name = _parse_video_name(path)
            events_sec, fps = self._lookup_events(video_name)
            y_bin_np = tools.build_frame_labels(
                events_sec=events_sec,
                fps=fps,
                n_features=n_features_raw,
                clip_len=16,
                target_len=self.clip_dim,
            )
            y_soft_np = tools.build_gaussian_target(y_bin_np, sigma=2.0)
            y_bin = torch.from_numpy(y_bin_np)    # [clip_dim] float32
            y_soft = torch.from_numpy(y_soft_np)  # [clip_dim] float32
            return clip_feature, clip_label, y_bin, y_soft, clip_length

        # test mode — keep legacy 3-tuple for compatibility with test.py
        return clip_feature, clip_label, clip_length
