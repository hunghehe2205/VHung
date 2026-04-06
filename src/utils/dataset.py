import os
import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools


class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict,
                 normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        path = self.df.loc[index]['path']
        clip_feature = np.load(path)

        if not self.test_mode:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']

        return clip_feature, clip_label, clip_length
