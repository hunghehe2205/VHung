import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from src.utils.tools import process_feat, process_split


class UCFDataset(data.Dataset):
    def __init__(self, feat_dim: int, file_path: str, test_mode: bool, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.feat_dim = feat_dim
        self.test_mode = test_mode
        self.normal = normal
        if normal and not test_mode:
            self.df = self.df.loc[self.df['label'] == 'Normal'].reset_index(drop=True)
        elif not test_mode:
            self.df = self.df.loc[self.df['label'] != 'Normal'].reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        feature = np.load(self.df.loc[index]['path'])
        if not self.test_mode:
            feature, feat_length = process_feat(feature, self.feat_dim)
        else:
            feature, feat_length = process_split(feature, self.feat_dim)

        feature = torch.tensor(feature)
        label = self.df.loc[index]['label']
        return feature, label, feat_length
