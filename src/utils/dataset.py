import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torch.utils.data import DataLoader

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
        feature = feature / (feature.norm(dim=-1, keepdim=True) + 1e-6)
        label = self.df.loc[index]['label']
        return feature, label, feat_length


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from torch.utils.data import DataLoader

    train_list = 'list/ucf_intern_rgb.csv'
    test_list = 'list/ucf_intern_rgbtest.csv'
    feat_dim = 256

    # 1. Train dataset - Normal
    normal_ds = UCFDataset(feat_dim, train_list, test_mode=False, normal=True)
    print(f'[Train Normal] samples: {len(normal_ds)}')
    feat, label, length = normal_ds[0]
    print(f'  feat: {feat.shape}, label: {label}, length: {length}')

    # 2. Train dataset - Anomaly
    anomaly_ds = UCFDataset(feat_dim, train_list, test_mode=False, normal=False)
    print(f'[Train Anomaly] samples: {len(anomaly_ds)}')
    feat, label, length = anomaly_ds[0]
    print(f'  feat: {feat.shape}, label: {label}, length: {length}')

    # 3. Test dataset
    test_ds = UCFDataset(feat_dim, test_list, test_mode=True)
    print(f'[Test] samples: {len(test_ds)}')
    feat, label, length = test_ds[0]
    print(f'  feat: {feat.shape}, label: {label}, length: {length}')

    # 4. DataLoader batching (train)
    normal_loader = DataLoader(normal_ds, batch_size=4, shuffle=True, drop_last=True)
    batch_feat, batch_label, batch_length = next(iter(normal_loader))
    print(f'[Train Batch] feat: {batch_feat.shape}, labels: {batch_label}, lengths: {batch_length}')

    # 5. DataLoader batching (test, batch_size=1)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    batch_feat, batch_label, batch_length = next(iter(test_loader))
    print(f'[Test Batch] feat: {batch_feat.shape}, label: {batch_label}, length: {batch_length}')

    print('\nAll tests passed!')
