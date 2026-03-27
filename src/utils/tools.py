import torch
import numpy as np


def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    return mask.bool()


def random_extract(feat, t_max):
    r = np.random.randint(feat.shape[0] - t_max)
    return feat[r:r + t_max, :]


def uniform_extract(feat, t_max, avg=True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    if avg:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        r = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
    return new_feat


def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=False):
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), feat.shape[0]


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            if i == 0:
                split_feat = feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])
            elif i < split_num - 1:
                split_feat = np.concatenate([split_feat, feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])], axis=0)
            else:
                split_feat = np.concatenate([split_feat, pad(feat[i * length:i * length + length, :], length).reshape(1, length, feat.shape[1])], axis=0)
        return split_feat, clip_length
