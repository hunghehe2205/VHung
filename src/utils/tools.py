import torch
import numpy as np


def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    for text in texts:
        label_vector = torch.zeros(len(prompt_text))
        if text in label_map:
            label_text = label_map[text]
            label_vector[prompt_text.index(label_text)] = 1
        label_vector = label_vector.unsqueeze(0)
        label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    return label_vectors


def get_prompt_text(label_map: dict):
    return list(label_map.values())


def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.zeros(batch_size, maxlen)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    return mask.bool()


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
    if feat.shape[0] <= min_len:
        return np.pad(feat, ((0, min_len - feat.shape[0]), (0, 0)), mode='constant', constant_values=0)
    return feat


def process_feat(feat, length):
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    else:
        split_num = int(clip_length / length) + 1
        for i in range(split_num):
            chunk = feat[i * length:i * length + length, :]
            if i < split_num - 1:
                chunk = chunk.reshape(1, length, feat.shape[1])
            else:
                chunk = pad(chunk, length).reshape(1, length, feat.shape[1])
            if i == 0:
                split_feat = chunk
            else:
                split_feat = np.concatenate([split_feat, chunk], axis=0)
        return split_feat, clip_length
