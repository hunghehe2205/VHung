import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d


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


def events_to_clip_mask(events, n_frames, fps, num_clips):
    """Convert HIVAU event intervals (seconds) to a binary clip-level mask."""
    duration = n_frames / fps
    mask = np.zeros(num_clips, dtype=np.float32)
    for start_sec, end_sec in events:
        start_sec = max(0.0, start_sec)
        end_sec = min(duration, end_sec)
        if start_sec >= end_sec:
            continue
        start_clip = int(np.floor(start_sec / duration * num_clips))
        end_clip = int(np.ceil(end_sec / duration * num_clips))
        start_clip = max(0, min(start_clip, num_clips))
        end_clip = max(0, min(end_clip, num_clips))
        mask[start_clip:end_clip] = 1.0
    return mask


def smooth_mask(mask, sigma=1.5, clamp_min=0.05, clamp_max=0.95):
    """Apply Gaussian smoothing to mask and clamp to soft label range."""
    smoothed = gaussian_filter1d(mask.astype(np.float32), sigma=sigma)
    smoothed = np.clip(smoothed, clamp_min, clamp_max)
    return smoothed


def process_mask(mask, length):
    """Resample or pad mask to fixed length, matching process_feat logic."""
    T = mask.shape[0]
    if T > length:
        r = np.linspace(0, T, length + 1, dtype=np.int32)
        resampled = np.array([mask[r[i]:r[i+1]].max() if r[i] != r[i+1] else mask[r[i]]
                              for i in range(length)], dtype=np.float32)
        return resampled
    elif T < length:
        return np.pad(mask, (0, length - T), mode='constant', constant_values=0.0)
    else:
        return mask.copy()
