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


def build_frame_labels(events_sec, fps, n_features, clip_len=16, target_len=256):
    """
    Build binary per-feature anomaly labels of shape [target_len].

    events_sec: iterable of (start_sec, end_sec) tuples (possibly empty).
    fps: frames-per-second of the source video.
    n_features: number of features actually present in the loaded file.
    clip_len: frames per feature (default 16, matching feature extractor stride).
    target_len: final length after pad/truncate (e.g. visual_length=256).

    Rule: any-overlap between feature window [i*clip_len, (i+1)*clip_len)
          and any event (in frame space) marks label 1.

    When n_features > target_len, downsamples via MAX (preserves any-overlap).
    When n_features < target_len, pads with zeros (normal).
    """
    events_frame = [(float(s) * fps, float(e) * fps) for s, e in events_sec]
    y_raw = np.zeros(n_features, dtype=np.float32)
    for i in range(n_features):
        ws = i * clip_len
        we = (i + 1) * clip_len
        for s, e in events_frame:
            if we > s and ws < e:  # overlap
                y_raw[i] = 1.0
                break

    if n_features >= target_len:
        # Uniform-MAX downsample to target_len (matches uniform_extract layout)
        r = np.linspace(0, n_features, target_len + 1, dtype=np.int32)
        y = np.zeros(target_len, dtype=np.float32)
        for i in range(target_len):
            lo, hi = r[i], r[i + 1]
            if lo == hi:
                y[i] = y_raw[lo] if lo < n_features else 0.0
            else:
                y[i] = y_raw[lo:hi].max()
        return y
    else:
        y = np.zeros(target_len, dtype=np.float32)
        y[:n_features] = y_raw
        return y


def build_gaussian_target(y_bin, sigma=2.0):
    """Peak-normalized Gaussian smoothing of a binary [T] label → soft [0,1]
    target. Inside-event positions stay at 1.0 (max of kernel is 1); outside
    positions within 3σ of a boundary get exponential falloff, further out is
    zero. Used as the TCN-head BCE target so the head gets a sharper gradient
    near GT edges than a plain step function would provide."""
    y_bin = np.asarray(y_bin, dtype=np.float32)
    T = y_bin.shape[0]
    if sigma <= 0 or y_bin.sum() == 0:
        return y_bin.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    padded = np.pad(y_bin, radius, mode='constant')
    out = np.zeros(T, dtype=np.float32)
    for t in range(T):
        out[t] = float((padded[t:t + 2 * radius + 1] * kernel).max())
    return np.clip(out, 0.0, 1.0)
