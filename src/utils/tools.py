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


def build_boundary_gaussian_targets(events_sec, fps, n_features, clip_len=16,
                                     target_len=256, sigma=2.0):
    """Gaussian-smoothed boundary targets for start/end prediction.
    Returns: (start_cls, end_cls) each [target_len] float32.
    Peak=1.0 at boundary snippet, smooth decay with given sigma (in snippets).
    """
    s_cls = np.zeros(target_len, dtype=np.float32)
    e_cls = np.zeros(target_len, dtype=np.float32)

    if not events_sec:
        return s_cls, e_cls

    r = np.linspace(0, n_features, target_len + 1, dtype=np.int32)
    t_idx = np.arange(target_len, dtype=np.float32)

    for s_sec, e_sec in events_sec:
        s_frame = float(s_sec) * fps
        e_frame = float(e_sec) * fps
        s_raw = s_frame / clip_len
        e_raw = e_frame / clip_len

        for t in range(target_len):
            if r[t] <= s_raw < r[t + 1] or (t == target_len - 1 and s_raw >= r[t]):
                gauss = np.exp(-0.5 * ((t_idx - t) / sigma) ** 2)
                s_cls = np.maximum(s_cls, gauss)
                break

        for t in range(target_len):
            if r[t] <= e_raw < r[t + 1] or (t == target_len - 1 and e_raw >= r[t]):
                gauss = np.exp(-0.5 * ((t_idx - t) / sigma) ** 2)
                e_cls = np.maximum(e_cls, gauss)
                break

    s_cls[s_cls < 0.01] = 0.0
    e_cls[e_cls < 0.01] = 0.0
    return s_cls, e_cls
