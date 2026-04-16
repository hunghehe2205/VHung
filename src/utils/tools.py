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


def build_boundary_offset_targets(events_sec, fps, n_features, clip_len=16, target_len=256):
    """Compute per-snippet start/end cls + offset targets from event timestamps.
    Returns: (start_cls, end_cls, start_off, end_off) each [target_len] float32.
    cls: 1.0 at boundary snippet, 0.0 elsewhere.
    off: sub-snippet offset in [0, 1) at boundary snippet, 0 elsewhere.
    """
    start_cls = np.zeros(target_len, dtype=np.float32)
    end_cls = np.zeros(target_len, dtype=np.float32)
    start_off = np.zeros(target_len, dtype=np.float32)
    end_off = np.zeros(target_len, dtype=np.float32)

    if not events_sec:
        return start_cls, end_cls, start_off, end_off

    # Snippet-to-raw mapping (same layout as uniform_extract / build_frame_labels)
    r = np.linspace(0, n_features, target_len + 1, dtype=np.int32)

    for s_sec, e_sec in events_sec:
        s_frame = float(s_sec) * fps
        e_frame = float(e_sec) * fps
        s_raw = s_frame / clip_len  # fractional raw snippet index
        e_raw = e_frame / clip_len

        # Find target snippet containing start
        for t in range(target_len):
            if r[t] <= s_raw < r[t + 1] or (t == target_len - 1 and s_raw >= r[t]):
                start_cls[t] = 1.0
                snippet_start_frame = r[t] * clip_len
                snippet_span = max((r[t + 1] - r[t]) * clip_len, clip_len)
                start_off[t] = np.clip((s_frame - snippet_start_frame) / snippet_span, 0.0, 0.999)
                break

        # Find target snippet containing end
        for t in range(target_len):
            if r[t] <= e_raw < r[t + 1] or (t == target_len - 1 and e_raw >= r[t]):
                end_cls[t] = 1.0
                snippet_start_frame = r[t] * clip_len
                snippet_span = max((r[t + 1] - r[t]) * clip_len, clip_len)
                end_off[t] = np.clip((e_frame - snippet_start_frame) / snippet_span, 0.0, 0.999)
                break

    return start_cls, end_cls, start_off, end_off
