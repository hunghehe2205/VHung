"""Map-quality metrics for logits3 anomaly distribution.

Designed to evaluate whether s_t ∈ [0,1] is suitable for the
CDF-cumsum sampler (see inference_viz/sampling.py).

All functions operate on NumPy arrays (CPU) and are side-effect free.
"""
from __future__ import annotations

import math
import numpy as np


def separation_stats(scores: np.ndarray, mask: np.ndarray) -> dict:
    """Gap and per-side means.

    Args:
        scores: (T,) float array, s_t ∈ [0,1].
        mask:   (T,) float/bool array, 1 inside events.

    Returns:
        dict with keys: in_mean, out_mean, gap.
        If no in-event frames exist (normal video), in_mean is NaN and gap is NaN.
        If no out-of-event frames exist (fully anomalous), out_mean is NaN.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    in_frames = scores[mask]
    out_frames = scores[~mask]
    in_mean = float(in_frames.mean()) if in_frames.size > 0 else float('nan')
    out_mean = float(out_frames.mean()) if out_frames.size > 0 else float('nan')
    gap = in_mean - out_mean
    return {'in_mean': in_mean, 'out_mean': out_mean, 'gap': gap}
