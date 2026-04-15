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


def mass_stats(scores: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> dict:
    """Event-Mass Ratio (EMR), Event-Time Ratio (ETR), and Mass Concentration Lift (MCL).

    EMR = Σ s_t(in-event) / Σ s_t(all)
    ETR = |in-event| / |all|
    MCL = EMR / ETR  (NaN if no events)
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    total_mass = scores.sum()
    in_mass = scores[mask].sum()
    emr = float(in_mass / (total_mass + eps)) if total_mass > 0 else 0.0
    etr = float(mask.sum() / mask.size) if mask.size > 0 else 0.0
    mcl = float(emr / etr) if etr > 0 else float('nan')
    return {'emr': emr, 'etr': etr, 'mcl': mcl, 'mass_lift': emr - etr}


def density_stats(scores: np.ndarray, mask: np.ndarray,
                  cov_thresholds: tuple = (0.3, 0.5),
                  top_fraction: float = 0.1,
                  eps: float = 1e-8) -> dict:
    """Metrics for in-event density and anti-spike.

    in_event_cov@τ: fraction of event frames with s_t > τ.
    peak_concentration: mass of top X% event frames / total event mass.
    in_event_entropy: normalized entropy H(p̂) / log(L_event), where
        p̂[t] = s_t / Σ_event s_t.  1 = uniform (dense), 0 = single spike.
    """
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    event_scores = scores[mask]
    L = event_scores.size
    out = {}
    for tau in cov_thresholds:
        key = f'in_event_cov_{int(round(tau * 10)):02d}'
        out[key] = float((event_scores > tau).mean()) if L > 0 else float('nan')

    if L == 0:
        out['peak_concentration'] = float('nan')
        out['in_event_entropy'] = float('nan')
        return out

    total_event_mass = event_scores.sum()
    if total_event_mass <= eps:
        out['peak_concentration'] = float('nan')
        out['in_event_entropy'] = 0.0
        return out

    k = max(1, int(math.ceil(L * top_fraction)))
    topk = np.sort(event_scores)[-k:]
    out['peak_concentration'] = float(topk.sum() / total_event_mass)

    p = event_scores / total_event_mass
    p_safe = np.clip(p, eps, 1.0)
    H = float(-(p * np.log(p_safe)).sum())
    out['in_event_entropy'] = float(H / math.log(L)) if L > 1 else 0.0
    return out
