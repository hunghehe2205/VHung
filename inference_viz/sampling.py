import numpy as np
from scipy import interpolate


def density_aware_sample(anomaly_score, select_frames=16, tau=0.1):
    """Sample snippet indices weighted by anomaly-score density.

    anomaly_score: 1-D array-like of per-snippet scores.
    select_frames: number of snippets to pick.
    tau: smoothing constant added to every score so zero-anomaly regions
         still contribute non-zero probability mass.

    Returns list[int] of length select_frames.
    """
    anomaly_score = np.asarray(anomaly_score, dtype=float)
    num_frames = anomaly_score.shape[0]

    if num_frames <= select_frames or anomaly_score.sum() < 1:
        return [int(x) for x in np.rint(np.linspace(0, num_frames - 1, select_frames))]

    scores = anomaly_score + tau
    score_cumsum = np.concatenate(([0.0], np.cumsum(scores)))
    max_score_cumsum = int(np.round(score_cumsum[-1]))
    f_upsample = interpolate.interp1d(
        score_cumsum,
        np.arange(num_frames + 1),
        kind='linear',
        axis=0,
        fill_value='extrapolate',
    )
    scale_x = np.linspace(1, max_score_cumsum, select_frames)
    sampled_idxs = f_upsample(scale_x)
    return [min(num_frames - 1, max(0, int(idx))) for idx in sampled_idxs]
