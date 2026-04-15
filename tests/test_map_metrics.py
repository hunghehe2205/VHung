import numpy as np
import pytest

from src.utils.map_metrics import separation_stats


def test_separation_perfect_map():
    # Event on frames 40..59, s_t = 1 inside, 0 outside
    scores = np.zeros(100)
    scores[40:60] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    s = separation_stats(scores, mask)
    assert s['gap'] == pytest.approx(1.0)
    assert s['in_mean'] == pytest.approx(1.0)
    assert s['out_mean'] == pytest.approx(0.0)


def test_separation_uniform_half():
    scores = np.full(100, 0.5)
    mask = np.zeros(100)
    mask[40:60] = 1.0
    s = separation_stats(scores, mask)
    assert s['gap'] == pytest.approx(0.0)


def test_separation_no_event_returns_nan_gap():
    scores = np.full(100, 0.1)
    mask = np.zeros(100)  # normal video
    s = separation_stats(scores, mask)
    assert np.isnan(s['gap'])
    assert s['in_mean'] is None or np.isnan(s['in_mean'])
    assert s['out_mean'] == pytest.approx(0.1)
