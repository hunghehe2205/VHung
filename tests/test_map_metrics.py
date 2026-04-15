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


from src.utils.map_metrics import mass_stats


def test_mass_perfect_map():
    scores = np.zeros(100)
    scores[40:60] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    m = mass_stats(scores, mask)
    assert m['emr'] == pytest.approx(1.0)
    assert m['etr'] == pytest.approx(0.2)
    assert m['mcl'] == pytest.approx(5.0)


def test_mass_uniform_map():
    scores = np.full(100, 0.5)
    mask = np.zeros(100)
    mask[40:60] = 1.0
    m = mass_stats(scores, mask)
    assert m['emr'] == pytest.approx(0.2)
    assert m['etr'] == pytest.approx(0.2)
    assert m['mcl'] == pytest.approx(1.0)


def test_mass_normal_video_returns_nan_mcl():
    scores = np.full(100, 0.1)
    mask = np.zeros(100)
    m = mass_stats(scores, mask)
    assert m['etr'] == pytest.approx(0.0)
    assert np.isnan(m['mcl'])


from src.utils.map_metrics import density_stats


def test_density_perfect_uniform_event():
    # Uniform 1.0 inside 20-frame event; PC is top-10% mass / event mass
    scores = np.zeros(100)
    scores[40:60] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    d = density_stats(scores, mask)
    assert d['in_event_cov_05'] == pytest.approx(1.0)
    assert d['in_event_cov_03'] == pytest.approx(1.0)
    # Top 10% of 20 frames = 2 frames out of total mass 20 → PC=0.1
    assert d['peak_concentration'] == pytest.approx(0.1)
    # Uniform distribution → normalized entropy = 1
    assert d['in_event_entropy'] == pytest.approx(1.0)


def test_density_single_spike():
    scores = np.zeros(100)
    scores[50] = 1.0
    mask = np.zeros(100)
    mask[40:60] = 1.0
    d = density_stats(scores, mask)
    assert d['peak_concentration'] == pytest.approx(1.0)
    assert d['in_event_cov_05'] == pytest.approx(1.0 / 20)
    assert d['in_event_entropy'] == pytest.approx(0.0)


def test_density_normal_video():
    scores = np.full(100, 0.1)
    mask = np.zeros(100)
    d = density_stats(scores, mask)
    assert np.isnan(d['peak_concentration'])
    assert np.isnan(d['in_event_entropy'])
    assert np.isnan(d['in_event_cov_05'])
