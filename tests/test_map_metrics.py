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


from src.utils.map_metrics import binary_detection_map, compute_per_video_metrics, composite_map_score


def test_binary_detection_map_perfect():
    # 2 videos; each has one event. Predictions match exactly.
    predictions = [
        np.concatenate([np.zeros(40), np.ones(20), np.zeros(40)]),
        np.concatenate([np.zeros(10), np.ones(30), np.zeros(60)]),
    ]
    gt_segments = [
        [[40, 60]],
        [[10, 40]],
    ]
    result = binary_detection_map(predictions, gt_segments)
    assert result['map_avg'] >= 0.99
    assert result['map_at_iou_05'] >= 0.99


def test_binary_detection_map_all_wrong():
    predictions = [np.zeros(100), np.zeros(100)]
    gt_segments = [[[40, 60]], [[10, 40]]]
    result = binary_detection_map(predictions, gt_segments)
    assert result['map_avg'] == pytest.approx(0.0)


def test_binary_detection_map_single_spike():
    # Spike at frame 50 inside GT [40,60] → IoU ≈ 1/20 = 0.05
    pred = np.zeros(100); pred[50] = 1.0
    result = binary_detection_map([pred], [[[40, 60]]])
    # At IoU >= 0.1, prediction matches nothing → mAP=0
    assert result['map_at_iou_01'] == pytest.approx(0.0)


def test_per_video_metrics_perfect():
    scores = np.zeros(100); scores[40:60] = 1.0
    mask = np.zeros(100); mask[40:60] = 1.0
    m = compute_per_video_metrics(scores, mask)
    assert m['gap'] == pytest.approx(1.0)
    assert m['mcl'] == pytest.approx(5.0)
    assert m['peak_concentration'] == pytest.approx(0.1)


def test_composite_map_score_uniform_zero():
    m = {'gap': 0.0, 'mcl': 1.0, 'map_avg': 0.0, 'peak_concentration': 0.1}
    score = composite_map_score(m)
    # gap=0, mcl/3=0.33, map=0, (1-pc)=0.9 → 0.25*0 + 0.25*0.33 + 0.25*0 + 0.25*0.9 = 0.308
    assert 0.3 < score < 0.32


def test_composite_map_score_perfect():
    m = {'gap': 1.0, 'mcl': 5.0, 'map_avg': 1.0, 'peak_concentration': 0.1}
    score = composite_map_score(m)
    # gap=1, clip(mcl/3,0,1)=1, map=1, (1-pc)=0.9 → 0.25*(1+1+1+0.9) = 0.975
    assert 0.97 < score < 0.98
