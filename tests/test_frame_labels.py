import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.tools import build_frame_labels


def test_no_event_all_zero():
    y = build_frame_labels(events_sec=[], fps=30.0, n_features=10,
                           clip_len=16, target_len=32)
    assert y.shape == (32,)
    assert y.dtype == np.float32
    assert y.sum() == 0.0


def test_event_marks_overlapping_features():
    # fps=30, event [0.0, 1.0]sec => frames [0, 30). clip_len=16 =>
    # feature 0 covers [0,16): overlap -> 1
    # feature 1 covers [16,32): overlap (16..30) -> 1
    # feature 2 covers [32,48): no overlap -> 0
    y = build_frame_labels(events_sec=[(0.0, 1.0)], fps=30.0,
                           n_features=5, clip_len=16, target_len=5)
    assert list(y) == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_multiple_events_union():
    y = build_frame_labels(events_sec=[(0.0, 0.5), (2.0, 3.0)], fps=30.0,
                           n_features=8, clip_len=16, target_len=8)
    # event1 frames [0,15): feature 0 covers [0,16) -> 1
    # event2 frames [60,90): feature 3 covers [48,64) overlaps at [60,64)-> 1
    #                        feature 4 covers [64,80): overlap -> 1
    #                        feature 5 covers [80,96): overlap at [80,90)-> 1
    assert list(y) == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_padding_when_short():
    # n_features=3 < target_len=10 -> last 7 slots are 0 (padding)
    y = build_frame_labels(events_sec=[(0.0, 1.0)], fps=30.0,
                           n_features=3, clip_len=16, target_len=10)
    assert y.shape == (10,)
    assert y[0] == 1.0
    assert y[3:].sum() == 0.0


def test_downsample_uses_max():
    # n_features=10, target_len=5 -> each target slot aggregates 2 source.
    # Use max (any-overlap survives).
    # Source y = [1,0, 0,0, 0,1, 0,0, 0,0] (events mapped to features 0 and 5)
    # Target should be [max(1,0)=1, max(0,0)=0, max(0,1)=1, max(0,0)=0, max(0,0)=0]
    # Achieve this by placing events at seconds matching features 0 and 5.
    # fps=30 clip_len=16: feature 0 -> frames [0,16)  => event at 0.0-0.5sec
    # feature 5 -> frames [80,96)                    => event at ~2.66-3.1 sec
    y = build_frame_labels(events_sec=[(0.0, 0.3), (2.66, 3.1)], fps=30.0,
                           n_features=10, clip_len=16, target_len=5)
    # After max-downsample, indices 0 and 2 (containing src 0 and src 5) should be 1.
    assert y[0] == 1.0
    assert y[2] == 1.0
    assert y[1] == 0.0
