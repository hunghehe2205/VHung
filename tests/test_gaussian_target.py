import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.tools import build_gaussian_target  # noqa: E402


def test_gaussian_target_has_soft_peak_at_event_center():
    y_bin = np.zeros(256, dtype=np.float32)
    y_bin[100:120] = 1.0
    y_soft = build_gaussian_target(y_bin, sigma=2.0)
    assert y_soft.shape == (256,)
    assert y_soft[110] > 0.99
    assert 0.0 < y_soft[121] < 1.0
    assert 0.0 < y_soft[99] < 1.0
    assert y_soft[50] == 0.0
    assert y_soft[200] == 0.0


def test_gaussian_target_all_zero_when_no_events():
    y_bin = np.zeros(256, dtype=np.float32)
    y_soft = build_gaussian_target(y_bin, sigma=2.0)
    assert (y_soft == 0).all()
