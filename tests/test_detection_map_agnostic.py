import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.detection_map import getDetectionMAP_agnostic


def test_agnostic_perfect_prediction():
    # 1 video, 100 frames, GT event frames [20, 50] (anomaly)
    pred = np.zeros(100, dtype=np.float32)
    pred[20:50] = 0.9   # high score inside event
    predictions = [pred]
    gtsegments = [[[20, 50]]]                 # one segment in video 0
    gtlabels = [['Abuse']]                    # class ignored by agnostic

    dmap, iou = getDetectionMAP_agnostic(predictions, gtsegments, gtlabels)

    assert iou == [0.1, 0.2, 0.3, 0.4, 0.5]
    # Perfect IoU=1.0 segment should yield AP=100% at all thresholds
    for v in dmap:
        assert v > 80.0, f"Expected >80 mAP, got {v}"


def test_agnostic_no_overlap():
    # prediction is far from GT
    pred = np.zeros(100, dtype=np.float32)
    pred[80:95] = 0.9
    gtsegments = [[[10, 30]]]
    gtlabels = [['Abuse']]
    dmap, _ = getDetectionMAP_agnostic([pred], gtsegments, gtlabels)
    # Zero true positives at every IoU threshold >= 0.1
    for v in dmap:
        assert v == 0.0


def test_agnostic_collapses_classes():
    # Same video has two GT segments with different class labels;
    # agnostic should treat them as a single "anomaly" class.
    pred = np.zeros(200, dtype=np.float32)
    pred[10:40] = 0.8
    pred[80:110] = 0.7
    gtsegments = [[[10, 40], [80, 110]]]
    gtlabels = [['Abuse', 'Fighting']]        # 2 classes, collapsed
    dmap, _ = getDetectionMAP_agnostic([pred], gtsegments, gtlabels)
    # Both predicted segments should match, so high mAP at every IoU
    for v in dmap:
        assert v > 50.0
