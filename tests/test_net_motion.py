import numpy as np

from shot_clipper.net_motion import diff_series, net_roi_px, shape_features

HOOP = (0.4, 0.3, 0.5, 0.35)  # x1, y1, x2, y2


def test_net_roi_sits_below_the_hoop():
    x1, y1, x2, y2 = net_roi_px(HOOP, frame_w=1000, frame_h=1000)
    assert y1 == 350  # starts exactly at hoop_y2
    assert y2 > y1
    assert x1 < 400 and x2 > 500  # small margin beyond the hoop's x-span


def test_net_roi_stays_within_frame_bounds():
    x1, y1, x2, y2 = net_roi_px((0.05, 0.95, 0.15, 0.98), frame_w=500, frame_h=500)
    assert 0 <= x1 < x2 <= 500
    assert 0 <= y1 < y2 <= 500


def test_diff_series_zero_for_identical_frames():
    frame = np.zeros((10, 10), dtype=np.uint8)
    series = diff_series([frame, frame, frame])
    assert series == [0.0, 0.0]


def test_diff_series_detects_change():
    a = np.zeros((10, 10), dtype=np.uint8)
    b = np.full((10, 10), 50, dtype=np.uint8)
    series = diff_series([a, b, a])
    assert series[0] == 50.0
    assert series[1] == 50.0


def test_shape_features_empty_series():
    f = shape_features([], "diff")
    assert f == {"diff_peak": 0.0, "diff_time_to_peak": 0.0,
                 "diff_decay_ratio": 0.0, "diff_impulsiveness": 0.0}


def test_shape_features_sharp_burst_then_decay():
    # a real swish: low baseline, one sharp spike, decays back down
    series = [1.0, 1.0, 20.0, 3.0, 1.0]
    f = shape_features(series, "diff")
    assert f["diff_peak"] == 20.0
    assert f["diff_time_to_peak"] == 2 / 5
    assert f["diff_decay_ratio"] < 0.2  # ends well below the peak
    assert f["diff_impulsiveness"] > 5  # peak stands out from the rest


def test_shape_features_sustained_activity_is_less_impulsive_than_a_burst():
    burst = [1.0, 1.0, 20.0, 1.0, 1.0]
    sustained = [15.0, 18.0, 20.0, 17.0, 16.0]
    f_burst = shape_features(burst, "diff")
    f_sustained = shape_features(sustained, "diff")
    assert f_burst["diff_impulsiveness"] > f_sustained["diff_impulsiveness"]
