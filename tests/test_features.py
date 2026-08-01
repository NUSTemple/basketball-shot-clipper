from shot_clipper.features import extract_features_from_track

HOOP = (0.4, 0.3, 0.5, 0.35)  # x1, y1, x2, y2


def test_no_crossing_when_ball_never_seen():
    track = [(0.0, None, None), (0.1, None, None)]
    f = extract_features_from_track(track, HOOP)
    assert f["has_crossing"] == 0
    assert f["n_detections"] == 0
    assert f["detection_rate"] == 0.0


def test_crossing_detected_with_expected_signs():
    track = [(0.0, 0.45, 0.20), (0.1, 0.45, 0.33)]
    f = extract_features_from_track(track, HOOP)
    assert f["has_crossing"] == 1
    assert f["gap_sec"] == 0.1
    assert f["vertical_drop"] > 0  # ball moved downward through the box
    assert f["bounced_back"] == 0


def test_bounce_back_detected_after_crossing():
    track = [
        (0.0, 0.45, 0.20),
        (0.1, 0.45, 0.33),
        (0.2, 0.45, 0.10),  # reappears above the rim shortly after
    ]
    f = extract_features_from_track(track, HOOP)
    assert f["has_crossing"] == 1
    assert f["bounced_back"] == 1
    assert f["time_to_bounce"] == 0.1


def test_detection_rate_counts_missed_frames():
    track = [
        (0.0, 0.45, 0.20),
        (0.05, None, None),
        (0.1, 0.45, 0.33),
    ]
    f = extract_features_from_track(track, HOOP)
    assert f["n_detections"] == 2
    assert f["detection_rate"] == 2 / 3


def test_horizontal_offset_reflects_distance_from_hoop_center():
    hoop_cx = 0.45
    track_centered = [(0.0, hoop_cx, 0.20), (0.1, hoop_cx, 0.33)]
    track_offset = [(0.0, 0.48, 0.20), (0.1, 0.48, 0.33)]
    f_centered = extract_features_from_track(track_centered, HOOP)
    f_offset = extract_features_from_track(track_offset, HOOP)
    assert f_centered["horiz_offset_above"] < f_offset["horiz_offset_above"]
