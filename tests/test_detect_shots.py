from shot_clipper.detect_shots import compute_roi, find_makes

HOOP = (0.4, 0.3, 0.5, 0.35)  # x1, y1, x2, y2


def test_find_makes_detects_ball_crossing_downward_through_hoop():
    # above the hoop, then below it shortly after -> one make
    track = [
        (0.0, 0.45, 0.20),  # approaching, above hoop
        (0.1, 0.45, 0.33),  # inside/through the hoop box
    ]
    makes = find_makes(track, HOOP)
    assert len(makes) == 1


def test_find_makes_ignores_ball_far_outside_horizontal_window():
    track = [
        (0.0, 0.9, 0.20),
        (0.1, 0.9, 0.33),
    ]
    assert find_makes(track, HOOP) == []


def test_find_makes_rejects_immediate_bounce_back():
    track = [
        (0.0, 0.45, 0.20),  # above
        (0.1, 0.45, 0.33),  # through
        (0.2, 0.45, 0.10),  # bounces back above within BOUNCE_BACK_CONFIRM_SEC
    ]
    assert find_makes(track, HOOP) == []


def test_find_makes_respects_cooldown_between_makes():
    track = [
        (0.0, 0.45, 0.20),
        (0.1, 0.45, 0.33),
        (0.3, 0.45, 0.20),
        (0.4, 0.45, 0.33),  # too soon after the first make
    ]
    assert len(find_makes(track, HOOP)) == 1


def test_find_makes_skips_frames_with_no_ball_detected():
    track = [
        (0.0, None, None),
        (0.05, 0.45, 0.20),
        (0.15, 0.45, 0.33),
    ]
    assert len(find_makes(track, HOOP)) == 1


def test_compute_roi_stays_within_frame_bounds():
    x1, y1, x2, y2 = compute_roi(HOOP, frame_w=3840, frame_h=2160)
    assert 0 <= x1 < x2 <= 3840
    assert y1 == 0
    assert 0 < y2 <= 2160


def test_compute_roi_covers_the_hoop_box_itself():
    hx1, hy1, hx2, hy2 = HOOP
    x1, y1, x2, y2 = compute_roi(HOOP, frame_w=3840, frame_h=2160)
    assert x1 <= hx1 * 3840
    assert x2 >= hx2 * 3840
    assert y2 >= hy2 * 2160
