from shot_clipper.validate import match


def test_match_pairs_detections_within_tolerance():
    matched, false_positives, missed = match([10.1, 20.0], [10.0, 30.0], tolerance=0.5)
    assert matched == [(10.1, 10.0)]
    assert false_positives == [20.0]
    assert missed == [30.0]


def test_match_all_hit():
    matched, false_positives, missed = match([5.0, 15.0], [5.2, 15.1], tolerance=0.5)
    assert len(matched) == 2
    assert false_positives == []
    assert missed == []


def test_match_picks_closest_candidate_when_multiple_in_range():
    matched, false_positives, missed = match([10.0], [9.8, 10.3], tolerance=1.0)
    assert matched == [(10.0, 9.8)]
    assert missed == [10.3]


def test_match_empty_inputs():
    matched, false_positives, missed = match([], [], tolerance=2.0)
    assert (matched, false_positives, missed) == ([], [], [])
