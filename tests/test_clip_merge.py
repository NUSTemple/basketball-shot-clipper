"""Makes less than merge_gap seconds apart become one clip, not several
near-identical ones - covers both re-detections of a single physical make
(find_makes' cooldown is 1.5s) and genuinely distinct back-to-back makes
close enough together that separate clips would mostly duplicate each
other's footage. Deliberately independent of clip padding (pre/post,
tested in test_detect_shots.py/clip_shots.py's cut_all instead) - how
close two makes must be to belong in one clip is a different question
from how much lead-in/lead-out that clip gets.
"""
import pytest

from shot_clipper.clip_shots import cluster_timestamps

MERGE_GAP = 5.0


def test_far_apart_stay_separate():
    assert cluster_timestamps([10.0, 30.0, 60.0], MERGE_GAP) == [[10.0], [30.0], [60.0]]


def test_close_together_merge():
    assert cluster_timestamps([10.0, 13.0], MERGE_GAP) == [[10.0, 13.0]]


def test_chain_merges_transitively():
    assert cluster_timestamps([10.0, 13.0, 16.0], MERGE_GAP) == [[10.0, 13.0, 16.0]]


def test_gap_just_too_big_stays_separate():
    assert cluster_timestamps([10.0, 15.01], MERGE_GAP) == [[10.0], [15.01]]


def test_boundary_gap_stays_separate():
    # "less than merge_gap" is a strict inequality - exactly merge_gap apart
    # does not merge
    assert cluster_timestamps([10.0, 15.0], MERGE_GAP) == [[10.0], [15.0]]


def test_just_under_boundary_merges():
    assert cluster_timestamps([10.0, 14.99], MERGE_GAP) == [[10.0, 14.99]]


def test_unsorted_input_is_handled():
    assert cluster_timestamps([16.0, 10.0, 13.0], MERGE_GAP) == [[10.0, 13.0, 16.0]]


def test_empty():
    assert cluster_timestamps([], MERGE_GAP) == []


def test_default_merge_gap_is_five_seconds():
    assert cluster_timestamps([10.0, 14.9]) == [[10.0, 14.9]]
    assert cluster_timestamps([10.0, 15.0]) == [[10.0], [15.0]]


@pytest.mark.parametrize("gap,expected", [
    (1.0, [[10.0], [13.0], [16.0]]),   # tight gap: nothing close enough to merge
    (10.0, [[10.0, 13.0, 16.0]]),      # generous gap: all three chain together
])
def test_merge_gap_is_configurable(gap, expected):
    assert cluster_timestamps([10.0, 13.0, 16.0], gap) == expected
