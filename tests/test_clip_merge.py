"""Overlapping candidates become one clip, not several near-identical ones.

find_makes' cooldown is 1.5s but a clip spans 7s, so one physical make can
yield two to four candidates whose windows almost entirely coincide. The
invariant that matters downstream: a merged clip must still put the FIRST
candidate at clip-local `pre` seconds, because net_motion.POST_WINDOW and
contact_sheet's thumbnail offset both assume the event sits at 5.0s.
"""
import pytest

from shot_clipper.clip_shots import cluster_timestamps

PRE, POST = 5.0, 2.0


def test_far_apart_stay_separate():
    assert cluster_timestamps([10.0, 30.0, 60.0], PRE, POST) == [[10.0], [30.0], [60.0]]


def test_overlapping_windows_merge():
    # 12.0 - 5 = 7.0, which is inside 10.0 + 2 = 12.0
    assert cluster_timestamps([10.0, 12.0], PRE, POST) == [[10.0, 12.0]]


def test_chain_merges_transitively():
    assert cluster_timestamps([10.0, 13.0, 16.0], PRE, POST) == [[10.0, 13.0, 16.0]]


def test_gap_just_too_big_stays_separate():
    # 10.0 + POST = 12.0; next window starts at 19.01 - 5 = 14.01 > 12.0
    assert cluster_timestamps([10.0, 19.01], PRE, POST) == [[10.0], [19.01]]


def test_boundary_touch_merges():
    # next window starts exactly where the previous one ends
    assert cluster_timestamps([10.0, 17.0], PRE, POST) == [[10.0, 17.0]]


def test_unsorted_input_is_handled():
    assert cluster_timestamps([16.0, 10.0, 13.0], PRE, POST) == [[10.0, 13.0, 16.0]]


def test_empty():
    assert cluster_timestamps([], PRE, POST) == []


@pytest.mark.parametrize("stamps", [
    [10.0, 12.0], [10.0, 11.0, 12.5, 13.0], [5.0], [0.5, 1.0],
])
def test_first_candidate_lands_at_pre_seconds(stamps):
    """The reason merging is safe: cutting [first-pre, last+post] leaves the
    first candidate exactly `pre` into the clip, so the 5.0s assumption
    net_motion and contact_sheet rely on still holds."""
    (group,) = cluster_timestamps(stamps, PRE, POST)
    start = group[0] - PRE
    assert group[0] - start == pytest.approx(PRE)
    duration = (group[-1] + POST) - start
    assert duration >= PRE + POST          # never shorter than an unmerged clip
    assert duration == pytest.approx(PRE + POST + (group[-1] - group[0]))


def test_every_candidate_is_covered_by_its_clip():
    stamps = [10.0, 12.0, 14.0, 40.0]
    for group in cluster_timestamps(stamps, PRE, POST):
        start, end = group[0] - PRE, group[-1] + POST
        for t in group:
            assert start <= t <= end
    # nothing dropped
    assert sorted(t for g in cluster_timestamps(stamps, PRE, POST) for t in g) == stamps
