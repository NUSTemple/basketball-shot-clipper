"""The manifest is what makes a clip addressable by source timestamp.

The rule with teeth is that shot names are never reused: labels.json keys
on "<video_stem>/shot_NNN.mp4", so a recycled name would hand a brand new
shot the deleted one's label and stars.
"""
from pathlib import Path

from shot_clipper import shot_manifest as sm


def test_absent_manifest_reads_as_none(tmp_path):
    """Every video cut before manifests existed is in this state."""
    assert sm.load(tmp_path) is None


def test_round_trips(tmp_path):
    data = sm.empty("DJI_0001.MP4", Path("C:/videos/DJI_0001.MP4"), duration_s=1068.3)
    sm.add_shot(data, 45.06, sm.DETECTED)
    sm.save(tmp_path, data)

    loaded = sm.load(tmp_path)
    assert loaded["video"] == "DJI_0001.MP4"
    assert loaded["duration_s"] == 1068.3
    assert loaded["shots"]["shot_001.mp4"] == {"t": 45.06, "source": "detected"}


def test_from_cut_results_indexes_by_filename(tmp_path):
    cut_results = [(2, 130.5, tmp_path / "shot_002.mp4"),
                   (1, 45.06, tmp_path / "shot_001.mp4")]
    data = sm.from_cut_results("DJI_0001.MP4", cut_results)

    assert sm.timestamp_for(data, "shot_001.mp4") == 45.06
    assert sm.timestamp_for(data, "shot_002.mp4") == 130.5
    # next name continues past the highest index, not past the count
    assert sm.add_shot(data, 300.0) == "shot_003.mp4"


def test_names_are_never_reused_after_deletion(tmp_path):
    data = sm.empty("DJI_0001.MP4")
    first = sm.add_shot(data, 10.0)
    del data["shots"][first]

    assert sm.add_shot(data, 20.0) != first, "a reused name inherits the old label"


def test_next_index_survives_a_reload(tmp_path):
    """next_index is persisted, but a hand-edited file may not have it -
    recover it from the highest index rather than restarting at 1."""
    data = sm.empty("DJI_0001.MP4")
    sm.add_shot(data, 10.0)
    sm.add_shot(data, 20.0)
    del data["next_index"]
    sm.save(tmp_path, data)

    assert sm.add_shot(sm.load(tmp_path), 30.0) == "shot_003.mp4"


def test_timestamp_for_missing_clip_is_none():
    data = sm.empty("DJI_0001.MP4")
    assert sm.timestamp_for(data, "shot_999.mp4") is None
    assert sm.timestamp_for(None, "shot_001.mp4") is None


def test_nearby_finds_the_closest_within_the_window():
    data = sm.empty("DJI_0001.MP4")
    sm.add_shot(data, 100.0)
    sm.add_shot(data, 108.0)

    assert sm.nearby(data, 101.5, window=3.0) == "shot_001.mp4"
    assert sm.nearby(data, 107.0, window=3.0) == "shot_002.mp4"
    assert sm.nearby(data, 130.0, window=3.0) is None
