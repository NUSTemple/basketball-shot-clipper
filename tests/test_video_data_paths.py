"""Per-video artifacts belong with the video, not in the app's data dir.

Calibrations and detection results describe one recording. Keeping them in
the checkout meant a file per video anyone ever processed accumulated there
- and got committed. They now live in a shot-clipper/ folder beside the
footage, so copying a shoot to another machine carries its calibration.

The rule with teeth: writes always go to the new location, but reads still
find the old one. A calibration is a hoop box somebody drew by hand, and
quietly failing to find one sends them to do it again.
"""
import json

import pytest

from shot_clipper import paths


@pytest.fixture
def video(tmp_path):
    """A source video in its own folder, away from the app data dir."""
    folder = tmp_path / "footage" / "DJI_001"
    folder.mkdir(parents=True)
    path = folder / "DJI_0001.MP4"
    path.write_bytes(b"")
    return path


@pytest.fixture(autouse=True)
def app_data_dir(tmp_path, monkeypatch):
    """Point the legacy data directory somewhere disposable."""
    root = tmp_path / "appdata"
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(root))
    return root


def test_config_is_written_beside_the_video(video):
    assert paths.config_path_for(video) == video.parent / "shot-clipper" / "DJI_0001.json"


def test_detection_is_written_beside_the_video(video):
    assert (paths.ground_truth_path_for(video)
            == video.parent / "shot-clipper" / "DJI_0001_detected.json")


def test_find_prefers_the_video_folder(video):
    wanted = paths.config_path_for(video)
    wanted.parent.mkdir(parents=True)
    wanted.write_text("{}")

    legacy = paths.legacy_config_path_for(video)
    legacy.parent.mkdir(parents=True)
    legacy.write_text("{}")

    assert paths.find_config(video) == wanted


def test_find_falls_back_to_a_legacy_calibration(video):
    """The case that matters: hoop boxes drawn before the move."""
    legacy = paths.legacy_config_path_for(video)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"hoop_bbox_norm": [0.1, 0.2, 0.3, 0.4]}))

    found = paths.find_config(video)
    assert found == legacy
    assert json.loads(found.read_text())["hoop_bbox_norm"] == [0.1, 0.2, 0.3, 0.4]


def test_find_returns_the_new_path_when_neither_exists(video):
    """So a caller can .is_file() the result, and so anything that writes
    what it gets back writes to the new location rather than the old one."""
    assert paths.find_config(video) == paths.config_path_for(video)
    assert paths.find_ground_truth(video) == paths.ground_truth_path_for(video)


def test_find_ground_truth_falls_back_too(video):
    legacy = paths.legacy_ground_truth_path_for(video)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"makes_sec": [1.0]}))

    assert paths.find_ground_truth(video) == legacy


def test_two_videos_in_different_folders_do_not_collide(tmp_path):
    """Same filename, different shoots - previously one data dir keyed only
    by stem, so the second one silently inherited the first's hoop box."""
    a = tmp_path / "monday" / "DJI_0001.MP4"
    b = tmp_path / "tuesday" / "DJI_0001.MP4"
    for path in (a, b):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"")

    assert paths.config_path_for(a) != paths.config_path_for(b)
