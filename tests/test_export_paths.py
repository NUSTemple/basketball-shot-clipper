"""Where an exported clip lands - flat, or one subfolder per scorer.

Covers export_out_path() rather than the /api/export-clips route, so these
run without a Flask app or a filesystem that can make symlinks.
"""
from pathlib import Path

from shot_clipper.label_ui.app import NO_SCORER_FOLDER, export_out_path

DEST = Path("/export")


def goal(stars=None, scorer=None):
    return {"label": "goal", "stars": stars, "scorer": scorer}


def test_flat_export_keeps_video_prefix():
    # the clip's folder is folded into the filename, so two videos' shot_001
    # can share one destination folder
    out = export_out_path(DEST, "DJI_0010/shot_001.mp4", goal(), False)
    assert out == DEST / "DJI_0010__shot_001.mp4"


def test_flat_export_appends_stars_then_scorer():
    out = export_out_path(DEST, "DJI_0010/shot_001.mp4", goal(5, "Alice"), False)
    assert out == DEST / "DJI_0010__shot_001_5star_Alice.mp4"


def test_grouped_export_uses_scorer_folder_and_drops_scorer_suffix():
    out = export_out_path(DEST, "DJI_0010/shot_001.mp4", goal(5, "Alice"), True)
    assert out == DEST / "Alice" / "DJI_0010__shot_001_5star.mp4"


def test_grouped_export_slugs_unsafe_names():
    out = export_out_path(DEST, "DJI_0010/shot_001.mp4", goal(4, "Bob / Jr"), True)
    assert out == DEST / "BobJr" / "DJI_0010__shot_001_4star.mp4"


def test_grouped_export_keeps_non_ascii_names():
    out = export_out_path(DEST, "DJI_0010/shot_001.mp4", goal(3, "陈大文"), True)
    assert out == DEST / "陈大文" / "DJI_0010__shot_001_3star.mp4"


def test_untagged_goal_gets_the_fallback_folder():
    out = export_out_path(DEST, "DJI_0010/shot_002.mp4", goal(4), True)
    assert out == DEST / NO_SCORER_FOLDER / "DJI_0010__shot_002_4star.mp4"


def test_non_goal_clips_share_the_fallback_folder():
    # a scorer/stars only mean something on a goal - a clip relabeled
    # no_goal keeps neither, in either mode
    entry = {"label": "no_goal", "stars": 5, "scorer": "Alice"}
    assert export_out_path(DEST, "DJI_0010/shot_003.mp4", entry, True) == (
        DEST / NO_SCORER_FOLDER / "DJI_0010__shot_003.mp4")
    assert export_out_path(DEST, "DJI_0010/shot_003.mp4", entry, False) == (
        DEST / "DJI_0010__shot_003.mp4")


def test_unlabeled_clip_has_no_suffix():
    assert export_out_path(DEST, "DJI_0010/shot_004.mp4", {}, False) == (
        DEST / "DJI_0010__shot_004.mp4")
    assert export_out_path(DEST, "DJI_0010/shot_004.mp4", {}, True) == (
        DEST / NO_SCORER_FOLDER / "DJI_0010__shot_004.mp4")
