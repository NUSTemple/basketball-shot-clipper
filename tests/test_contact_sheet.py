from pathlib import Path

from shot_clipper.contact_sheet import group_key, sort_groups, star_from_filename


def test_star_from_filename_present():
    assert star_from_filename(Path("DJI_0001__shot_012_5star.mp4")) == 5


def test_star_from_filename_absent():
    assert star_from_filename(Path("DJI_0001__shot_012.mp4")) is None


def test_group_key_flat_folder():
    root = Path("/data/goal")
    assert group_key(root / "shot_001.mp4", root) == ""


def test_group_key_subfolder():
    root = Path("/data/goal")
    assert group_key(root / "5star" / "shot_001.mp4", root) == "5star"


def test_sort_groups_stars_descending_before_other_names():
    groups = ["unrated", "2star", "5star", "3star", "aaa_extra"]
    assert sort_groups(groups) == ["5star", "3star", "2star", "aaa_extra", "unrated"]


def test_sort_groups_flat_folder_has_single_empty_key():
    assert sort_groups([""]) == [""]
