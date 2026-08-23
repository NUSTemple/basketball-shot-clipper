"""A squad narrows the scorer picker to who was actually there.

The rule that matters is what happens when a squad *isn't* set: the picker
has to fall back to the whole roster, or every video labelled before this
existed would offer nobody to tag.
"""
import pytest

from shot_clipper import paths, squads


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path))
    return tmp_path


def test_date_comes_from_the_dji_filename():
    assert squads.date_key_for("DJI_20260822170858_0002_D.MP4") == "2026-08-22"
    assert squads.date_key_for("DJI_20260822170858_0002_D") == "2026-08-22"


def test_a_name_without_a_date_has_none():
    """A different camera or a hand-named folder gets no squad rather than
    a guessed one."""
    assert squads.date_key_for("practice-clip.mp4") is None
    assert squads.squad_for_video("practice-clip.mp4") is None


def test_unset_squad_reads_as_none_not_empty():
    """None means 'offer everyone'. An empty list would mean 'offer nobody',
    which is the opposite of the intended fallback."""
    assert squads.squad_for("2026-08-22") is None


def test_round_trips_sorted():
    squads.set_squad("2026-08-22", ["TP", "alice", "Bob"])
    assert squads.squad_for("2026-08-22") == ["alice", "Bob", "TP"]


def test_duplicates_collapse_case_insensitively():
    kept = squads.set_squad("2026-08-22", ["Alice", "alice", " Alice "])
    assert kept == ["Alice"]


def test_blank_names_are_dropped():
    assert squads.set_squad("2026-08-22", ["Alice", "", "   "]) == ["Alice"]


def test_empty_squad_clears_rather_than_offering_nobody():
    squads.set_squad("2026-08-22", ["Alice"])
    squads.set_squad("2026-08-22", [])
    assert squads.squad_for("2026-08-22") is None


def test_squad_for_video_uses_its_date():
    squads.set_squad("2026-08-22", ["Alice"])
    assert squads.squad_for_video("DJI_20260822170858_0002_D.MP4") == ["Alice"]
    assert squads.squad_for_video("DJI_20260815171620_0015_D.MP4") is None


def test_tagging_a_new_name_adds_them_to_that_squad():
    """Otherwise you type a name, tag the clip, and the picker still doesn't
    offer them for the next one."""
    squads.set_squad("2026-08-22", ["Alice"])
    squads.add_to_squad("2026-08-22", "Bob")
    assert squads.squad_for("2026-08-22") == ["Alice", "Bob"]


def test_adding_to_a_date_with_no_squad_stays_unset():
    """A date deliberately left open shouldn't acquire a one-person squad
    the first time somebody is tagged - that would silently switch it from
    'offer everyone' to 'offer one person'."""
    squads.add_to_squad("2026-08-22", "Bob")
    assert squads.squad_for("2026-08-22") is None


def test_adding_an_existing_name_is_a_no_op():
    squads.set_squad("2026-08-22", ["Alice"])
    squads.add_to_squad("2026-08-22", "alice")
    assert squads.squad_for("2026-08-22") == ["Alice"]


def test_squads_are_per_date():
    squads.set_squad("2026-08-22", ["Alice"])
    squads.set_squad("2026-08-15", ["Bob"])
    assert squads.squad_for("2026-08-22") == ["Alice"]
    assert squads.squad_for("2026-08-15") == ["Bob"]


def test_non_ascii_names_survive_the_round_trip():
    """The roster is mostly Chinese names; json.dumps would escape them by
    default and make the file unreadable by hand."""
    squads.set_squad("2026-08-22", ["熊瑶", "阿德"])
    assert set(squads.squad_for("2026-08-22")) == {"熊瑶", "阿德"}
    assert "熊瑶" in squads.squads_path().read_text(encoding="utf-8")
