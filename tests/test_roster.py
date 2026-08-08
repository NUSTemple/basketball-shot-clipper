"""roster.py backs the Review panel's scorer picker - see docs/PLAYER_IDENTIFICATION.md
for how this is meant to plug into an eventual automated suggester."""
import pytest

from shot_clipper import roster


@pytest.fixture(autouse=True)
def _tmp_dataset_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(roster, "DATASET_DIR", tmp_path)


def test_load_roster_missing_file_is_empty():
    assert roster.load_roster() == []


def test_add_player_persists_and_sorts():
    roster.add_player("Bob")
    roster.add_player("alice")
    assert roster.load_roster() == ["alice", "Bob"]


def test_add_player_is_idempotent():
    roster.add_player("Alice")
    roster.add_player("Alice")
    assert roster.load_roster() == ["Alice"]


def test_add_player_strips_whitespace():
    roster.add_player("  Alice  ")
    assert roster.load_roster() == ["Alice"]


def test_add_player_rejects_empty_name():
    with pytest.raises(ValueError):
        roster.add_player("   ")


@pytest.mark.parametrize("name,expected", [
    ("Alice", "Alice"),
    ("Alice B.", "AliceB."),
    ("李明", "李明"),
    ("a/b\\c", "abc"),
    ("   ", "player"),
])
def test_scorer_slug(name, expected):
    assert roster.scorer_slug(name) == expected
