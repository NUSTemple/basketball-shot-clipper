"""A re-run that finds fewer candidates must not leave the previous run's
higher-numbered clips behind, looking like part of the new run."""
from shot_clipper.clip_shots import SCORES_FILENAME, clear_previous_clips


def test_removes_old_clips_and_scores_but_nothing_else(tmp_path):
    for name in ("shot_001.mp4", "shot_010.mp4", SCORES_FILENAME, "notes.txt"):
        (tmp_path / name).write_text("x")
    clear_previous_clips(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.txt"]


def test_missing_dir_is_fine(tmp_path):
    clear_previous_clips(tmp_path / "not-there")
