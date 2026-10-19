"""Pre-label suggestions come from their own per-video file and field, kept
apart from the trained filter's scores, which are on a different scale."""
import json

from shot_clipper.clip_shots import PRELABEL_FILENAME, SCORES_FILENAME
from shot_clipper.label_ui.app import list_clips


def test_prelabel_and_filter_scores_are_separate_fields(tmp_path):
    vid = tmp_path / "acct" / "VID"
    vid.mkdir(parents=True)
    for name in ("shot_001.mp4", "shot_002.mp4"):
        (vid / name).write_bytes(b"x")
    (vid / PRELABEL_FILENAME).write_text(json.dumps({"shot_001.mp4": 0.91}))
    (vid / SCORES_FILENAME).write_text(json.dumps({"shot_002.mp4": 0.04}))

    by_shot = {c["shot"]: c for c in list_clips(tmp_path)}
    assert by_shot["shot_001"]["prelabel_score"] == 0.91
    assert by_shot["shot_001"]["filter_score"] is None
    assert by_shot["shot_002"]["prelabel_score"] is None
    assert by_shot["shot_002"]["filter_score"] == 0.04
