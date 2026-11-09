"""AI suggestions from make-highlights: written per video folder, listed with
each clip, and accepted in Review with one labels.json write."""
import json

import pytest

from shot_clipper.clip_shots import AI_SUGGESTIONS_FILENAME
from shot_clipper.dataset_labels import load_labels
from shot_clipper.highlights import write_ai_suggestions
from shot_clipper.label_ui import app as app_module
from shot_clipper.person_id import PLAYERS_FILENAME


@pytest.fixture
def env(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    for video in ("VID", "OTHER"):
        (clips_dir / video).mkdir(parents=True)
        for i in range(1, 4):
            (clips_dir / video / f"shot_00{i}.mp4").write_bytes(b"x")
    (clips_dir / "VID" / AI_SUGGESTIONS_FILENAME).write_text(json.dumps({"clips": {
        "shot_001.mp4": {"stars": 4, "reel": True, "scorer": "Allen"},
        "shot_002.mp4": {"stars": 2, "reel": False}}}))
    base = tmp_path / "dataset"
    base.mkdir()
    (base / "labels.json").write_text('{"VID/shot_003.mp4": {"label": "goal", "stars": 3, "scorer": "Eddie"}}')
    app_module.app.config["CLIPS_DIR"] = clips_dir
    monkeypatch.setattr(app_module, "dataset_base_for_request", lambda: base)
    saves = []
    real_save = app_module.save_labels
    monkeypatch.setattr(app_module, "save_labels", lambda l, b=None: (saves.append(1), real_save(l, b)))
    with app_module.app.test_client() as c:
        yield c, base, saves


def test_clip_list_carries_the_suggestions(env):
    client, _, _ = env
    by_path = {c["path"]: c for c in client.get("/api/clips").get_json()["clips"]}
    assert by_path["VID/shot_001.mp4"].items() >= {"ai_rated": True, "ai_stars": 4, "ai_reel": True, "ai_scorer": "Allen"}.items()
    assert by_path["VID/shot_002.mp4"]["ai_scorer"] is None
    # a clip the AI didn't call a goal, in a rated video: rated, no suggestion
    assert by_path["VID/shot_003.mp4"].items() >= {"ai_rated": True, "ai_stars": None, "ai_reel": False}.items()
    # a video without the file is never treated as rated
    assert by_path["OTHER/shot_001.mp4"]["ai_rated"] is False


def test_accept_saves_goal_stars_and_scorer_in_one_write(env):
    client, base, saves = env
    resp = client.post("/api/label", json={"clip": "VID/shot_001.mp4", "label": "goal", "stars": 4, "scorer": "Allen"})
    assert resp.status_code == 200
    entry = load_labels(base)["VID/shot_001.mp4"]
    assert entry["label"] == "goal" and entry["stars"] == 4 and entry["scorer"] == "Allen"
    assert "source" not in entry  # the person agreed - it's their label
    assert len(saves) == 1
    assert "Allen" in json.loads((base / "roster.json").read_text())["players"]


def test_relabeling_a_goal_keeps_its_scorer(env):
    client, base, _ = env
    client.post("/api/label", json={"clip": "VID/shot_003.mp4", "label": "goal"})
    assert load_labels(base)["VID/shot_003.mp4"].items() >= {"stars": 3, "scorer": "Eddie"}.items()
    client.post("/api/label", json={"clip": "VID/shot_003.mp4", "label": "no_goal"})
    entry = load_labels(base)["VID/shot_003.mp4"]
    assert entry["stars"] is None and "scorer" not in entry


@pytest.mark.parametrize("body", [
    {"clip": "VID/shot_001.mp4", "label": "no_goal", "stars": 3},
    {"clip": "VID/shot_001.mp4", "label": "goal", "stars": 7},
    {"clip": "VID/shot_001.mp4", "label": "goal", "scorer": "  "},
])
def test_rejects_bad_accepts_without_writing(env, body):
    client, base, saves = env
    assert client.post("/api/label", json=body).status_code == 400
    assert not saves


def test_pipeline_writes_suggestions_for_every_processed_video(tmp_path):
    clips_root = tmp_path / "clips"
    for v in ("A", "B", "C"):
        (clips_root / v).mkdir(parents=True)
    (clips_root / "A" / PLAYERS_FILENAME).write_text(json.dumps({"clips": {
        "shot_001.mp4": {"group": "x-00", "suggested": "老曹"}, "shot_002.mp4": {"group": "x-01", "suggested": None}}}))
    ranked = [{"key": "A/shot_001.mp4", "video": "A", "clip": "shot_001.mp4", "stars": 5},
              {"key": "A/shot_002.mp4", "video": "A", "clip": "shot_002.mp4", "stars": 2}]
    results = {"A": {}, "B": {}, "C": {"skipped": "no rim found"}}
    write_ai_suggestions(ranked, results, {"top_plays": {"clips": ["A/shot_001.mp4"]}}, clips_root)
    a = json.loads((clips_root / "A" / AI_SUGGESTIONS_FILENAME).read_text())["clips"]
    assert a == {"shot_001.mp4": {"stars": 5, "reel": True, "scorer": "老曹"},
                 "shot_002.mp4": {"stars": 2, "reel": False}}
    assert json.loads((clips_root / "B" / AI_SUGGESTIONS_FILENAME).read_text())["clips"] == {}
    assert not (clips_root / "C" / AI_SUGGESTIONS_FILENAME).exists()


def test_player_crop_serves_the_saved_file_without_ffmpeg(env, monkeypatch):
    from shot_clipper.person_id.crops import crop_path

    client, _, _ = env
    clip = app_module.app.config["CLIPS_DIR"] / "VID" / "shot_001.mp4"
    crop_path(clip).parent.mkdir()
    crop_path(clip).write_bytes(b"\xff\xd8saved-jpeg")
    monkeypatch.setattr(app_module, "cut_crop", lambda *a: pytest.fail("cut a crop that was already saved"))
    resp = client.get("/player-crop/VID/shot_001.mp4")
    assert resp.status_code == 200 and resp.data == b"\xff\xd8saved-jpeg"
    # no saved crop and no shooter: 404, not a crash
    assert client.get("/player-crop/VID/shot_002.mp4").status_code == 404
