"""/api/label-bulk: Review's "mark hidden likely misses as Miss". One write
for the whole batch, never overrides a person's label, and tags what it sets
so training can tell the AI's guesses from real labels."""
import pytest

from shot_clipper.dataset_labels import load_labels
from shot_clipper.label_ui import app as app_module


@pytest.fixture
def env(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    (clips_dir / "VID").mkdir(parents=True)
    for i in range(1, 5):
        (clips_dir / "VID" / f"shot_00{i}.mp4").write_bytes(b"x")
    base = tmp_path / "dataset"
    base.mkdir()
    (base / "labels.json").write_text('{"VID/shot_001.mp4": {"label": "goal", "stars": 4}}')
    app_module.app.config["CLIPS_DIR"] = clips_dir
    monkeypatch.setattr(app_module, "dataset_base_for_request", lambda: base)
    saves = []
    real_save = app_module.save_labels
    monkeypatch.setattr(app_module, "save_labels", lambda l, b=None: (saves.append(1), real_save(l, b)))
    with app_module.app.test_client() as c:
        yield c, base, saves


def test_marks_unlabeled_clips_once_and_never_overrides_a_person(env):
    client, base, saves = env
    resp = client.post("/api/label-bulk", json={
        "clips": ["VID/shot_001.mp4", "VID/shot_002.mp4", "VID/shot_003.mp4"], "label": "no_goal", "source": "ai"})
    assert resp.get_json() == {"ok": True, "labeled": 2, "skipped": 1}
    labels = load_labels(base)
    assert labels["VID/shot_001.mp4"] == {"label": "goal", "stars": 4}  # the person's goal stands
    assert labels["VID/shot_002.mp4"]["label"] == "no_goal"
    assert labels["VID/shot_002.mp4"]["source"] == "ai"
    assert len(saves) == 1  # one write for the batch


def test_clip_list_reports_the_source_and_a_manual_relabel_clears_it(env):
    client, base, _ = env
    client.post("/api/label-bulk", json={"clips": ["VID/shot_002.mp4"], "label": "no_goal", "source": "ai"})
    by_path = {c["path"]: c for c in client.get("/api/clips").get_json()["clips"]}
    assert by_path["VID/shot_002.mp4"]["label_source"] == "ai"
    assert by_path["VID/shot_003.mp4"]["label_source"] is None
    client.post("/api/label", json={"clip": "VID/shot_002.mp4", "label": "goal"})
    assert "source" not in load_labels(base)["VID/shot_002.mp4"]


@pytest.mark.parametrize("body,status", [
    ({"clips": [], "label": "no_goal"}, 400),
    ({"clips": ["VID/shot_002.mp4"], "label": "maybe"}, 400),
    ({"clips": ["VID/shot_002.mp4"], "label": "no_goal", "source": "robot"}, 400),
    ({"clips": ["VID/nope.mp4"], "label": "no_goal"}, 404),
    ({"clips": ["../../etc/passwd"], "label": "no_goal"}, 400),
])
def test_rejects_bad_requests_without_writing(env, body, status):
    client, base, saves = env
    assert client.post("/api/label-bulk", json=body).status_code == status
    assert saves == []
