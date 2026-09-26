"""/api/clips/delete - removes the clip file, its label entry, and its
cached thumbnail, confined to clips_dir_for_request() like every other
clip-touching endpoint."""
import hashlib

from shot_clipper.label_ui import app as app_module
from shot_clipper.dataset_labels import load_labels


def _make_clip(clips_dir, video="DJI_0010", name="shot_001.mp4"):
    video_dir = clips_dir / video
    video_dir.mkdir(parents=True, exist_ok=True)
    clip_path = video_dir / name
    clip_path.write_bytes(b"fake mp4 bytes")
    return clip_path


def test_delete_removes_clip_label_and_thumbnail(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    clip_path = _make_clip(clips_dir)
    rel = "DJI_0010/shot_001.mp4"

    dataset_base = tmp_path / "dataset"
    dataset_base.mkdir()
    (dataset_base / "labels.json").write_text(
        f'{{"{rel}": {{"label": "goal", "stars": 5, "scorer": "Alice"}}}}')

    thumb_cache = tmp_path / "thumbs"
    thumb_cache.mkdir()
    digest = hashlib.sha1(str(clip_path.resolve()).encode()).hexdigest()
    thumb_path = thumb_cache / f"{digest}.jpg"
    thumb_path.write_bytes(b"fake jpg")

    monkeypatch.setattr(app_module, "THUMBNAIL_CACHE_DIR", thumb_cache)
    app_module.app.config["CLIPS_DIR"] = clips_dir
    monkeypatch.setattr(app_module, "dataset_base_for_request", lambda: dataset_base)

    with app_module.app.test_client() as client:
        resp = client.post("/api/clips/delete", json={"clips": [rel]})

    assert resp.status_code == 200
    body = resp.get_json()
    assert body == {"ok": True, "deleted": 1, "missing": []}
    assert not clip_path.exists()
    assert not thumb_path.exists()
    assert rel not in load_labels(dataset_base)


def test_delete_reports_missing_clips_without_erroring(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    dataset_base = tmp_path / "dataset"
    dataset_base.mkdir()

    monkeypatch.setattr(app_module, "THUMBNAIL_CACHE_DIR", tmp_path / "thumbs")
    app_module.app.config["CLIPS_DIR"] = clips_dir
    monkeypatch.setattr(app_module, "dataset_base_for_request", lambda: dataset_base)

    with app_module.app.test_client() as client:
        resp = client.post("/api/clips/delete", json={"clips": ["DJI_9999/shot_999.mp4"]})

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "deleted": 0, "missing": ["DJI_9999/shot_999.mp4"]}


def test_delete_rejects_path_escaping_clips_dir(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("not a clip")

    monkeypatch.setattr(app_module, "THUMBNAIL_CACHE_DIR", tmp_path / "thumbs")
    app_module.app.config["CLIPS_DIR"] = clips_dir
    monkeypatch.setattr(app_module, "dataset_base_for_request", lambda: tmp_path / "dataset")

    with app_module.app.test_client() as client:
        resp = client.post("/api/clips/delete", json={"clips": ["../secret.txt"]})

    assert resp.status_code == 400
    assert outside.exists()
