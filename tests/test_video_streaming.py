"""/video/source/<relpath> must never send more than MAX_VIDEO_CHUNK_BYTES in
one response: Cloud Run rejects HTTP/1 responses over 32 MiB, and a <video>
seek sends an open-ended "bytes=N-" range that would otherwise return the
whole rest of a multi-GB file (a real 500 in production)."""
from shot_clipper.label_ui import app as app_module

CHUNK = 1024


def _setup(tmp_path, monkeypatch, size):
    data = bytes(i % 256 for i in range(size))
    video = tmp_path / "videos" / "u" / "game.MP4"
    video.parent.mkdir(parents=True)
    video.write_bytes(data)
    monkeypatch.setattr(app_module, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "MAX_VIDEO_CHUNK_BYTES", CHUNK)
    return data


def test_open_ended_range_is_capped(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, size=CHUNK * 5)
    with app_module.app.test_client() as c:
        r = c.get("/video/source/videos/u/game.MP4", headers={"Range": "bytes=100-"})
    assert r.status_code == 206
    assert r.data == data[100:100 + CHUNK]
    assert r.headers["Content-Range"] == f"bytes 100-{100 + CHUNK - 1}/{len(data)}"
    assert r.mimetype == "video/mp4"


def test_range_near_end_returns_only_what_remains(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, size=CHUNK * 5)
    start = len(data) - 10
    with app_module.app.test_client() as c:
        r = c.get("/video/source/videos/u/game.MP4", headers={"Range": f"bytes={start}-"})
    assert r.status_code == 206
    assert r.data == data[start:]


def test_unsatisfiable_range_is_416(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, size=CHUNK)
    with app_module.app.test_client() as c:
        r = c.get("/video/source/videos/u/game.MP4", headers={"Range": f"bytes={len(data) + 5}-"})
    assert r.status_code == 416


def test_small_file_without_range_is_plain_200(tmp_path, monkeypatch):
    data = _setup(tmp_path, monkeypatch, size=CHUNK // 2)
    with app_module.app.test_client() as c:
        r = c.get("/video/source/videos/u/game.MP4")
    assert r.status_code == 200
    assert r.data == data
