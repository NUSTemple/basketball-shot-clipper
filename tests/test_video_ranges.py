"""/video/<path> never sends one response bigger than VIDEO_CHUNK_BYTES with
a Content-Length: Cloud Run 500s any HTTP/1 response over 32 MiB, which broke
playback of every clip over that size."""
import pytest

from shot_clipper.label_ui import app as app_module

REL = "acct/VID/shot_001.mp4"


@pytest.fixture
def client(tmp_path, monkeypatch):
    clip = tmp_path / REL
    clip.parent.mkdir(parents=True)
    clip.write_bytes(bytes(range(256)) * 100)  # 25,600 bytes
    monkeypatch.setattr(app_module, "VIDEO_CHUNK_BYTES", 10_000)
    app_module.app.config["CLIPS_DIR"] = tmp_path
    with app_module.app.test_client() as c:
        yield c, clip.read_bytes()


def test_open_range_is_capped_to_one_chunk(client):
    c, data = client
    resp = c.get(f"/video/{REL}", headers={"Range": "bytes=0-"})
    assert resp.status_code == 206
    assert resp.headers["Content-Range"] == "bytes 0-9999/25600"
    assert resp.data == data[:10_000]


def test_following_requests_walk_the_rest_of_the_file(client):
    c, data = client
    got, start = b"", 0
    while start < len(data):
        resp = c.get(f"/video/{REL}", headers={"Range": f"bytes={start}-"})
        assert resp.status_code == 206
        got += resp.data
        start += len(resp.data)
    assert got == data


def test_small_explicit_range_is_served_exactly(client):
    c, data = client
    resp = c.get(f"/video/{REL}", headers={"Range": "bytes=100-199"})
    assert resp.headers["Content-Range"] == "bytes 100-199/25600"
    assert resp.data == data[100:200]


def test_unsatisfiable_range_is_416(client):
    c, _ = client
    resp = c.get(f"/video/{REL}", headers={"Range": "bytes=999999-"})
    assert resp.status_code == 416
    assert resp.headers["Content-Range"] == "bytes */25600"


def test_no_range_on_a_big_file_streams_without_content_length(client):
    c, data = client
    resp = c.get(f"/video/{REL}")
    assert resp.status_code == 200
    assert "Content-Length" not in resp.headers
    assert resp.data == data


def test_missing_clip_is_404(client):
    c, _ = client
    assert c.get("/video/acct/VID/nope.mp4").status_code == 404
