"""The admin Jobs view shows every user's jobs - and only to an admin."""
import pytest

from shot_clipper.label_ui import app as app_module
from shot_clipper.label_ui import jobs
from shot_clipper.label_ui.jobstore import write_job

ADMIN = "owner@example.com"
HEADER = "X-Goog-Authenticated-User-Email"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "job_path", lambda job_id: tmp_path / f"{job_id}.json")
    monkeypatch.setattr(app_module, "MULTI_USER", True)
    monkeypatch.setattr(app_module, "ADMIN_EMAILS", {ADMIN})
    write_job(tmp_path / "a1.json", {
        "id": "a1", "state": "error", "user": "alice@example.com", "created_at": 100,
        "kind": "single", "video": "/data/videos/alice/2026-10-05/uploads/game.mp4",
        "message": "interrupted", "error": "worker stopped", "traceback": "secret stack"})
    write_job(tmp_path / "b1.json", {
        "id": "b1", "state": "done", "user": "bob@example.com", "created_at": 200,
        "kind": "batch", "queue": ["/data/videos/bob/x.mp4", "/data/videos/bob/y.mp4"],
        "completed_videos": [{}], "message": "done: 2 video(s) processed"})
    with app_module.app.test_client() as c:
        yield c


def test_admin_sees_all_users_jobs_newest_first(client):
    resp = client.get("/api/admin/jobs", headers={HEADER: ADMIN})
    assert resp.status_code == 200
    rows = resp.get_json()["jobs"]
    assert [(r["id"], r["user"], r["state"]) for r in rows] == [
        ("b1", "bob@example.com", "done"), ("a1", "alice@example.com", "error")]
    assert rows[0]["video"] == "2 videos" and rows[0]["video_detail"] == "1/2 done"
    assert rows[1]["video"] == "game.mp4" and rows[1]["error"] == "worker stopped"
    assert rows[1]["updated_at"]


def test_rows_do_not_leak_paths_or_tracebacks(client):
    body = client.get("/api/admin/jobs", headers={HEADER: ADMIN}).get_data(as_text=True)
    assert "/data/videos" not in body and "secret stack" not in body


def test_non_admin_is_refused(client):
    resp = client.get("/api/admin/jobs", headers={HEADER: "alice@example.com"})
    assert resp.status_code == 403


def test_not_available_outside_multi_user_mode(client, monkeypatch):
    monkeypatch.setattr(app_module, "MULTI_USER", False)
    assert client.get("/api/admin/jobs").status_code == 404
