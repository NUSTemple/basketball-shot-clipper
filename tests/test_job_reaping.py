"""A job whose worker died must stop saying "running".

On the hosted deployment the worker runs in a different container from the
label UI, and a crashed worker instance is replaced by a fresh one - so the
pid a job recorded can't be probed by whoever reads it later, and may even
be alive there as an unrelated process.
"""
import json
import os
import time

import pytest

from shot_clipper.label_ui import jobs, worker
from shot_clipper.label_ui.jobstore import read_job, write_job


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "job_path", lambda job_id: tmp_path / f"{job_id}.json")
    return tmp_path


def _running_job(jobs_dir, pid, host, age=0.0):
    job_file = jobs_dir / "abc.json"
    write_job(job_file, {"id": "abc", "state": "running", "created_at": 1})
    pid_file = jobs_dir / "abc.pid"
    pid_file.write_text(json.dumps({"pid": pid, "host": host}))
    stamp = time.time() - age
    for f in (job_file, pid_file):
        os.utime(f, (stamp, stamp))
    return job_file


def test_live_pid_from_another_instance_does_not_keep_job_alive(jobs_dir):
    # our own pid is certainly alive here - but it was recorded elsewhere
    job_file = _running_job(jobs_dir, os.getpid(), "localhost:some-other-boot", age=600)
    assert jobs.get_job("abc")["state"] == "error"
    assert read_job(job_file)["message"] == "interrupted"


def test_fresh_heartbeat_from_another_instance_keeps_job_running(jobs_dir):
    _running_job(jobs_dir, 999_999, "localhost:some-other-boot", age=600)
    stamp = time.time() - 5
    os.utime(jobs_dir / "abc.pid", (stamp, stamp))
    assert jobs.get_job("abc")["state"] == "running"


def test_live_pid_on_this_instance_keeps_job_running(jobs_dir):
    _running_job(jobs_dir, os.getpid(), jobs._instance_id(), age=600)
    assert jobs.get_job("abc")["state"] == "running"


def test_list_jobs_reaps_dead_job(jobs_dir):
    _running_job(jobs_dir, 999_999, jobs._instance_id())
    assert [j["state"] for j in jobs.list_jobs()] == ["error"]


def test_heartbeat_restamps_pid_file(jobs_dir, monkeypatch):
    monkeypatch.setattr(jobs, "HEARTBEAT_INTERVAL_SEC", 0.02)
    pid_file = jobs_dir / "abc.pid"
    with worker._heartbeat("abc"):
        deadline = time.time() + 5
        while not pid_file.is_file() and time.time() < deadline:
            time.sleep(0.01)
    assert json.loads(pid_file.read_text()) == {"pid": os.getpid(), "host": jobs._instance_id()}


def test_reaped_job_is_reported(jobs_dir, capsys):
    _running_job(jobs_dir, 999_999, jobs._instance_id())
    jobs.get_job("abc")
    assert "job_failed" in capsys.readouterr().err


def test_log_event_is_structured_on_cloud_run(monkeypatch, capsys):
    monkeypatch.setenv("K_SERVICE", "worker")
    jobs.log_event("job_failed", "boom", job_id="abc", user="a@b.c")
    entry = json.loads(capsys.readouterr().err)
    assert entry["severity"] == "ERROR" and entry["event"] == "job_failed"
    assert entry["job_id"] == "abc" and entry["user"] == "a@b.c"


def _queued(jobs_dir, age):
    write_job(jobs_dir / "q1.json",
              {"id": "q1", "state": "queued", "created_at": time.time() - age})


@pytest.fixture
def hosted(monkeypatch):
    triggered = []
    monkeypatch.setattr(jobs, "WORKER_URL", "https://worker.example")
    monkeypatch.setattr(jobs, "_trigger_worker", lambda: triggered.append(1))
    monkeypatch.setattr(jobs, "_reported_stuck", set())
    return triggered


def test_long_queued_job_with_no_worker_is_reported_once_and_retriggered(
        jobs_dir, hosted, capsys):
    _queued(jobs_dir, age=jobs.STUCK_QUEUED_AFTER_SEC + 60)
    jobs.list_jobs()
    jobs.list_jobs()
    assert capsys.readouterr().err.count("job_stuck_queued") == 1
    assert hosted == [1]


def test_queued_behind_a_running_job_is_not_stuck(jobs_dir, hosted, capsys):
    _queued(jobs_dir, age=jobs.STUCK_QUEUED_AFTER_SEC + 60)
    _running_job(jobs_dir, os.getpid(), jobs._instance_id())
    jobs.list_jobs()
    assert "job_stuck_queued" not in capsys.readouterr().err
    assert hosted == []


def test_recently_queued_job_is_not_stuck(jobs_dir, hosted, capsys):
    _queued(jobs_dir, age=30)
    jobs.list_jobs()
    assert hosted == []
