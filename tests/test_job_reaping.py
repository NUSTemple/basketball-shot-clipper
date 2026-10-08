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
