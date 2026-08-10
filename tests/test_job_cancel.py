"""Cancelling a running job has to work on Windows too.

cancel_job used to call os.killpg, which simply does not exist on Windows -
the AttributeError is not an OSError, so it slipped past cancel_job's except
clause and surfaced as a 500 in the browser. The job did still stop, via the
cooperative flag, which is what made the bug easy to miss: the UI errored
while the work quietly wound down anyway.

These cover the tree teardown itself, since a worker that dies while its
ffmpeg child keeps decoding into a pipe nobody reads is the failure the /T
flag exists to prevent.
"""
import json
import socket
import subprocess
import sys
import time

import pytest

from shot_clipper.label_ui import jobs
from shot_clipper.label_ui.jobs import _terminate_process_tree

# Parent spawns a child and reports its pid, mirroring worker -> ffmpeg.
_PARENT_SRC = (
    "import subprocess,sys,time;"
    "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
    "print(c.pid,flush=True);"
    "time.sleep(60)"
)


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    import os
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


@pytest.fixture
def process_tree():
    kwargs = {"stdout": subprocess.PIPE, "text": True}
    if sys.platform != "win32":
        kwargs["start_new_session"] = True  # matches _spawn_worker
    proc = subprocess.Popen([sys.executable, "-c", _PARENT_SRC], **kwargs)
    child_pid = int(proc.stdout.readline().strip())
    yield proc, child_pid
    for pid in (proc.pid, child_pid):
        try:
            _terminate_process_tree(pid)
        except OSError:
            pass


def test_terminates_worker_and_its_children(process_tree):
    proc, child_pid = process_tree
    assert _alive(proc.pid) and _alive(child_pid)

    _terminate_process_tree(proc.pid)

    assert _wait_gone(proc.pid), "worker survived cancellation"
    # The whole point of /T (and of the POSIX process group): an orphaned
    # ffmpeg would keep decoding a 4K video with nothing reading the pipe.
    assert _wait_gone(child_pid), "child orphaned by cancellation"


def test_missing_pid_raises_process_lookup_error():
    """cancel_job's except clause is written against ProcessLookupError, so
    both platforms have to report a vanished worker the same way."""
    with pytest.raises(ProcessLookupError):
        _terminate_process_tree(999_999)


def test_cancel_marks_job_stopped_when_worker_already_died(tmp_path, monkeypatch):
    """Stop has to work on a job whose worker is already gone.

    JOBS_DIR is a relative Path, so chdir puts the whole job store under
    tmp_path. The pid recorded here cannot exist, which is what a crashed or
    externally-killed worker leaves behind. That used to abort cancel_job
    before it wrote the final state, so the UI showed "running" until the
    180s stale reaper caught up and Stop appeared to do nothing.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "jobs").mkdir(parents=True)

    job_id = "deadworker01"
    jobs.write_job(jobs.job_path(job_id),
                   {"id": job_id, "state": "running", "created_at": time.time()})
    jobs._pid_path(job_id).write_text(
        json.dumps({"pid": 999_999, "host": socket.gethostname()}))

    assert jobs.cancel_job(job_id) is True

    job = jobs.get_job(job_id)
    assert job["state"] == "cancelled", "Stop left a dead job stuck at running"
    assert job["message"] == "cancelled by user"
