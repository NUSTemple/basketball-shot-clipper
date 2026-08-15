"""Writing job status must survive a locked file, and must never kill the job.

On Windows os.replace() fails with ERROR_ACCESS_DENIED while any other
process holds the destination open - Python's open() doesn't grant
FILE_SHARE_DELETE. Two openers collide here constantly: the Flask app
polling job status against the worker writing progress, and a sync client
(OneDrive) opening each freshly written file when data/ lives inside a
synced folder.

read_job had retries for this; write_job didn't. A progress save raising
PermissionError propagated out of pipeline.process_one_video into
worker._run_one's catch-all, which marked the job "error" - throwing away a
finished hour of GPU detection over a 200-byte status file, in one observed
case at "cutting clips: 36/39".
"""
import json
import sys
import threading
import time

import pytest

from shot_clipper.label_ui import jobstore
from shot_clipper.label_ui.jobstore import JobWriter, read_job, write_job


@pytest.fixture
def job_file(tmp_path):
    return tmp_path / "jobs" / "abc123.json"


def test_write_job_round_trips(job_file):
    write_job(job_file, {"id": "abc123", "state": "running"})
    assert read_job(job_file) == {"id": "abc123", "state": "running"}


def test_write_job_retries_a_transient_lock(job_file, monkeypatch):
    """The first two replaces fail the way Windows does; the third lands."""
    real_replace = jobstore.os.replace
    attempts = []

    def flaky_replace(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(jobstore.os, "replace", flaky_replace)
    monkeypatch.setattr(jobstore.time, "sleep", lambda _: None)

    write_job(job_file, {"id": "abc123", "state": "done"})

    assert len(attempts) == 3
    assert read_job(job_file)["state"] == "done"


def test_write_job_raises_once_retries_are_exhausted(job_file, monkeypatch):
    """A lifecycle write that genuinely can't land has to be loud - worker
    ._run_one is writing the final state and the caller needs to know."""
    def always_denied(src, dst):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(jobstore.os, "replace", always_denied)
    monkeypatch.setattr(jobstore.time, "sleep", lambda _: None)

    with pytest.raises(PermissionError):
        write_job(job_file, {"id": "abc123"})

    # and it doesn't litter a half-written .tmp for the next run to trip over
    assert not job_file.with_suffix(".json.tmp").exists()


def test_progress_save_never_raises(job_file, monkeypatch, capsys):
    """The whole point: a dropped progress update is a stale percentage in
    the UI, not a failed job."""
    monkeypatch.setattr(jobstore.os, "replace", _denied)
    monkeypatch.setattr(jobstore.time, "sleep", lambda _: None)

    writer = JobWriter({"id": "abc123", "message": "scanning"}, job_file)
    writer.save(force=True)  # must not raise

    assert writer.dropped == 1
    assert "could not write job progress" in capsys.readouterr().out


def test_progress_recovers_with_current_state(job_file, monkeypatch):
    """save() writes the whole job dict, so the next save that gets through
    carries the progress the dropped ones were meant to report."""
    monkeypatch.setattr(jobstore.time, "sleep", lambda _: None)
    job = {"id": "abc123", "message": "scanning: 10%"}
    writer = JobWriter(job, job_file, min_interval=0.0)

    monkeypatch.setattr(jobstore.os, "replace", _denied)
    writer.save(force=True)
    job["message"] = "scanning: 20%"
    writer.save(force=True)
    assert writer.dropped == 2

    monkeypatch.undo()
    job["message"] = "scanning: 30%"
    writer.save(force=True)

    assert writer.dropped == 0
    assert read_job(job_file)["message"] == "scanning: 30%"


def test_repeated_failures_warn_sparsely(job_file, monkeypatch, capsys):
    """Once per second for an hour would bury the log that explains it."""
    monkeypatch.setattr(jobstore.os, "replace", _denied)
    monkeypatch.setattr(jobstore.time, "sleep", lambda _: None)

    writer = JobWriter({"id": "abc123"}, job_file, min_interval=0.0)
    for _ in range(JobWriter.WARN_EVERY):
        writer.save(force=True)

    # the 1st and the WARN_EVERY-th, not all of them
    assert capsys.readouterr().out.count("could not write job progress") == 2


def _denied(src, dst):
    raise PermissionError(13, "Access is denied")


@pytest.mark.skipif(sys.platform != "win32",
                    reason="only Windows refuses a replace onto an open file")
def test_survives_a_real_reader_holding_the_file(job_file):
    """The actual bug, unmocked: something else has the job file open while
    the worker replaces it. This is what the Flask app's status polling and
    OneDrive's uploader each do."""
    write_job(job_file, {"id": "abc123", "message": "first"})
    released = threading.Event()

    def hold_it_open():
        with job_file.open() as fh:
            fh.read()
            time.sleep(0.15)
        released.set()

    holder = threading.Thread(target=hold_it_open)
    holder.start()
    time.sleep(0.02)  # let it get the handle before we try to replace

    write_job(job_file, {"id": "abc123", "message": "second"})

    holder.join()
    assert released.is_set(), "the write should have waited out the lock"
    assert json.loads(job_file.read_text())["message"] == "second"
