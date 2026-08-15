"""Atomic on-disk job status storage shared by jobs.py (the Flask app's
view) and worker.py (the detached subprocess that does the actual work).
Using a file instead of an in-memory dict is what lets a job survive the
label UI's Flask process restarting - see jobs.py for why the work runs in
a subprocess at all.
"""
import json
import os
import time
from pathlib import Path

from .. import paths


def jobs_dir() -> Path:
    return paths.jobs_dir()


def job_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"


def cancel_flag_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.cancel"


def request_cancel(job_id: str) -> None:
    """Cooperative stop signal, checked by pipeline.py during detection -
    the only mechanism that works regardless of whether the worker ended up
    in the same container/pid-namespace as whoever's asking (see
    jobs.cancel_job): a raw pid number from a different container means
    nothing there, or worse, could belong to an unrelated process."""
    cancel_flag_path(job_id).parent.mkdir(parents=True, exist_ok=True)
    cancel_flag_path(job_id).touch()


def cancel_requested(job_id: str) -> bool:
    return cancel_flag_path(job_id).is_file()


def clear_cancel_flag(job_id: str) -> None:
    cancel_flag_path(job_id).unlink(missing_ok=True)


# Windows fails os.replace() with ERROR_ACCESS_DENIED if *any* process has
# the destination open - Python's open() doesn't grant FILE_SHARE_DELETE - so
# the write side needs the same retry the read side below already has. Two
# processes routinely hold a job file open here: the Flask app polling status
# while the worker writes progress, and, when data/ lives inside a synced
# folder (OneDrive, Dropbox), the sync engine opening each freshly written
# file to upload it. Both locks are brief, so a short backoff clears them.
WRITE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4)


def write_job(job_file: Path, job: dict) -> None:
    """Atomically replace `job_file` with `job`, retrying transient Windows
    sharing violations. Raises the last OSError if every attempt fails -
    callers writing a lifecycle state (see worker._run_one) want to know."""
    job_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = job_file.with_suffix(".json.tmp")
    payload = json.dumps(job)
    for delay in (*WRITE_RETRY_DELAYS, None):
        try:
            tmp.write_text(payload)
            os.replace(tmp, job_file)
            return
        except OSError:
            if delay is None:
                # Don't leave a half-written .tmp behind for the next run to
                # trip over; failing to clean up is not worth masking the
                # original error.
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            time.sleep(delay)


def read_job(job_file: Path) -> dict | None:
    """Read job file with retry logic for Windows file locking.

    On Windows, the worker subprocess writing to the file can temporarily
    block the main process from reading it. Retry a few times with small
    delays rather than failing immediately.
    """
    if not job_file.is_file():
        return None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            return json.loads(job_file.read_text())
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.05)  # 50ms delay before retry
            else:
                # Last attempt failed, return None to avoid crash
                return None
        except json.JSONDecodeError:
            return None


class JobWriter:
    """Wraps write_job with a time-based throttle so per-frame progress
    callbacks (thousands of calls over a long video) don't turn into
    thousands of file writes. save(force=True) bypasses the throttle for
    state transitions that should be visible right away.

    Every save here is best-effort: a progress update that can't reach disk
    is a stale percentage in the UI, and it must never be the thing that
    ends the job. It used to be - a sharing violation on the status file
    would propagate out of the pipeline into worker._run_one's catch-all and
    mark an hour of finished GPU work as "error" at 64%. The next successful
    save carries the current progress anyway, since save() always writes the
    whole job dict rather than a delta, and if the *worker* really has died
    jobs._reap_stale_jobs still notices via the heartbeat.
    """

    # How many consecutive silent failures before saying something. The first
    # one is worth a line in the worker log; a persistent problem is worth
    # repeating, but not once per second.
    WARN_EVERY = 50

    def __init__(self, job: dict, job_file: Path, min_interval: float = 1.0):
        self.job = job
        self.job_file = job_file
        self.min_interval = min_interval
        self._last = 0.0
        self.dropped = 0

    def save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last) < self.min_interval:
            return
        self._last = now
        try:
            write_job(self.job_file, self.job)
        except OSError as e:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % self.WARN_EVERY == 0:
                print(f"warning: could not write job progress "
                      f"({self.dropped} update(s) dropped so far): {e}",
                      flush=True)
        else:
            self.dropped = 0
