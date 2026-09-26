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

# Overridable so the hosted deployment can point this at the GCS-mounted
# volume (e.g. /data/_jobs) instead of the container's own ephemeral
# filesystem - job history written under the plain "data/jobs" default would
# otherwise vanish every time a scale-to-zero Cloud Run instance recycles.
# Shared across all users by design (see app.py's per-job "user" field,
# checked at the API layer) rather than split per-user - one GPU worker
# drains one shared queue regardless of who queued what.
JOBS_DIR = Path(os.environ.get("SHOT_CLIPPER_JOBS_DIR", "data/jobs"))


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def cancel_flag_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.cancel"


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


def write_job(job_file: Path, job: dict) -> None:
    job_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = job_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job))
    os.replace(tmp, job_file)


def read_job(job_file: Path) -> dict | None:
    """Read job file with retry logic for transient filesystem hiccups.

    On Windows, the worker subprocess writing to the file can temporarily
    block the main process from reading it (PermissionError). On the hosted
    deployment, write_job()'s atomic os.replace() has no true equivalent on
    the GCS-mounted volume - gcsfuse emulates rename as copy-then-delete,
    so a reader that already globbed the directory a moment before a
    concurrent writer (the worker saves progress roughly every second
    while a job runs) replaced the file can see the old file disappear out
    from under it, surfacing as OSError: [Errno 116] Stale file handle -
    confirmed live once label-ui's ~2s job-status polling and a running
    job's writes actually overlapped for real. Retrying is correct either
    way: the file is momentarily unavailable, not gone. is_file() moved
    inside the loop too, since the same stale-handle error can surface on
    the stat call, not just the read.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not job_file.is_file():
                return None
            return json.loads(job_file.read_text())
        except OSError:
            if attempt < max_retries - 1:
                time.sleep(0.05)  # 50ms delay before retry
            else:
                # Last attempt failed, return None to avoid crash
                return None
        except json.JSONDecodeError:
            return None
    return None


class JobWriter:
    """Wraps write_job with a time-based throttle so per-frame progress
    callbacks (thousands of calls over a long video) don't turn into
    thousands of file writes. save(force=True) bypasses the throttle for
    state transitions that should be visible right away."""

    def __init__(self, job: dict, job_file: Path, min_interval: float = 1.0):
        self.job = job
        self.job_file = job_file
        self.min_interval = min_interval
        self._last = 0.0

    def save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last) < self.min_interval:
            return
        self._last = now
        write_job(self.job_file, self.job)
