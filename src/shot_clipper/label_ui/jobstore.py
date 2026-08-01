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

JOBS_DIR = Path("data/jobs")


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def write_job(job_file: Path, job: dict) -> None:
    job_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = job_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job))
    os.replace(tmp, job_file)


def read_job(job_file: Path) -> dict | None:
    if not job_file.is_file():
        return None
    try:
        return json.loads(job_file.read_text())
    except json.JSONDecodeError:
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
