"""Background job tracking for the label UI's "process a new video" feature.
Jobs run as detached subprocesses (see worker.py), not in-process threads -
YOLO detection on a full video can take several minutes, and running it
in-process meant restarting the label UI's Flask server killed whatever job
was in flight. A detached subprocess (start_new_session=True) keeps going
regardless of what happens to the Flask process; get_job() just reads
whatever the subprocess has most recently written to disk.

Single-job-at-a-time: YOLO inference is heavy relative to a personal
machine's GPU/CPU, so running several at once would just contend with
itself.
"""
import subprocess
import sys
import time
import uuid

from .jobstore import JOBS_DIR, job_path, read_job, write_job


def _active_job() -> dict | None:
    if not JOBS_DIR.is_dir():
        return None
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job and job.get("state") == "running":
            return job
    return None


def start_job(spec: dict) -> str:
    """spec becomes the job's initial fields (video/folder/config paths/etc
    - whatever worker.py needs to do the work), merged with job bookkeeping
    fields. Raises RuntimeError if another job is already running."""
    active = _active_job()
    if active is not None:
        raise RuntimeError(f"another job is already running: {active['id']}")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "state": "running",
        "message": "starting",
        "created_at": time.time(),
        "error": None,
        **spec,
    }
    job_file = job_path(job_id)
    write_job(job_file, job)

    log_path = JOBS_DIR / f"{job_id}.log"
    with open(log_path, "w") as log_file:
        subprocess.Popen(
            [sys.executable, "-m", "shot_clipper.label_ui.worker", str(job_file)],
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,  # detach: survives the label UI process exiting
        )
    return job_id


def get_job(job_id: str) -> dict | None:
    return read_job(job_path(job_id))
