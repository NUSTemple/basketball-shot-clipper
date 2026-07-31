"""In-memory background job tracking for the label UI's "process a new
video" feature (detect + clip run server-side, since a full video can take
several minutes). Single-process, single-worker: only one job runs at a time
since YOLO inference is heavy relative to a personal machine's GPU/CPU.
"""
import threading
import time
import traceback
import uuid

_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_active_job_id: str | None = None


def start_job(meta: dict, work) -> str:
    """work(job) runs in a background thread and should mutate job in place
    (job["message"] = ... for progress) as it runs. Raises RuntimeError if
    another job is already in progress.
    """
    global _active_job_id
    with _lock:
        if _active_job_id is not None and _jobs[_active_job_id]["state"] == "running":
            raise RuntimeError(f"another job is already running: {_active_job_id}")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "state": "running",
            "message": "starting",
            "created_at": time.time(),
            "error": None,
            **meta,
        }
        _jobs[job_id] = job
        _active_job_id = job_id

    def run():
        try:
            work(job)
            job["state"] = "done"
        except Exception as e:  # noqa: BLE001 - reported to the caller via job["error"], not swallowed
            job["state"] = "error"
            job["error"] = str(e)
            job["traceback"] = traceback.format_exc()

    threading.Thread(target=run, daemon=True).start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        return _jobs.get(job_id)
