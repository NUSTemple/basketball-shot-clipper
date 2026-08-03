"""Background job tracking for the label UI's "process a new video" feature.

Every job is queued to disk first (see jobstore.py) - who actually starts
it depends on SHOT_CLIPPER_EXTERNAL_WORKER:

- Unset (native, or a single-container Docker setup): start_job() spawns
  the worker subprocess itself, immediately, detached
  (start_new_session=True) so it survives the label UI's Flask process
  restarting.
- Set (the two-container Docker setup - see docker-compose.yml's `worker`
  service): start_job() only queues; run_poller(), running as that
  service's own long-lived process, picks up queued jobs from the shared
  data/jobs/ volume and runs them one at a time. Now restarting the *web*
  container can't touch a job at all - it's a different container's
  process tree entirely, not just a detached one within the same container.

Either way, get_job() just reads whatever's most recently been written to
disk, and cancel_job() signals the worker subprocess's process group -
callers don't need to know which mode is active.

Single-job-at-a-time: YOLO inference is heavy relative to a personal
machine's GPU/CPU, so running several at once would just contend with
itself.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid

from .jobstore import JOBS_DIR, job_path, read_job, request_cancel, write_job

POLL_INTERVAL = 2.0
EXTERNAL_WORKER = os.environ.get("SHOT_CLIPPER_EXTERNAL_WORKER") == "1"


# A worker that dies without writing a final state - the app was restarted,
# the machine slept, the process was killed - leaves its job file saying
# "running" forever. Since start_job() only spawns a worker when nothing is
# already running, one such file would wedge the queue permanently: every
# later job just sits at "queued" with nothing left alive to pick it up. So
# "running" on disk is treated as a claim to verify, not a fact.
STALE_AFTER_SEC = 180.0


def _pid_path(job_id: str):
    return JOBS_DIR / f"{job_id}.pid"


def _process_alive(pid: int) -> bool | None:
    """True/False when we can tell, None when we can't.

    Deliberately never signals the pid: os.kill(pid, 0) is a liveness probe
    on POSIX, but on Windows os.kill ignores the signal number and calls
    TerminateProcess - it would kill the very worker it's asking about.
    """
    try:
        import psutil
    except ImportError:
        if os.name == "nt":
            return None  # no safe probe available - caller falls back to mtime
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, just not ours to signal
        return True
    return psutil.pid_exists(pid)


def _worker_dead(job: dict) -> bool:
    """Has the worker behind this "running" job gone away?"""
    job_id = job.get("id")
    pid_file = _pid_path(job_id)
    if pid_file.is_file():
        try:
            info = json.loads(pid_file.read_text())
            # a pid from another container's namespace means nothing here
            if info.get("host") == socket.gethostname():
                alive = _process_alive(int(info["pid"]))
                if alive is not None:
                    return not alive
        except (ValueError, TypeError, json.JSONDecodeError, KeyError, OSError):
            pass
    # Couldn't ask the OS. Fall back to "has it reported progress lately?" -
    # a live worker rewrites its job file at least once a second while
    # scanning (see jobstore.JobWriter), so a long-untouched file is dead.
    try:
        age = time.time() - job_path(job_id).stat().st_mtime
    except OSError:
        return False
    return age > STALE_AFTER_SEC


def _reap_stale_jobs() -> None:
    """Mark jobs whose worker vanished as errored, so they stop blocking the
    queue and show up honestly in Job Status instead of as forever-running."""
    if not JOBS_DIR.is_dir():
        return
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if not job or job.get("state") != "running":
            continue
        if _worker_dead(job):
            job["state"] = "error"
            job["error"] = ("worker stopped before finishing - the app was restarted "
                            "or the process was killed. Re-run this job to try again.")
            job["message"] = "interrupted"
            write_job(f, job)


def _active_job() -> dict | None:
    """Return the first job that's queued or running (oldest first)."""
    if not JOBS_DIR.is_dir():
        return None
    jobs = []
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job and job.get("state") in ("queued", "running"):
            jobs.append((job.get("created_at", 0), job))
    if jobs:
        return sorted(jobs, key=lambda entry: entry[0])[0][1]
    return None


def _has_running_job() -> bool:
    """Is a worker actually chewing on a job right now? Call _reap_stale_jobs()
    first so a dead worker's leftover claim doesn't count."""
    if not JOBS_DIR.is_dir():
        return False
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job and job.get("state") == "running":
            return True
    return False


def _spawn_worker(job_id: str, job_file) -> subprocess.Popen:
    log_path = JOBS_DIR / f"{job_id}.log"
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "shot_clipper.label_ui.worker", str(job_file)],
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group, so cancel_job can killpg it
        )
    # a sidecar file, not a field on the job dict itself - the worker also
    # reads/rewrites that same job file, and a write race could silently
    # drop the pid (the worker's in-memory copy, loaded before this line
    # runs, doesn't have it, so its next save() would overwrite it away).
    # Records the hostname too: in the two-container setup this file is
    # written by the `worker` container but read by `label-ui` in
    # cancel_job() - pid numbers are only meaningful within the container
    # (pid namespace) that assigned them, so cancel_job must be able to
    # tell whether "this pid" actually means anything to it.
    _pid_path(job_id).write_text(json.dumps({"pid": proc.pid, "host": socket.gethostname()}))
    return proc


def start_job(spec: dict) -> str:
    """Queue a new job. If no job is currently running, spawn a worker to
    process it immediately. Otherwise it waits in the queue for the current
    job to finish. Multiple jobs can be queued and will run sequentially."""
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "state": "queued",
        "message": "queued",
        "created_at": time.time(),
        "error": None,
        **spec,
    }
    job_file = job_path(job_id)
    write_job(job_file, job)

    # One worker drains the whole queue (see worker.main), so we only need to
    # start one when none is live. Reap first: without that, a single dead
    # worker's stale "running" file blocks every future job forever.
    if not EXTERNAL_WORKER:
        _reap_stale_jobs()
        if not _has_running_job():
            _spawn_worker(job_id, job_file)
    return job_id


def kick_queue() -> str | None:
    """Resume queued work left over from a previous run.

    Workers are only ever spawned by start_job, so a job still sitting at
    "queued" when the app restarts - because the previous worker was killed
    mid-drain - would wait forever for someone to submit something new.
    Called at startup; returns the job it started, if any.
    """
    if EXTERNAL_WORKER:
        return None  # the worker container's poller owns this
    _reap_stale_jobs()
    if _has_running_job():
        return None
    picked = _next_queued_job()
    if picked is None:
        return None
    job_id, job_file = picked
    _spawn_worker(job_id, job_file)
    return job_id


def get_job(job_id: str) -> dict | None:
    return read_job(job_path(job_id))


def get_queue_position(job_id: str) -> int | None:
    """Get queue position (1-indexed) for a job, or None if not queued/running."""
    if not JOBS_DIR.is_dir():
        return None

    jobs = []
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job and job.get("state") in ("queued", "running"):
            jobs.append((job.get("created_at", 0), job.get("id")))

    jobs.sort()
    for idx, (_, jid) in enumerate(jobs, start=1):
        if jid == job_id:
            return idx
    return None


def list_jobs(limit: int = 50) -> list[dict]:
    """Every job this label UI has ever run, newest first - data/jobs/*.json
    files are never auto-deleted, so this is genuine history, not just "the
    current job". Lets Detect offer a "rerun with the same settings" action
    instead of re-typing/re-browsing everything. Capped to keep the
    response small over a long-lived install."""
    if not JOBS_DIR.is_dir():
        return []
    all_jobs = [job for job in (read_job(f) for f in JOBS_DIR.glob("*.json")) if job]
    all_jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return all_jobs[:limit]


def cancel_job(job_id: str) -> bool:
    """Stop a queued-but-not-started job outright, or ask a running one to
    stop. Two mechanisms, since we might not be in the same container as
    the worker:

    - request_cancel() sets a cooperative flag that pipeline.py checks
      during detection - works no matter which container the worker ended
      up in, but isn't instant (only checked between frame batches).
    - A direct SIGTERM to the worker's process group, for an instant stop -
      but only if _pid_path's recorded hostname matches ours, i.e. we're
      actually in the same pid namespace as whoever spawned it. A raw pid
      number from a different container's namespace could be unassigned
      there, or worse, could belong to some unrelated process.

    Returns False if the job isn't queued/running."""
    job = get_job(job_id)
    if job is None or job.get("state") not in ("queued", "running"):
        return False

    if job["state"] == "queued":
        job["state"] = "cancelled"
        job["message"] = "cancelled by user"
        write_job(job_path(job_id), job)
        return True

    request_cancel(job_id)
    pid_file = _pid_path(job_id)
    if pid_file.is_file():
        try:
            info = json.loads(pid_file.read_text())
            if info.get("host") == socket.gethostname():
                os.killpg(info["pid"], signal.SIGTERM)
                # a killed process can't write its own final state, so we
                # do it here - but only when we know the kill actually hit
                # the right target (see docstring)
                job["state"] = "cancelled"
                job["message"] = "cancelled by user"
                write_job(job_path(job_id), job)
        except (ValueError, TypeError, json.JSONDecodeError, KeyError,
                ProcessLookupError, PermissionError):
            pass  # cooperative flag above still applies regardless
    return True


def _next_queued_job() -> tuple[str, "os.PathLike"] | None:
    if not JOBS_DIR.is_dir():
        return None
    queued = []
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job and job.get("state") == "queued":
            queued.append((job.get("created_at", 0), job["id"], f))
    if not queued:
        return None
    queued.sort(key=lambda entry: entry[0])
    _, job_id, job_file = queued[0]
    return job_id, job_file


def run_poller() -> None:
    """Entry point for the separate `worker` container/service
    (SHOT_CLIPPER_EXTERNAL_WORKER=1) - watches data/jobs/ for queued work
    and runs one job at a time to completion. See module docstring."""
    print(f"[worker] watching {JOBS_DIR} for queued jobs (poll every {POLL_INTERVAL}s)", flush=True)
    while True:
        picked = _next_queued_job()
        if picked is None:
            time.sleep(POLL_INTERVAL)
            continue
        job_id, job_file = picked
        print(f"[worker] starting job {job_id}", flush=True)
        proc = _spawn_worker(job_id, job_file)
        proc.wait()
        print(f"[worker] job {job_id} finished", flush=True)


if __name__ == "__main__":
    run_poller()
