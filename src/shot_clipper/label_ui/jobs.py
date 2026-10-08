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
from pathlib import Path

from .jobstore import JOBS_DIR, job_path, read_job, request_cancel, write_job

POLL_INTERVAL = 2.0
EXTERNAL_WORKER = os.environ.get("SHOT_CLIPPER_EXTERNAL_WORKER") == "1"
# Hosted deployment only: the worker Cloud Run service's own URL, scaled to
# zero between jobs. Unset (the local/Docker two-container setup) means
# there's nothing to wake - that worker container runs run_poller() as a
# permanently-live process instead, watching the same queue on its own.
WORKER_URL = os.environ.get("SHOT_CLIPPER_WORKER_URL")


# A worker that dies without writing a final state - the app was restarted,
# the machine slept, the process was killed - leaves its job file saying
# "running" forever. Since start_job() only spawns a worker when nothing is
# already running, one such file would wedge the queue permanently: every
# later job just sits at "queued" with nothing left alive to pick it up. So
# "running" on disk is treated as a claim to verify, not a fact.
STALE_AFTER_SEC = 180.0
# How often a live worker re-stamps its pid file (see worker._heartbeat) -
# well under STALE_AFTER_SEC, so a few slow gcsfuse writes in a row still
# don't make a healthy job look dead.
HEARTBEAT_INTERVAL_SEC = 20.0


# A job still "queued" this long with nothing running means no worker ever
# picked it up - see _check_stuck. Long enough that a worker's cold start
# (image pull + GPU attach, a couple of minutes) never trips it.
STUCK_QUEUED_AFTER_SEC = 900.0
_reported_stuck: set[str] = set()


def log_event(event: str, message: str, **fields) -> None:
    """Report a problem somewhere an operator will actually see it.

    Job failures used to live only in the job's own JSON file, which nobody
    looks at unless the affected user complains. On Cloud Run a JSON line on
    stderr becomes a structured Cloud Logging entry - `event` is what the
    alert policies filter on (see docs/ALERTING.md), `severity` is what makes
    it an error. Anywhere else it's just a readable line on the console.
    """
    if os.environ.get("K_SERVICE"):
        line = json.dumps({"severity": "ERROR", "event": event,
                           "message": f"shot-clipper {event}: {message}", **fields})
    else:
        line = f"[jobs] {event}: {message}"
    print(line, file=sys.stderr, flush=True)


def _log_job_failed(job: dict) -> None:
    log_event("job_failed", f"{job.get('error') or job.get('message')}",
              job_id=job.get("id"), user=job.get("user"),
              video=Path(job.get("video") or job.get("current_video") or "").name,
              error=str(job.get("error") or job.get("message"))[:500])


def _pid_path(job_id: str):
    return JOBS_DIR / f"{job_id}.pid"


def _instance_id() -> str:
    """Identifies the pid namespace a pid file was written from, so a reader
    can tell whether that pid means anything to it.

    The hostname alone is not enough: every Cloud Run container reports
    "localhost", so label-ui, worker, and each fresh worker instance after a
    crash all looked like the same machine. Confirmed live - a job whose
    container was OOM-killed stayed "running" for days, because the pid it
    recorded happened to be alive (as some unrelated process) in whichever
    container checked. The kernel's boot id is unique per Cloud Run instance;
    where it doesn't exist (macOS, Windows) the hostname is what it always
    was, a real machine name.
    """
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = ""
    return f"{socket.gethostname()}:{boot_id}" if boot_id else socket.gethostname()


def write_pid_file(job_id: str, pid: int) -> None:
    _pid_path(job_id).write_text(json.dumps({"pid": pid, "host": _instance_id()}))


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


def _terminate_process_tree(pid: int) -> None:
    """Stop the worker and the ffmpeg/YOLO children it spawned.

    POSIX: _spawn_worker gave the worker its own session, so a single killpg
    reaches the whole group.

    Windows has no os.killpg at all - reaching it raises AttributeError, which
    is not an OSError and so sailed straight past cancel_job's except clause
    and out to the browser as a 500. os.kill is no substitute either: on
    Windows it ignores the signal and calls TerminateProcess against that one
    pid, which would stop the worker and orphan the ffmpeg it spawned, leaving
    it decoding into a pipe nobody reads. taskkill /T walks the child tree.

    Raises ProcessLookupError when the pid is already gone, so both platforms
    fail the same way.
    """
    if os.name != "nt":
        os.killpg(pid, signal.SIGTERM)
        return

    proc = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return
    # 128 is taskkill's "no such process". Anything else is a real failure and
    # must not be swallowed, or cancel_job would record a kill that never
    # landed and mark a still-running job "cancelled".
    if proc.returncode == 128:
        raise ProcessLookupError(pid)
    raise OSError(f"taskkill failed (exit {proc.returncode}): "
                  f"{(proc.stderr or '').strip()}")


def _worker_dead(job: dict) -> bool:
    """Has the worker behind this "running" job gone away?"""
    job_id = job.get("id")
    pid_file = _pid_path(job_id)
    if pid_file.is_file():
        try:
            info = json.loads(pid_file.read_text())
            # a pid from another container's namespace means nothing here
            if info.get("host") == _instance_id():
                alive = _process_alive(int(info["pid"]))
                if alive is not None:
                    return not alive
        except (ValueError, TypeError, json.JSONDecodeError, KeyError, OSError):
            pass
    # Couldn't ask the OS - the worker is in another container, or was in
    # one that no longer exists. Fall back to "has it shown signs of life
    # lately?": a live worker re-stamps its pid file every
    # HEARTBEAT_INTERVAL_SEC for as long as the job runs (worker._heartbeat),
    # on top of rewriting the job file as progress comes in, so both going
    # untouched this long means nothing is alive behind them.
    mtimes = []
    for path in (job_path(job_id), pid_file):
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    if not mtimes:
        return False
    return time.time() - max(mtimes) > STALE_AFTER_SEC


def _reap_if_dead(job_file, job: dict) -> bool:
    """Mark one "running" job as errored if its worker is gone. Mutates
    `job` in place so a caller holding it sees the corrected state."""
    if job.get("state") != "running" or not _worker_dead(job):
        return False
    job["state"] = "error"
    job["error"] = ("worker stopped before finishing - the app was restarted "
                    "or the process was killed. Re-run this job to try again.")
    job["message"] = "interrupted"
    write_job(job_file, job)
    _log_job_failed(job)
    return True


def _reap_stale_jobs() -> None:
    """Mark jobs whose worker vanished as errored, so they stop blocking the
    queue and show up honestly in Job Status instead of as forever-running."""
    if not JOBS_DIR.is_dir():
        return
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job:
            _reap_if_dead(f, job)


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
            # POSIX only (Windows subprocess ignores it) - gives the worker its
            # own process group so cancel_job can killpg the whole thing. On
            # Windows _terminate_process_tree walks the child tree instead.
            start_new_session=True,
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
    write_pid_file(job_id, proc.pid)
    return proc


def _trigger_worker() -> None:
    """Wake the hosted worker Cloud Run service - a fire-and-forget ping,
    not a wait for it to finish draining the queue. Job Status polls
    get_job() independently, reading the same GCS-mounted job files the
    worker writes to, so nothing here needs to wait for a response. A
    short read timeout is expected to fire (the worker is busy actually
    processing, not replying) and isn't treated as a failure - the job
    stays queued regardless, and a later trigger or the next submission
    picks it up if this particular ping didn't land."""
    if not WORKER_URL:
        return
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token
        import requests
        token = google.oauth2.id_token.fetch_id_token(
            google.auth.transport.requests.Request(), WORKER_URL)
        try:
            resp = requests.post(f"{WORKER_URL}/drain",
                                 headers={"Authorization": f"Bearer {token}"}, timeout=(3, 1))
        except requests.exceptions.ReadTimeout:
            return  # the expected outcome: it's busy draining, not replying
        if resp.status_code >= 400:
            log_event("worker_trigger_failed",
                      f"worker answered /drain with HTTP {resp.status_code}",
                      status=resp.status_code)
    except Exception as e:  # noqa: BLE001 - must never fail the submission itself
        log_event("worker_trigger_failed", f"could not reach worker: {e}")


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
    else:
        _trigger_worker()
    return job_id


def drain_queue_once() -> None:
    """Process every currently-queued job, oldest first, then return - the
    request-triggered counterpart to run_poller(): a Cloud Run instance
    only keeps billing while a request is in flight, so this has to return
    once there's nothing left to do rather than sleep and poll forever."""
    _reap_stale_jobs()
    while True:
        picked = _next_queued_job()
        if picked is None:
            return
        job_id, job_file = picked
        proc = _spawn_worker(job_id, job_file)
        proc.wait()
        # The worker subprocess's own output goes to its .log file beside the
        # job, not to this process's stderr - so this is the one place a
        # hosted job's failure can be reported to Cloud Logging. Reaping
        # first covers a subprocess that died without writing a final state
        # (it reports the failure itself).
        job = read_job(job_file)
        if job and not _reap_if_dead(job_file, job) and job.get("state") == "error":
            _log_job_failed(job)


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
    # Reaped on read, not only when the next job is submitted: Job Status
    # polls this, and on the hosted deployment nothing else runs between a
    # worker dying (e.g. OOM-killed) and someone queuing more work - so the
    # dead job otherwise shows "running" indefinitely.
    job = read_job(job_path(job_id))
    if job:
        _reap_if_dead(job_path(job_id), job)
    return job


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
    all_jobs = []
    for f in JOBS_DIR.glob("*.json"):
        job = read_job(f)
        if job:
            _reap_if_dead(f, job)  # see get_job
            all_jobs.append(job)
    _check_stuck(all_jobs)
    all_jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return all_jobs[:limit]


def _check_stuck(all_jobs: list[dict]) -> None:
    """Notice queued jobs that no worker is coming for, and try again.

    Hosted deployment only: the worker is woken by a single fire-and-forget
    ping at submission time (_trigger_worker), and after a worker crash the
    jobs queued behind the one that died are left with nothing to wake it
    again. Queued-for-a-while is only a problem when nothing is running -
    behind a long job it's just a queue. Reported once per job per process;
    the alert policy's own rate limit handles the rest.
    """
    if not WORKER_URL or any(j.get("state") == "running" for j in all_jobs):
        return
    now = time.time()
    stuck = [j for j in all_jobs
             if j.get("state") == "queued" and j.get("id") not in _reported_stuck
             and now - j.get("created_at", now) > STUCK_QUEUED_AFTER_SEC]
    if not stuck:
        return
    for job in stuck:
        _reported_stuck.add(job["id"])
        log_event("job_stuck_queued",
                  f"queued {int((now - job['created_at']) / 60)} min with no worker running",
                  job_id=job["id"], user=job.get("user"),
                  video=Path(job.get("video") or "").name)
    _trigger_worker()


def cancel_job(job_id: str) -> bool:
    """Stop a queued-but-not-started job outright, or ask a running one to
    stop. Two mechanisms, since we might not be in the same container as
    the worker:

    - request_cancel() sets a cooperative flag that pipeline.py checks
      during detection - works no matter which container the worker ended
      up in, but isn't instant (only checked between frame batches).
    - A direct SIGTERM to the worker's process group, for an instant stop -
      but only if _pid_path's recorded instance id matches ours, i.e. we're
      actually in the same pid namespace as whoever spawned it. A raw pid
      number from a different container's namespace could be unassigned
      there, or worse, could belong to some unrelated process.

    Returns False if the job isn't queued/running."""
    # read_job, not get_job: get_job would reap a dead worker's job to
    # "error" first, and Stop on such a job should record "cancelled".
    job = read_job(job_path(job_id))
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
            if info.get("host") == _instance_id():
                try:
                    _terminate_process_tree(int(info["pid"]))
                except ProcessLookupError:
                    # Worker already gone - it crashed, or was killed with the
                    # terminal that spawned it. Pressing Stop used to do nothing
                    # visible here: the write below was skipped, so the job sat
                    # at "running" until the 180s stale reaper noticed. A pid
                    # that no longer exists is proof the job isn't running, so
                    # record the state rather than treating it as a failure.
                    pass
                # a killed process can't write its own final state, so we
                # do it here - but only when we know the job is really stopped
                # (see docstring)
                job["state"] = "cancelled"
                job["message"] = "cancelled by user"
                write_job(job_path(job_id), job)
        # OSError covers ProcessLookupError/PermissionError plus a taskkill
        # that failed or isn't installed. Cancelling must never 500: the
        # cooperative flag set above stops the job either way, just not
        # instantly.
        except (ValueError, TypeError, json.JSONDecodeError, KeyError, OSError):
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
