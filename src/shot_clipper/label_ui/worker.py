"""Standalone subprocess entry point that drains the label-UI job queue -
each job being one video, or a sequential batch of them. jobs.py launches
this as a detached subprocess (start_new_session=True) specifically so that
restarting the label UI's Flask process doesn't kill a job partway through -
this process keeps writing its progress to the job file on disk regardless
of whether the label UI is still around to read it.

Usage: python -m shot_clipper.label_ui.worker <job_file>

<job_file> already exists (written by jobs.start_job before spawning this
process). It's the job that triggered this worker, but it gets no special
treatment: this process runs every queued job oldest-first until none are
left, so several jobs submitted back-to-back are handled by one worker
rather than contending over the GPU.
"""
import contextlib
import json
import os
import shutil
import socket
import tempfile
import traceback
from pathlib import Path

from . import jobs, pipeline
from .jobstore import JobWriter, clear_cancel_flag, write_job

# Hosted deployment only: video_path normally points at the GCS-mounted
# volume, where gcsfuse is far slower than local disk both for detection's
# full sequential read of the file and for cutting's many per-clip seeks
# into that same file - measured on a real job: ~8MB/s effective read
# throughput and GPU utilization sitting at 3-7% the whole run, the file
# read is starving the GPU, not the reverse. Unset (the local/Docker
# default) skips this entirely - video_path is already on fast storage
# there, and copying it again would just waste time for no benefit.
#
# This costs memory, not disk: Cloud Run's only writable filesystem is a
# tmpfs backed by the instance's RAM (see the container runtime contract),
# so a local copy has to fit inside whatever's left of worker's memory
# allocation after CUDA/torch's own overhead - sized against that when
# choosing worker's --memory in cloudbuild.yaml, not assumed to scale for
# free to an arbitrarily large source video.
COPY_VIDEO_LOCALLY = os.environ.get("SHOT_CLIPPER_COPY_VIDEO_LOCALLY") == "1"


@contextlib.contextmanager
def _local_video(video_path: Path):
    if not COPY_VIDEO_LOCALLY:
        yield video_path
        return
    tmp_dir = Path(tempfile.mkdtemp(prefix="shot-clipper-src-"))
    try:
        local_path = tmp_dir / video_path.name
        shutil.copyfile(video_path, local_path)
        yield local_path
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_single(job: dict, writer: JobWriter) -> None:
    video_path = Path(job["video"])
    with _local_video(video_path) as local_video:
        result = pipeline.process_one_video(
            local_video, Path(job["config_path"]), Path(job["ground_truth_path"]),
            Path(job["out_dir"]), job["use_filter"], job, writer,
            fps=job.get("detect_fps"), reuse_detection=job.get("reuse_detection", False),
        )
    job["n_makes"] = result["n_makes"]
    job["clips_dir"] = result["clips_dir"]
    job["used_filter"] = job["use_filter"]
    job["duration_s"] = result.get("duration_s")
    job["resolution"] = result.get("resolution")
    job["fps"] = result.get("fps")
    job["size_mb"] = result.get("size_mb")
    if job["use_filter"]:
        job["n_kept"] = result["n_kept"]
        job["n_dropped"] = result["n_dropped"]
        job["message"] = (f"done: {result['n_kept']} candidate clips ready to label "
                           f"({result['n_dropped']} filtered out)")
    else:
        job["message"] = f"done: {result['n_makes']} candidate clips ready to label"


def _run_batch(job: dict, writer: JobWriter) -> None:
    queue = [Path(p) for p in job["queue"]]
    out_dir = Path(job["out_dir"])
    job["completed_videos"] = []
    writer.save(force=True)

    for i, video_path in enumerate(queue, start=1):
        job["current_video_index"] = i
        job["current_video"] = video_path.name
        writer.save(force=True)
        prefix = f"[{i}/{len(queue)}] {video_path.name}: "
        config_path = pipeline.CONFIGS_DIR / f"{video_path.stem}.json"
        ground_truth_path = pipeline.GROUND_TRUTH_DIR / f"{video_path.stem}_detected.json"
        with _local_video(video_path) as local_video:
            result = pipeline.process_one_video(
                local_video, config_path, ground_truth_path, out_dir, job["use_filter"],
                job, writer, prefix=prefix, fps=job.get("detect_fps"),
            )
        job["completed_videos"].append(result)
        writer.save(force=True)

    job["message"] = (f"done: {len(queue)} video(s) processed"
                       + (f", {len(job['skipped_uncalibrated'])} skipped (no calibration)"
                          if job.get("skipped_uncalibrated") else ""))


def _run_one(job_file: Path) -> None:
    job = json.loads(job_file.read_text())
    # Whoever queued this job wrote state="queued"; claiming it here (rather
    # than in the caller) is what makes "running" mean "a live process owns
    # this", which is exactly what jobs._reap_stale_jobs() relies on.
    job["state"] = "running"
    job["message"] = "starting"
    write_job(job_file, job)
    # Record ourselves as this job's owner. jobs._spawn_worker only writes a
    # pid for the job it was launched with, so every *subsequent* job this
    # worker drains had none - leaving cancel_job with no process to signal
    # and _worker_dead falling back to a 180s heartbeat before noticing a
    # killed worker.
    jobs._pid_path(job["id"]).write_text(
        json.dumps({"pid": os.getpid(), "host": socket.gethostname()}))
    writer = JobWriter(job, job_file)
    try:
        if job.get("kind") == "batch":
            _run_batch(job, writer)
        else:
            _run_single(job, writer)
        job["state"] = "done"
    except pipeline.JobCancelled:
        job["state"] = "cancelled"
        job["message"] = "cancelled by user"
    except ImportError as e:
        job["state"] = "error"
        job["error"] = (f"detection pipeline needs the `ml` extras "
                        f"(run `poetry install --with ml`) - import failed: {e}")
        job["traceback"] = traceback.format_exc()
    except Exception as e:  # noqa: BLE001 - reported via job["error"], not swallowed
        job["state"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
    write_job(job_file, job)
    clear_cancel_flag(job["id"])


def main() -> None:
    # argv[1] (the job that triggered this worker) is already on disk as
    # "queued" like any other, so it needs no special case: drain the whole
    # queue oldest-first and exit once it's empty. One worker per queue (see
    # jobs.start_job) keeps YOLO from contending with itself, and lets a
    # batch of jobs queued up front run through without another spawn.
    while True:
        nxt = jobs._next_queued_job()
        if nxt is None:
            return
        _, job_file = nxt
        _run_one(Path(job_file))


if __name__ == "__main__":
    main()
