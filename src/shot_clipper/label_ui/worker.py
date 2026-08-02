"""Standalone subprocess entry point for one label-UI detection job (single
video, or a sequential batch). jobs.py launches this as a detached
subprocess (start_new_session=True) specifically so that restarting the
label UI's Flask process doesn't kill a job partway through - this process
keeps writing its progress to the job file on disk regardless of whether
the label UI is still around to read it.

Usage: python -m shot_clipper.label_ui.worker <job_file>

<job_file> already exists (written by jobs.start_job before spawning this
process) and holds the job spec plus initial state; this process reads it,
does the work, and rewrites it in place as progress is made.
"""
import json
import sys
import traceback
from pathlib import Path

from . import pipeline
from .jobstore import JobWriter, clear_cancel_flag, write_job


def _run_single(job: dict, writer: JobWriter) -> None:
    video_path = Path(job["video"])
    result = pipeline.process_one_video(
        video_path, Path(job["config_path"]), Path(job["ground_truth_path"]),
        Path(job["out_dir"]), job["use_filter"], job, writer,
        fps=job.get("detect_fps"),
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
        result = pipeline.process_one_video(
            video_path, config_path, ground_truth_path, out_dir, job["use_filter"],
            job, writer, prefix=prefix, fps=job.get("detect_fps"),
        )
        job["completed_videos"].append(result)
        writer.save(force=True)

    job["message"] = (f"done: {len(queue)} video(s) processed"
                       + (f", {len(job['skipped_uncalibrated'])} skipped (no calibration)"
                          if job.get("skipped_uncalibrated") else ""))


def main() -> None:
    job_file = Path(sys.argv[1])
    job = json.loads(job_file.read_text())
    # jobs.start_job() writes state="queued" and either spawns this process
    # right away or leaves it for run_poller() to pick up later - either
    # way, this is the moment the job actually starts, so mark it here
    # rather than relying on whichever caller to have done it already
    job["state"] = "running"
    job["message"] = "starting"
    write_job(job_file, job)
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
    except ImportError:
        job["state"] = "error"
        job["error"] = "detection pipeline needs the `ml` extras: run `poetry install --with ml`"
    except Exception as e:  # noqa: BLE001 - reported via job["error"], not swallowed
        job["state"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
    write_job(job_file, job)
    clear_cancel_flag(job["id"])


if __name__ == "__main__":
    main()
