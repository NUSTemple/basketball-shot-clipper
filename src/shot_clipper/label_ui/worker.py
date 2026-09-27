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
from .jobstore import JobWriter, cancel_requested, clear_cancel_flag, read_job, write_job
from ..db.repositories.cut_clips import record_cut
from ..db.repositories.markers import bulk_insert_auto_markers
from ..db.repositories.videos import set_status
from ..db.session import get_session

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


def _run_detect_markers(job: dict, writer: JobWriter) -> None:
    """v2's upload -> auto-detect -> unconfirmed markers path
    (docs/REQUIREMENTS_V2.md #3) - detect only, no cutting, then write the
    results straight to Postgres instead of a ground_truth JSON file being
    the end product (run_detect_markers_job still writes that file too, for
    parity with the CLI tools, but this job's actual output is the
    `markers` rows). On any failure, the video's status flips to 'error'
    with the message surfaced - this except branch runs in addition to,
    not instead of, _run_one's own job-level error handling."""
    video_path = Path(job["video"])
    video_id = job["video_id"]
    try:
        with _local_video(video_path) as local_video:
            result = pipeline.run_detect_markers_job(
                local_video, Path(job["config_path"]), Path(job["ground_truth_path"]),
                job, writer, fps=job.get("detect_fps"),
            )
        with get_session() as session:
            bulk_insert_auto_markers(session, video_id, result["makes_sec"])
            set_status(session, video_id, "ready", duration_s=result.get("duration_s"))
        job["n_markers"] = len(result["makes_sec"])
        job["duration_s"] = result.get("duration_s")
        job["message"] = f"done: {len(result['makes_sec'])} suggested marker(s) ready to review"
    except pipeline.JobCancelled:
        with get_session() as session:
            set_status(session, video_id, "uploaded")  # back to square one, not stuck "detecting"
        raise
    except Exception as e:
        with get_session() as session:
            set_status(session, video_id, "error", detect_error=str(e))
        raise


def _cluster_markers(markers: list[dict], merge_gap: float) -> list[list[dict]]:
    """Same algorithm as clip_shots.cluster_timestamps, but grouping marker
    dicts (not bare floats) so each output clip's constituent marker ids
    stay attached - clip_shots.cluster_timestamps alone would lose that
    association. merge_gap = pre + post: two markers' padded windows
    [t-pre, t+post] overlap exactly when their raw gap is under that."""
    ordered = sorted(markers, key=lambda m: m["timestamp_s"])
    clusters: list[list[dict]] = []
    for m in ordered:
        if clusters and (m["timestamp_s"] - clusters[-1][-1]["timestamp_s"]) < merge_gap:
            clusters[-1].append(m)
        else:
            clusters.append([m])
    return clusters


def _run_cut_markers(job: dict, writer: JobWriter) -> None:
    """v2's explicit, on-demand cut/export action (docs/REQUIREMENTS_V2.md
    #6) - one job spans every video the export query matched (same
    multi-video-per-job shape as _run_batch above), cutting each video's
    matching markers with clip_shots.cut_clip and recording the result as a
    cut_clips row (+ cut_clip_markers provenance) rather than a pre-cut
    dataset file. Cuts land under the source video's own directory
    regardless of who triggered the export - see api/export.py."""
    from .. import clip_shots

    pre, post = job["pre"], job["post"]
    merge_gap = pre + post
    created_by_user_id = job["created_by_user_id"]
    cut_clip_ids: list[int] = []

    for vi, video_job in enumerate(job["videos"], start=1):
        video_path = Path(video_job["video_path"])
        clusters = _cluster_markers(video_job["markers"], merge_gap)
        job["message"] = f"[{vi}/{len(job['videos'])}] cutting {len(clusters)} clip(s) from {video_path.name}"
        writer.save(force=True)

        with _local_video(video_path) as local_video:
            for i, cluster in enumerate(clusters, start=1):
                if cancel_requested(job["id"]):
                    raise pipeline.JobCancelled()
                start = cluster[0]["timestamp_s"] - pre
                end = cluster[-1]["timestamp_s"] + post
                out_dir = Path(video_job["out_dir"])
                # job["id"] (not video_id) makes this unique across separate
                # export runs on the same video - two exports both matching
                # marker overlaps would otherwise both compute i=1 and
                # silently overwrite each other's clip file on disk despite
                # having distinct cut_clips rows.
                out_path = out_dir / f"clip_{job['id']}_{i:03d}.mp4"
                clip_shots.cut_clip(local_video, start, end - start, out_path)
                with get_session() as session:
                    cut = record_cut(
                        session, video_job["video_id"], video_job["gcs_relpath_prefix"] + out_path.name,
                        max(0.0, start), end, created_by_user_id,
                        marker_ids=[m["id"] for m in cluster],
                    )
                    cut_clip_ids.append(cut.id)
                job["message"] = f"[{vi}/{len(job['videos'])}] cut {i}/{len(clusters)} clip(s) from {video_path.name}"
                writer.save()

    job["cut_clip_ids"] = cut_clip_ids
    job["message"] = f"done: {len(cut_clip_ids)} clip(s) cut across {len(job['videos'])} video(s)"


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
    # read_job() (not a raw read) for the same reason list_jobs() needs it:
    # gcsfuse can surface a stale-file-handle error to a reader racing a
    # concurrent writer's rename, and this file was written moments ago by
    # whoever queued the job - read_job() retries rather than treating that
    # as fatal.
    job = read_job(job_file)
    if job is None:
        raise OSError(f"could not read queued job file: {job_file}")
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
        elif job.get("kind") == "detect_markers":
            _run_detect_markers(job, writer)
        elif job.get("kind") == "cut_markers":
            _run_cut_markers(job, writer)
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
