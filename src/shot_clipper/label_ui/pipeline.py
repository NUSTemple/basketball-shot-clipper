"""Detect+cut(+filter) pipeline for one source video, shared by the label
UI's job endpoints (app.py, via worker.py) and nothing else - the CLI tools
(detect_shots.py, clip_shots.py) have their own thin __main__ wrappers
around the same underlying functions.
"""
import json
import os
import subprocess
import time
from pathlib import Path

from .. import inference, video_source
from ..device_config import get_device, device_summary
from .jobstore import cancel_requested

# Overridable so the hosted deployment can point these at the GCS-mounted
# volume instead of each container's own ephemeral filesystem - label-ui
# and worker are separate containers with separate local disks, so a
# calibration saved by one (at the plain "data/configs" default) would be
# completely invisible to the other, not just lost on a scale-to-zero
# restart. Both containers must be given the same value for this to work.
CONFIGS_DIR = Path(os.environ.get("SHOT_CLIPPER_CONFIGS_DIR", "data/configs"))
GROUND_TRUTH_DIR = Path(os.environ.get("SHOT_CLIPPER_GROUND_TRUTH_DIR", "data/ground_truth"))
FILTER_MODEL_PATH = Path("models/shot_filter.joblib")
FILTER_META_PATH = Path("models/shot_filter_meta.json")


class JobCancelled(Exception):
    """Raised from within process_one_video (via on_progress) when
    jobs.cancel_job() has set the cooperative stop flag for this job -
    caught specially by worker.py so a user-requested stop shows up as
    "cancelled", not "error"."""


def probe_video(video_path: Path) -> dict:
    """Read duration/resolution/fps via ffprobe (ffmpeg's companion CLI,
    already required by the whole app - see README Install) so Job Status
    can show what's being processed without needing the `ml` extras."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=width,height,avg_frame_rate",
        "-of", "json", str(video_path),
    ]
    meta = {}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(proc.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        data = {}

    duration = data.get("format", {}).get("duration")
    if duration is not None:
        meta["duration_s"] = round(float(duration), 1)

    stream = (data.get("streams") or [{}])[0]
    if stream.get("width") and stream.get("height"):
        meta["resolution"] = f"{stream['width']}x{stream['height']}"
    fps_raw = stream.get("avg_frame_rate")
    if fps_raw and fps_raw != "0/0":
        num, _, den = fps_raw.partition("/")
        if float(den or 1) != 0:
            meta["fps"] = round(float(num) / float(den or 1), 1)

    try:
        meta["size_mb"] = round(video_path.stat().st_size / (1024 * 1024), 1)
    except OSError:
        pass
    return meta


def process_one_video(video_path: Path, config_path: Path, ground_truth_path: Path,
                       out_dir: Path, use_filter: bool, job: dict, writer, prefix: str = "",
                       fps: float | None = None, reuse_detection: bool = False) -> dict:
    """Detect, cut, and (optionally) filter one video, updating job["message"]
    in place as it goes (prefix distinguishes it in a multi-video batch) and
    persisting via writer.save() so progress survives whatever's reading
    the job - see jobstore.JobWriter. Returns a result summary dict.

    fps controls detect_shots.run_detection's temporal sampling rate - lower
    means fewer frames scanned per second of video, so faster but a real
    recall risk (a make that only shows the ball in the hoop for a couple
    of frames can get sampled right past). The 15fps default is the one
    that's actually been validated (see README); anything lower is an
    explicit, user-chosen speed/recall tradeoff, not a new default.

    reuse_detection skips run_detection entirely and loads makes_sec
    straight from an existing ground_truth_path instead - for when you
    just want to re-cut/re-filter clips (new calibration crop doesn't
    matter here, only the timestamps do) without re-running the slow YOLO
    scan. Silently falls back to a real detection run if the file's
    missing, so callers don't need to re-check existence themselves."""
    from .. import clip_shots, detect_shots

    out_subdir = out_dir / video_path.stem

    device = get_device()
    device_info = device_summary(device)
    decoder_info = video_source.describe()
    # Which inference backend, reported for the same reason decoder_info is:
    # the ONNX path picks its own accelerator (DirectML/CoreML/CPU) through
    # onnxruntime, which device_summary knows nothing about - it asks torch,
    # and in a torch-less install would flatly answer "CPU" while the GPU was
    # doing the work. A backend that silently ran somewhere slow is invisible
    # in the UI otherwise; it just takes longer.
    backend_info = inference.resolve_backend()

    meta = probe_video(video_path)
    job["current_video_meta"] = meta
    job["device"] = device
    job["device_info"] = device_info
    job["decoder_info"] = decoder_info
    job["backend"] = backend_info
    writer.save(force=True)

    if reuse_detection and ground_truth_path.is_file():
        job["message"] = f"{prefix}reusing existing detection results..."
        job["scan_progress"] = None
        writer.save(force=True)
        makes = clip_shots.load_timestamps(ground_truth_path)
    else:
        duration = meta.get("duration_s")
        scan_start = None

        def on_progress(t):
            nonlocal scan_start
            if cancel_requested(job["id"]):
                raise JobCancelled()
            now = time.monotonic()
            if scan_start is None:
                scan_start = now
            elapsed = now - scan_start

            pct = round(min(100, t / duration * 100)) if duration else None
            # skip ETA on the first fraction of a second of video - the rate
            # estimate from a near-zero sample swings wildly and looks broken
            eta = None
            if duration and t > 1.0 and elapsed > 0:
                rate = t / elapsed  # video-seconds processed per wall-clock second
                if rate > 0:
                    eta = max(0, (duration - t) / rate)

            job["message"] = (f"{prefix}scanning video: {t:.1f}s processed"
                               + (f" ({pct}%)" if pct is not None else ""))
            job["scan_progress"] = {
                "seconds": round(t, 1), "duration_s": duration, "pct": pct,
                "elapsed_s": round(elapsed, 1),
                "eta_s": round(eta, 1) if eta is not None else None,
            }
            writer.save()

        job["message"] = (f"{prefix}running ball detection on {device_info} ({backend_info}), "
                           f"decoding via {decoder_info} (this can take a few minutes)...")
        job["scan_progress"] = None
        writer.save(force=True)
        detect_kwargs = {"progress_cb": on_progress}
        if fps:
            detect_kwargs["fps"] = fps
        makes = detect_shots.run_detection(video_path, config_path, ground_truth_path, **detect_kwargs)

    if cancel_requested(job["id"]):
        raise JobCancelled()

    def on_cut_progress(i, total):
        job["message"] = f"{prefix}cutting clips: {i}/{total}"
        writer.save()

    job["message"] = f"{prefix}found {len(makes)} candidate makes, cutting clips..."
    job["scan_progress"] = None
    writer.save(force=True)
    cut_results = clip_shots.cut_all(
        video_path, makes, out_subdir, progress_cb=on_cut_progress,
        cancel_check=lambda: cancel_requested(job["id"]))

    if cancel_requested(job["id"]):
        raise JobCancelled()

    result = {"video": video_path.name, "n_makes": len(makes), "clips_dir": str(out_subdir), **meta}
    if use_filter:
        def on_filter_progress(i, total):
            job["message"] = f"{prefix}scoring candidates with the trained filter: {i}/{total}"
            writer.save()

        job["message"] = f"{prefix}scoring candidates with the trained filter..."
        writer.save(force=True)
        hoop_bbox_norm = detect_shots.load_config(config_path)
        kept, dropped = clip_shots.filter_clips(
            cut_results, hoop_bbox_norm, FILTER_MODEL_PATH, filter_meta_path=FILTER_META_PATH,
            progress_cb=on_filter_progress, cancel_check=lambda: cancel_requested(job["id"]))
        result["n_kept"] = len(kept)
        result["n_dropped"] = len(dropped)

        if cancel_requested(job["id"]):
            raise JobCancelled()
    else:
        result["n_kept"] = len(makes)
        result["n_dropped"] = 0
    return result
