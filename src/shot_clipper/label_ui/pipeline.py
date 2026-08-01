"""Detect+cut(+filter) pipeline for one source video, shared by the label
UI's job endpoints (app.py, via worker.py) and nothing else - the CLI tools
(detect_shots.py, clip_shots.py) have their own thin __main__ wrappers
around the same underlying functions.
"""
import json
import subprocess
from pathlib import Path

CONFIGS_DIR = Path("data/configs")
GROUND_TRUTH_DIR = Path("data/ground_truth")
FILTER_MODEL_PATH = Path("models/shot_filter.joblib")
FILTER_META_PATH = Path("models/shot_filter_meta.json")


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
                       out_dir: Path, use_filter: bool, job: dict, writer, prefix: str = "") -> dict:
    """Detect, cut, and (optionally) filter one video, updating job["message"]
    in place as it goes (prefix distinguishes it in a multi-video batch) and
    persisting via writer.save() so progress survives whatever's reading
    the job - see jobstore.JobWriter. Returns a result summary dict."""
    from .. import clip_shots, detect_shots

    out_subdir = out_dir / video_path.stem

    meta = probe_video(video_path)
    job["current_video_meta"] = meta
    writer.save(force=True)

    def on_progress(t):
        job["message"] = f"{prefix}scanning video: {t:.1f}s processed"
        writer.save()

    job["message"] = f"{prefix}running ball detection (this can take a few minutes)..."
    writer.save(force=True)
    makes = detect_shots.run_detection(video_path, config_path, ground_truth_path, progress_cb=on_progress)
    job["message"] = f"{prefix}found {len(makes)} candidate makes, cutting clips..."
    writer.save(force=True)
    cut_results = clip_shots.cut_all(video_path, makes, out_subdir)

    result = {"video": video_path.name, "n_makes": len(makes), "clips_dir": str(out_subdir), **meta}
    if use_filter:
        job["message"] = f"{prefix}scoring candidates with the trained filter..."
        writer.save(force=True)
        hoop_bbox_norm = detect_shots.load_config(config_path)
        kept, dropped = clip_shots.filter_clips(
            cut_results, hoop_bbox_norm, FILTER_MODEL_PATH, filter_meta_path=FILTER_META_PATH)
        result["n_kept"] = len(kept)
        result["n_dropped"] = len(dropped)
    else:
        result["n_kept"] = len(makes)
        result["n_dropped"] = 0
    return result
