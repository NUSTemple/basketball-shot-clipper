"""Local web UI for labeling shot-detection candidate clips as goal / no_goal.

Reads clip files from --clips-dir (default: $SHOT_CLIPPER_CLIPS_DIR, or the
folder where clip_shots.py writes candidate clips, one subfolder per source
video, shot_NNN.mp4 each) and reads/writes data/dataset/labels.json in this
repo as each clip is labeled. That labels.json file, together with the clips
it points at, is the goal/no_goal dataset.

Usage:
    shot-clipper-label-ui [--clips-dir PATH] [--port 5050]
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import jobs
from ..dataset_labels import VALID_LABELS, labels_path, load_labels, save_labels

DEFAULT_CLIPS_DIR = Path(os.environ.get(
    "SHOT_CLIPPER_CLIPS_DIR",
    "/Users/pengtan/Videos/20260725 Basketball Video/clips",
))
CONFIGS_DIR = Path("data/configs")
GROUND_TRUTH_DIR = Path("data/ground_truth")
FILTER_MODEL_PATH = Path("models/shot_filter.joblib")
FILTER_META_PATH = Path("models/shot_filter_meta.json")

app = Flask(__name__)
app.config["CLIPS_DIR"] = DEFAULT_CLIPS_DIR


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code


def list_clips(clips_dir: Path):
    clips = []
    for video_dir in sorted(p for p in clips_dir.iterdir() if p.is_dir()):
        for clip_path in sorted(video_dir.glob("*.mp4")):
            rel = f"{video_dir.name}/{clip_path.name}"
            clips.append({"path": rel, "video": video_dir.name, "shot": clip_path.stem})
    return clips


def resolve_within(clips_dir: Path, relpath: str) -> Path:
    full = (clips_dir / relpath).resolve()
    clips_dir_resolved = clips_dir.resolve()
    if not full.is_relative_to(clips_dir_resolved):
        abort(400, "invalid clip path")
    return full


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/clips")
def api_clips():
    clips_dir = app.config["CLIPS_DIR"]
    if not clips_dir.is_dir():
        return jsonify({"error": f"clips dir not found: {clips_dir}"}), 404
    labels = load_labels()
    clips = list_clips(clips_dir)
    for c in clips:
        entry = labels.get(c["path"])
        c["label"] = entry["label"] if entry else None
    return jsonify({"clips_dir": str(clips_dir), "clips": clips})


@app.post("/api/label")
def api_label():
    body = request.get_json(force=True)
    clip = body.get("clip")
    label = body.get("label")
    if not clip:
        abort(400, "missing clip")
    if label is not None and label not in VALID_LABELS:
        abort(400, f"label must be one of {sorted(VALID_LABELS)} or null")

    clips_dir = app.config["CLIPS_DIR"]
    full = resolve_within(clips_dir, clip)
    if not full.is_file():
        abort(404, "clip not found")

    labels = load_labels()
    if label is None:
        labels.pop(clip, None)
    else:
        labels[clip] = {"label": label, "labeled_at": datetime.now(timezone.utc).isoformat()}
    save_labels(labels)
    return jsonify({"ok": True})


@app.post("/api/process-video")
def api_process_video():
    """Kick off detect+clip for a full source video in the background, so its
    candidate clips show up in /api/clips once done. Requires the `ml` extras
    (ultralytics/opencv/torch) to be installed, and a hoop calibration
    already saved for this video (see shot-clipper-calibrate)."""
    body = request.get_json(force=True)
    video_path_str = body.get("video_path")
    if not video_path_str:
        abort(400, "missing video_path")
    video_path = Path(video_path_str).expanduser()
    if not video_path.is_file():
        abort(400, f"video not found: {video_path}")

    config_path = CONFIGS_DIR / f"{video_path.stem}.json"
    if not config_path.is_file():
        abort(400, f"no hoop calibration found for this video at {config_path} - "
                    f"run `poetry run shot-clipper-calibrate \"{video_path}\"` first")

    clips_dir = app.config["CLIPS_DIR"]
    out_subdir = clips_dir / video_path.stem
    output_path = GROUND_TRUTH_DIR / f"{video_path.stem}_detected.json"
    use_filter = body.get("use_filter", True) and FILTER_MODEL_PATH.is_file()

    def work(job):
        try:
            from .. import clip_shots, detect_shots
        except ImportError as e:
            raise RuntimeError(
                "detection pipeline needs the `ml` extras: "
                "run `poetry install --with ml`"
            ) from e

        def on_progress(t):
            job["message"] = f"scanning video: {t:.1f}s processed"

        job["message"] = "running ball detection (this can take a few minutes)..."
        makes = detect_shots.run_detection(
            video_path, config_path, output_path, progress_cb=on_progress,
            filter_model_path=FILTER_MODEL_PATH if use_filter else None,
            filter_meta_path=FILTER_META_PATH if use_filter else None)
        job["message"] = f"found {len(makes)} candidate makes, cutting clips..."
        clip_shots.cut_all(video_path, makes, out_subdir)
        job["n_makes"] = len(makes)
        job["clips_dir"] = str(out_subdir)
        job["used_filter"] = use_filter
        job["message"] = (f"done: {len(makes)} candidate clips ready to label"
                           + (" (trained filter applied)" if use_filter else ""))

    try:
        job_id = jobs.start_job(
            {"video": str(video_path), "clips_video_dir": str(out_subdir)}, work)
    except RuntimeError as e:
        abort(409, str(e))
    return jsonify({"job_id": job_id})


@app.get("/api/process-video/<job_id>")
def api_process_video_status(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        abort(404)
    return jsonify(job)


@app.get("/video/<path:relpath>")
def serve_video(relpath):
    clips_dir = app.config["CLIPS_DIR"]
    full = resolve_within(clips_dir, relpath)
    if not full.is_file():
        abort(404)
    return send_from_directory(full.parent, full.name, conditional=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    app.config["CLIPS_DIR"] = args.clips_dir.resolve()
    print(f"labeling clips from: {app.config['CLIPS_DIR']}")
    print(f"labels saved to:     {labels_path()}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
