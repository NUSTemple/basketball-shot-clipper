"""Local web UI for labeling shot-detection candidate clips as goal / no_goal,
and rating goal clips 1-5 stars (how good/highlight-worthy the make is).

Reads clip files from --clips-dir (default: $SHOT_CLIPPER_CLIPS_DIR, or the
folder where clip_shots.py writes candidate clips, one subfolder per source
video, shot_NNN.mp4 each) and reads/writes data/dataset/labels.json in this
repo as each clip is labeled. That labels.json file, together with the clips
it points at, is the goal/no_goal dataset - and, via stars, a ranked
shortlist of your best highlights (see shot-clipper-build-dataset --min-stars).

Usage:
    shot-clipper-label-ui [--clips-dir PATH] [--port 5050]
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import jobs
from ..clip_shots import SCORES_FILENAME
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
        scores_path = video_dir / SCORES_FILENAME
        scores = json.loads(scores_path.read_text()) if scores_path.is_file() else {}
        for clip_path in sorted(video_dir.glob("*.mp4")):
            rel = f"{video_dir.name}/{clip_path.name}"
            clips.append({"path": rel, "video": video_dir.name, "shot": clip_path.stem,
                          "filter_score": scores.get(clip_path.name)})
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
        c["stars"] = entry.get("stars") if entry else None
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
        # a star rating only means something for a goal clip - dropping to
        # no_goal (or re-labeling) clears any previous rating
        stars = labels.get(clip, {}).get("stars") if label == "goal" else None
        labels[clip] = {"label": label, "labeled_at": datetime.now(timezone.utc).isoformat(),
                         "stars": stars}
    save_labels(labels)
    return jsonify({"ok": True})


@app.post("/api/star")
def api_star():
    """Rate a goal clip 1-5 stars (how good/highlight-worthy it is), or
    clear the rating with stars: null. Only valid on a clip already labeled
    goal - see shot-clipper-build-dataset --min-stars for using ratings to
    pick your best clips."""
    body = request.get_json(force=True)
    clip = body.get("clip")
    stars = body.get("stars")
    if not clip:
        abort(400, "missing clip")
    if stars is not None and (not isinstance(stars, int) or not (1 <= stars <= 5)):
        abort(400, "stars must be an integer 1-5, or null to clear")

    labels = load_labels()
    entry = labels.get(clip)
    if entry is None or entry["label"] != "goal":
        abort(400, "clip must be labeled goal before it can be rated")

    entry["stars"] = stars
    entry["rated_at"] = datetime.now(timezone.utc).isoformat()
    save_labels(labels)
    return jsonify({"ok": True})


@app.post("/api/clips-dir")
def api_set_clips_dir():
    """Change which folder the app browses/labels and cuts new clips into.
    Creates the folder if it doesn't exist yet (e.g. starting a fresh
    project) - browsing just shows "no clips found" until something's cut
    there."""
    body = request.get_json(force=True)
    clips_dir_str = body.get("clips_dir")
    if not clips_dir_str:
        abort(400, "missing clips_dir")
    clips_dir = Path(clips_dir_str).expanduser().resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)
    app.config["CLIPS_DIR"] = clips_dir
    return jsonify({"ok": True, "clips_dir": str(clips_dir)})


def _import_ml_or_raise():
    try:
        from .. import clip_shots, detect_shots
        return clip_shots, detect_shots
    except ImportError as e:
        raise RuntimeError(
            "detection pipeline needs the `ml` extras: run `poetry install --with ml`"
        ) from e


def _process_one_video(video_path: Path, job: dict, use_filter: bool, prefix: str = "") -> dict:
    """Detect, cut, and (optionally) filter one video, updating job["message"]
    as it goes (prefix distinguishes it in a multi-video batch). Returns a
    result summary dict; also used directly by api_process_video."""
    clip_shots, detect_shots = _import_ml_or_raise()

    config_path = CONFIGS_DIR / f"{video_path.stem}.json"
    clips_dir = app.config["CLIPS_DIR"]
    out_subdir = clips_dir / video_path.stem
    output_path = GROUND_TRUTH_DIR / f"{video_path.stem}_detected.json"

    def on_progress(t):
        job["message"] = f"{prefix}scanning video: {t:.1f}s processed"

    job["message"] = f"{prefix}running ball detection (this can take a few minutes)..."
    makes = detect_shots.run_detection(video_path, config_path, output_path, progress_cb=on_progress)
    job["message"] = f"{prefix}found {len(makes)} candidate makes, cutting clips..."
    cut_results = clip_shots.cut_all(video_path, makes, out_subdir)

    result = {"video": video_path.name, "n_makes": len(makes), "clips_dir": str(out_subdir)}
    if use_filter:
        job["message"] = f"{prefix}scoring candidates with the trained filter..."
        hoop_bbox_norm = detect_shots.load_config(config_path)
        kept, dropped = clip_shots.filter_clips(
            cut_results, hoop_bbox_norm, FILTER_MODEL_PATH, filter_meta_path=FILTER_META_PATH)
        result["n_kept"] = len(kept)
        result["n_dropped"] = len(dropped)
    else:
        result["n_kept"] = len(makes)
        result["n_dropped"] = 0
    return result


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

    out_subdir = app.config["CLIPS_DIR"] / video_path.stem
    use_filter = body.get("use_filter", True) and FILTER_MODEL_PATH.is_file()

    def work(job):
        result = _process_one_video(video_path, job, use_filter)
        job["n_makes"] = result["n_makes"]
        job["clips_dir"] = result["clips_dir"]
        job["used_filter"] = use_filter
        if use_filter:
            job["n_kept"] = result["n_kept"]
            job["n_dropped"] = result["n_dropped"]
            job["message"] = (f"done: {result['n_kept']} candidate clips ready to label "
                               f"({result['n_dropped']} filtered out)")
        else:
            job["message"] = f"done: {result['n_makes']} candidate clips ready to label"

    try:
        job_id = jobs.start_job(
            {"video": str(video_path), "clips_video_dir": str(out_subdir)}, work)
    except RuntimeError as e:
        abort(409, str(e))
    return jsonify({"job_id": job_id})


@app.post("/api/process-batch")
def api_process_batch():
    """Kick off detect+clip for every video in a folder, one at a time in
    the background (sequential, not parallel - YOLO inference is heavy
    enough on a personal machine that running several at once would just
    contend with itself). Each video's clips show up in /api/clips as soon
    as that video finishes - you don't have to wait for the whole batch to
    start reviewing. Videos without an existing hoop calibration are
    reported back as skipped, not queued."""
    body = request.get_json(force=True)
    folder_str = body.get("folder")
    if not folder_str:
        abort(400, "missing folder")
    folder = Path(folder_str).expanduser()
    if not folder.is_dir():
        abort(400, f"not a folder: {folder}")

    video_exts = {".mp4", ".mov"}
    all_videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in video_exts)
    if not all_videos:
        abort(400, f"no video files found in {folder}")

    queue, skipped_uncalibrated = [], []
    for video_path in all_videos:
        if (CONFIGS_DIR / f"{video_path.stem}.json").is_file():
            queue.append(video_path)
        else:
            skipped_uncalibrated.append(video_path.name)
    if not queue:
        abort(400, f"none of the {len(all_videos)} video(s) in {folder} have a hoop calibration yet - "
                    f"run shot-clipper-calibrate on at least one first")

    use_filter = body.get("use_filter", True) and FILTER_MODEL_PATH.is_file()

    def work(job):
        job["total_videos"] = len(queue)
        job["completed_videos"] = []
        job["skipped_uncalibrated"] = skipped_uncalibrated

        for i, video_path in enumerate(queue, start=1):
            job["current_video_index"] = i
            job["current_video"] = video_path.name
            prefix = f"[{i}/{len(queue)}] {video_path.name}: "
            result = _process_one_video(video_path, job, use_filter, prefix=prefix)
            job["completed_videos"].append(result)

        job["message"] = (f"done: {len(queue)} video(s) processed"
                           + (f", {len(skipped_uncalibrated)} skipped (no calibration)"
                              if skipped_uncalibrated else ""))

    try:
        job_id = jobs.start_job({"folder": str(folder), "total_videos": len(queue)}, work)
    except RuntimeError as e:
        abort(409, str(e))
    return jsonify({"job_id": job_id, "queued": [v.name for v in queue],
                     "skipped_uncalibrated": skipped_uncalibrated})


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
