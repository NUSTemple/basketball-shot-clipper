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
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException

from . import jobs
from .pipeline import CONFIGS_DIR, FILTER_MODEL_PATH, GROUND_TRUTH_DIR
from ..clip_shots import SCORES_FILENAME
from ..dataset_labels import VALID_LABELS, labels_path, load_labels, save_labels

DEFAULT_CLIPS_DIR = Path(os.environ.get(
    "SHOT_CLIPPER_CLIPS_DIR",
    "/Users/pengtan/Videos/20260725 Basketball Video/clips",
))

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


def resolve_user_path(path_str: str) -> Path:
    """Users sometimes paste a file:// URL (e.g. dragged from a Finder
    window into a browser, or copied from an address bar) instead of a
    plain filesystem path - Path("file:///Users/...%20...") doesn't exist
    on disk even though the file does, which just looks like a confusing
    "not found" error. Try the literal input first (a real path can
    legitimately contain "%20" or start with "file"), and only fall back to
    stripping the file:// scheme and percent-decoding if that path doesn't
    actually exist.
    """
    literal = Path(path_str).expanduser()
    if literal.exists():
        return literal
    normalized = path_str
    if normalized.startswith("file://"):
        normalized = urlparse(normalized).path
    normalized = unquote(normalized)
    return Path(normalized).expanduser()


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


def _osascript_choose(kind: str, prompt: str) -> dict:
    """Run a native macOS "choose file"/"choose folder" dialog and return the
    selected path - lets the UI offer a real file picker instead of a text
    field to paste a path into (which is how a copied file:// URL or a typo
    creates a confusing "not found" error). Blocks this request's thread
    until the user responds; app.run(threaded=True) keeps the rest of the
    app responsive meanwhile. Only available when running natively on macOS
    with osascript on PATH - the plain text inputs remain a fallback
    everywhere else (Docker, other OSes)."""
    if shutil.which("osascript") is None:
        abort(400, "native file picker needs macOS (osascript not found) - type/paste the path instead")
    verb = "choose file" if kind == "file" else "choose folder"
    type_clause = ' of type {"public.movie"}' if kind == "file" else ""
    safe_prompt = prompt.replace('"', "")
    script = f'POSIX path of ({verb} with prompt "{safe_prompt}"{type_clause})'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        if "User canceled" in result.stderr:
            return {"cancelled": True}
        abort(500, f"picker failed: {result.stderr.strip()}")
    return {"path": result.stdout.strip()}


@app.post("/api/pick-video")
def api_pick_video():
    return jsonify(_osascript_choose("file", "Select a video"))


@app.post("/api/pick-folder")
def api_pick_folder():
    body = request.get_json(silent=True) or {}
    return jsonify(_osascript_choose("folder", body.get("prompt", "Select a folder")))


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
    clips_dir = resolve_user_path(clips_dir_str).resolve()
    clips_dir.mkdir(parents=True, exist_ok=True)
    app.config["CLIPS_DIR"] = clips_dir
    return jsonify({"ok": True, "clips_dir": str(clips_dir)})


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
    video_path = resolve_user_path(video_path_str)
    if not video_path.is_file():
        abort(400, f"video not found: {video_path}")

    config_path = CONFIGS_DIR / f"{video_path.stem}.json"
    if not config_path.is_file():
        abort(400, f"no hoop calibration found for this video at {config_path} - "
                    f"run `poetry run shot-clipper-calibrate \"{video_path}\"` first")

    clips_dir_str = body.get("clips_dir")
    out_dir = resolve_user_path(clips_dir_str).resolve() if clips_dir_str else app.config["CLIPS_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_subdir = out_dir / video_path.stem
    use_filter = body.get("use_filter", True) and FILTER_MODEL_PATH.is_file()
    ground_truth_path = GROUND_TRUTH_DIR / f"{video_path.stem}_detected.json"

    spec = {
        "kind": "single",
        "video": str(video_path),
        "clips_video_dir": str(out_subdir),
        "config_path": str(config_path),
        "ground_truth_path": str(ground_truth_path),
        "out_dir": str(out_dir),
        "use_filter": use_filter,
    }
    try:
        job_id = jobs.start_job(spec)
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
    folder = resolve_user_path(folder_str)
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

    clips_dir_str = body.get("clips_dir")
    out_dir = resolve_user_path(clips_dir_str).resolve() if clips_dir_str else app.config["CLIPS_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    use_filter = body.get("use_filter", True) and FILTER_MODEL_PATH.is_file()

    spec = {
        "kind": "batch",
        "folder": str(folder),
        "queue": [str(v) for v in queue],
        "skipped_uncalibrated": skipped_uncalibrated,
        "out_dir": str(out_dir),
        "use_filter": use_filter,
        "total_videos": len(queue),
    }
    try:
        job_id = jobs.start_job(spec)
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
    # threaded=True: /api/pick-video and /api/pick-folder block their request
    # thread on a native OS dialog until the user responds - without this,
    # that would freeze every other request (clip loading, labeling, job
    # polling) for as long as the dialog is open.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
