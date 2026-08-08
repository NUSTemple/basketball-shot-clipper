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
import hashlib
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
from ..clip_shots import SCORES_FILENAME, load_timestamps
from ..contact_sheet import extract_thumbnail
from ..dataset_labels import VALID_LABELS, labels_path, load_labels, save_labels
from ..roster import add_player, load_roster, scorer_slug

DEFAULT_CLIPS_DIR = Path(os.environ.get(
    "SHOT_CLIPPER_CLIPS_DIR",
    "/Users/pengtan/Videos/20260725 Basketball Video/clips",
))
THUMBNAIL_CACHE_DIR = Path("data/thumbnails_cache")
# where the in-app folder browser (/api/browse-dir) starts and stays confined
# to - both source videos and clip output normally live somewhere under here,
# so there's no reason the picker should ever wander into unrelated system
# folders (Docker's /root, /etc, /usr, ... or a native machine's full home
# directory clutter)
MEDIA_ROOT = Path(os.environ.get("SHOT_CLIPPER_MEDIA_ROOT", "/Users/pengtan/Videos"))

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
        c["scorer"] = entry.get("scorer") if entry else None
        c["assist"] = entry.get("assist") if entry else None
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


def _tag_goal_clip(field: str, clip: str, value):
    """Shared validation/write for /api/scorer and /api/assist: both are a
    single string field on a goal-labeled clip, cleared with null."""
    if value is not None and not isinstance(value, str):
        abort(400, f"{field} must be a string, or null to clear")

    labels = load_labels()
    entry = labels.get(clip)
    if entry is None or entry["label"] != "goal":
        abort(400, f"clip must be labeled goal before it can be tagged with {field}")

    entry[field] = value.strip() or None if value else None
    save_labels(labels)


@app.post("/api/scorer")
def api_scorer():
    """Tag who scored a goal clip (or clear with scorer: null). Only valid
    on a clip already labeled goal - same pattern as /api/star. This is the
    manual source of truth an automated jersey/face suggester could
    eventually feed into (see docs/PLAYER_IDENTIFICATION.md), not built yet."""
    body = request.get_json(force=True)
    clip = body.get("clip")
    if not clip:
        abort(400, "missing clip")
    _tag_goal_clip("scorer", clip, body.get("scorer"))
    return jsonify({"ok": True})


@app.post("/api/assist")
def api_assist():
    """Tag who assisted a goal clip (or clear with assist: null) - same
    pattern as /api/scorer, sharing the same roster of player names."""
    body = request.get_json(force=True)
    clip = body.get("clip")
    if not clip:
        abort(400, "missing clip")
    _tag_goal_clip("assist", clip, body.get("assist"))
    return jsonify({"ok": True})


@app.get("/api/roster")
def api_roster():
    return jsonify({"players": load_roster()})


@app.post("/api/roster")
def api_add_player():
    body = request.get_json(force=True)
    name = body.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        abort(400, "missing player name")
    return jsonify({"players": add_player(name)})


@app.post("/api/export-clips")
def api_export_clips():
    """Export a hand-picked list of clips (e.g. selected in the Library
    grid) into an arbitrary destination folder for further processing -
    same flat naming as shot-clipper-build-dataset (<video>__shot_NNN[_Nstar].mp4,
    so files from different source videos don't collide) but for a specific
    selection rather than the whole labeled dataset. Symlinks by default
    (matches build_dataset.py); pass "copy": true to copy real files
    instead (needed if the destination will be used somewhere the clips
    folder isn't reachable, e.g. an external drive or a different machine)."""
    body = request.get_json(force=True)
    clip_paths = body.get("clips")
    dest_str = body.get("dest")
    copy = bool(body.get("copy"))
    if not clip_paths or not isinstance(clip_paths, list):
        abort(400, "missing clips")
    if not dest_str:
        abort(400, "missing dest")

    clips_dir = app.config["CLIPS_DIR"]
    dest = resolve_user_path(dest_str).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    labels = load_labels()
    exported, missing = [], []
    for clip_rel in clip_paths:
        src = resolve_within(clips_dir, clip_rel)
        if not src.is_file():
            missing.append(clip_rel)
            continue
        entry = labels.get(clip_rel, {})
        is_goal = entry.get("label") == "goal"
        stars = entry.get("stars") if is_goal else None
        scorer = entry.get("scorer") if is_goal else None
        base = Path(clip_rel.replace("/", "__")).stem
        suffix = f"_{stars}star" if stars else ""
        suffix += f"_{scorer_slug(scorer)}" if scorer else ""
        out_path = dest / f"{base}{suffix}.mp4"
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()
        if copy:
            shutil.copy2(src, out_path)
        else:
            out_path.symlink_to(src.resolve())
        exported.append(out_path.name)

    return jsonify({"ok": True, "exported": len(exported), "missing": missing, "dest": str(dest)})


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


@app.get("/api/browse-dir")
def api_browse_dir():
    """List one directory's contents server-side, for the in-app folder
    browser modal - a fallback for wherever the native macOS picker
    (/api/pick-video, /api/pick-folder) can't work, since osascript only
    exists on macOS and can never run inside the Docker container. Doesn't
    expose anything a user couldn't already reach by typing an arbitrary
    path into the clips-folder/video-path fields directly - this just makes
    finding that path interactive instead of requiring you to know it
    upfront. kind="video" also lists .mp4/.mov files (to browse into and
    pick one); kind="folder" (default) only lists subdirectories."""
    default_start = str(MEDIA_ROOT) if MEDIA_ROOT.is_dir() else str(Path.home())
    path_str = request.args.get("path") or default_start
    kind = request.args.get("kind", "folder")
    current = resolve_user_path(path_str)
    if current.is_file():
        current = current.parent
    if not current.is_dir():
        abort(400, f"not a folder: {current}")
    current = current.resolve()

    root = MEDIA_ROOT.resolve() if MEDIA_ROOT.is_dir() else None
    if root and not current.is_relative_to(root):
        current = root  # never wander outside the configured media root

    video_exts = {".mp4", ".mov"}
    dirs, files = [], []
    try:
        children = sorted(current.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        abort(403, f"permission denied: {current}")
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir():
                dirs.append(child.name)
            elif kind == "video" and child.suffix.lower() in video_exts:
                files.append(child.name)
        except OSError:
            continue

    parent = None
    if current.parent != current and (root is None or current != root):
        parent = str(current.parent)
    return jsonify({"path": str(current), "parent": parent, "dirs": dirs, "files": files,
                     "root": str(root) if root else None})


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


@app.get("/api/calibrate-frame")
def api_calibrate_frame():
    """Extract one representative frame from a video (a still, via ffmpeg -
    no `ml` extras needed, unlike shot-clipper-calibrate's OpenCV window, so
    this works in Docker too) for the in-browser hoop-calibration UI to draw
    a box on. Cached like thumbnails, keyed by video + timestamp."""
    video_path_str = request.args.get("video")
    if not video_path_str:
        abort(400, "missing video")
    video_path = resolve_user_path(video_path_str)
    if not video_path.is_file():
        abort(400, f"video not found: {video_path}")

    from .pipeline import probe_video
    meta = probe_video(video_path)
    duration = meta.get("duration_s")
    t = request.args.get("t", type=float)
    if t is None:
        t = duration / 2 if duration else 1.0

    digest = hashlib.sha1(f"{video_path.resolve()}::{t:.2f}".encode()).hexdigest()
    out_path = (THUMBNAIL_CACHE_DIR / f"calib_{digest}.jpg").resolve()
    if not out_path.is_file():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-ss", f"{max(0.0, t):.2f}", "-i", str(video_path),
               "-frames:v", "1", "-q:v", "2", str(out_path)]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            abort(500, "could not extract a frame from this video")
    return send_from_directory(out_path.parent, out_path.name, conditional=True)


@app.post("/api/save-calibration")
def api_save_calibration():
    """Save a hoop calibration drawn in the browser - the same
    data/configs/<video>.json shot-clipper-calibrate writes, minus the
    frame_index field (meaningless here since the frame came from a
    timestamp, not a frame count; detection only ever reads hoop_bbox_norm)."""
    body = request.get_json(force=True)
    video_path_str = body.get("video")
    bbox = body.get("hoop_bbox_norm")
    frame_width = body.get("frame_width")
    frame_height = body.get("frame_height")
    if not video_path_str:
        abort(400, "missing video")
    if not (isinstance(bbox, list) and len(bbox) == 4):
        abort(400, "missing or invalid hoop_bbox_norm")
    video_path = resolve_user_path(video_path_str)
    if not video_path.is_file():
        abort(400, f"video not found: {video_path}")

    config_path = CONFIGS_DIR / f"{video_path.stem}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "video": video_path.name,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "hoop_bbox_norm": bbox,
    }
    config_path.write_text(json.dumps(cfg, indent=2))
    return jsonify({"ok": True, "config_path": str(config_path)})


def _valid_detect_fps(body: dict) -> float | None:
    """Optional detection-speed override from the Detect tab's speed picker
    - None means "use run_detection's own default" (15fps, the one that's
    actually been validated; see pipeline.process_one_video)."""
    fps = body.get("fps")
    if fps is None:
        return None
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        abort(400, "fps must be a number")
    if not (1 <= fps <= 30):
        abort(400, "fps must be between 1 and 30")
    return fps


@app.get("/api/existing-detection")
def api_existing_detection():
    """Check if calibration and detection results exist for a video."""
    video_path_str = request.args.get("video")
    if not video_path_str:
        abort(400, "missing video")
    video_path = resolve_user_path(video_path_str)

    # Check calibration
    config_path = CONFIGS_DIR / f"{video_path.stem}.json"
    calibration_exists = config_path.is_file()

    # Check existing detection
    ground_truth_path = GROUND_TRUTH_DIR / f"{video_path.stem}_detected.json"
    detection_exists = False
    n_makes = 0
    detected_at = None

    if ground_truth_path.is_file():
        try:
            makes = load_timestamps(ground_truth_path)
            detection_exists = True
            n_makes = len(makes)
            detected_at = ground_truth_path.stat().st_mtime
        except (json.JSONDecodeError, KeyError, OSError):
            detection_exists = False

    return jsonify({
        "calibration_exists": calibration_exists,
        "exists": detection_exists,
        "n_makes": n_makes,
        "detected_at": detected_at,
    })


@app.post("/api/process-video")
def api_process_video():
    """Kick off detect+clip for a full source video in the background, so its
    candidate clips show up in /api/clips once done. Requires the `ml` extras
    (ultralytics/opencv/torch) to be installed, and a hoop calibration
    already saved for this video (see shot-clipper-calibrate, or calibrate
    it right here in Detect)."""
    body = request.get_json(force=True)
    video_path_str = body.get("video_path")
    if not video_path_str:
        abort(400, "missing video_path")
    video_path = resolve_user_path(video_path_str)
    if not video_path.is_file():
        abort(400, f"video not found: {video_path}")

    config_path = CONFIGS_DIR / f"{video_path.stem}.json"
    if not config_path.is_file():
        abort(400, "no hoop calibration found for this video yet - click "
                    "\"Calibrate hoop\" below to draw one, then try again")

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
        "detect_fps": _valid_detect_fps(body),
        "reuse_detection": bool(body.get("reuse_detection")),
    }
    job_id = jobs.start_job(spec)
    queue_pos = jobs.get_queue_position(job_id)
    return jsonify({"job_id": job_id, "queue_position": queue_pos})


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
        "detect_fps": _valid_detect_fps(body),
    }
    job_id = jobs.start_job(spec)
    queue_pos = jobs.get_queue_position(job_id)
    return jsonify({"job_id": job_id, "queue_position": queue_pos,
                     "queued": [v.name for v in queue],
                     "skipped_uncalibrated": skipped_uncalibrated})


@app.get("/api/jobs")
def api_list_jobs():
    return jsonify({"jobs": jobs.list_jobs()})


@app.get("/api/process-video/<job_id>")
def api_process_video_status(job_id):
    job = jobs.get_job(job_id)
    if job is None:
        abort(404)
    return jsonify(job)


@app.post("/api/process-video/<job_id>/cancel")
def api_cancel_job(job_id):
    if not jobs.cancel_job(job_id):
        abort(409, "job isn't running (already finished, or its process is gone)")
    return jsonify({"ok": True})


@app.get("/video/<path:relpath>")
def serve_video(relpath):
    clips_dir = app.config["CLIPS_DIR"]
    full = resolve_within(clips_dir, relpath)
    if not full.is_file():
        abort(404)
    return send_from_directory(full.parent, full.name, conditional=True)


def _thumbnail_for(clip_path: Path) -> Path | None:
    """Cached thumbnail for one clip, generated on first request. Cached by
    a hash of the clip's absolute path rather than alongside the clip itself
    (like the CLI shot-clipper-contact-sheet does) - the clips folder can be
    a read-only mount (Docker), and different clips folders can share the
    same relative path (video/shot_NNN.mp4), so the cache needs its own
    identity. Returns None if ffmpeg couldn't grab a frame at all."""
    digest = hashlib.sha1(str(clip_path.resolve()).encode()).hexdigest()
    # absolute: send_from_directory resolves a relative directory against
    # Flask's root_path (the package dir), not the process cwd, so a
    # relative path here would silently 404 even after being written fine
    out_path = (THUMBNAIL_CACHE_DIR / f"{digest}.jpg").resolve()
    if out_path.is_file() and out_path.stat().st_mtime >= clip_path.stat().st_mtime:
        return out_path
    # 5.0s matches the default [t-5s, t+2s] clip cut (see clip_shots.py) -
    # that's where the shot/make moment lands; short/custom-cut clips fall
    # back to a frame near the start rather than showing nothing.
    if extract_thumbnail(clip_path, out_path, at=5.0) or extract_thumbnail(clip_path, out_path, at=0.3):
        return out_path
    return None


@app.get("/thumbnail/<path:relpath>")
def serve_thumbnail(relpath):
    clips_dir = app.config["CLIPS_DIR"]
    full = resolve_within(clips_dir, relpath)
    if not full.is_file():
        abort(404)
    thumb = _thumbnail_for(full)
    if thumb is None:
        abort(404, "could not generate a thumbnail for this clip")
    return send_from_directory(thumb.parent, thumb.name, conditional=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()
    app.config["CLIPS_DIR"] = args.clips_dir.resolve()
    print(f"labeling clips from: {app.config['CLIPS_DIR']}")
    print(f"labels saved to:     {labels_path()}")
    resumed = jobs.kick_queue()
    if resumed:
        print(f"resuming queued job: {resumed}")
    # threaded=True: /api/pick-video and /api/pick-folder block their request
    # thread on a native OS dialog until the user responds - without this,
    # that would freeze every other request (clip loading, labeling, job
    # polling) for as long as the dialog is open.
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
