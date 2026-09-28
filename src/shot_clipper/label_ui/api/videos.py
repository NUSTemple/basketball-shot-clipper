"""v2 video registry - the metadata half of an uploaded video (bytes stay
on GCS, registered separately via the existing /api/uploads +
/api/uploads/complete flow, unchanged). Visibility is fully open
(docs/REQUIREMENTS_V2.md #2): GET routes list every video, not just the
caller's own.
"""
import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_from_directory

from ...db.repositories.calibration_profiles import get_profile
from ...db.repositories.games import get_game
from ...db.repositories.users import get_or_create_user
from ...db.repositories.videos import (
    attach_calibration_profile,
    create_video,
    get_by_gcs_relpath,
    get_video,
    list_videos,
    set_game,
)
from ...db.session import get_session
from .. import jobs
from ..auth import require_user_email

bp = Blueprint("api_v2_videos", __name__, url_prefix="/api/v2/videos")

# Where a basket calibration gets materialized into the JSON shape
# detect_shots.load_config() already expects, and where the (superseded by
# Postgres, but still written for CLI-tooling parity) ground-truth timestamp
# file lands - keyed by video id so no job-id-before-queuing chicken/egg
# problem, and namespaced away from the legacy per-user dataset paths this
# doesn't touch.
_V2_CONFIGS_DIR = Path("_v2_configs")
_V2_GROUND_TRUTH_DIR = Path("_v2_ground_truth")


def _serialize(video):
    return {
        "id": video.id,
        "owner_user_id": video.owner_user_id,
        "game_id": video.game_id,
        "gcs_relpath": video.gcs_relpath,
        "original_filename": video.original_filename,
        "calibration_profile_id": video.calibration_profile_id,
        "duration_s": float(video.duration_s) if video.duration_s is not None else None,
        "status": video.status,
        "detect_error": video.detect_error,
        "created_at": video.created_at.isoformat() if video.created_at else None,
        "detected_at": video.detected_at.isoformat() if video.detected_at else None,
    }


def _queue_detect(session, video, profile, data_root: Path) -> str:
    """Materialize the video's basket calibration into a transient config
    file matching detect_shots.load_config()'s existing schema, then queue
    a detect_markers job - zero changes needed to calibrate_hoop.py /
    detect_shots.py / pipeline.py's config-file contract. `profile` must
    already be validated non-None by the caller, before it touches the
    video row - a bad FK surfaces as an ugly 500 IntegrityError otherwise."""
    video_abs_path = data_root / video.gcs_relpath
    if not video_abs_path.is_file():
        abort(400, f"video file not found at {video_abs_path}")

    config_path = data_root / _V2_CONFIGS_DIR / f"video_{video.id}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        "video": video.original_filename,
        "frame_width": profile.frame_width,
        "frame_height": profile.frame_height,
        "hoop_bbox_norm": profile.hoop_bbox_norm,
    }))

    ground_truth_path = data_root / _V2_GROUND_TRUTH_DIR / f"video_{video.id}.json"

    spec = {
        "kind": "detect_markers",
        "video": str(video_abs_path),
        "video_id": video.id,
        "config_path": str(config_path),
        "ground_truth_path": str(ground_truth_path),
        "user": require_user_email(),
    }
    return jobs.start_job(spec)


@bp.get("")
def list_all_videos():
    require_user_email()
    game_id = request.args.get("game_id", type=int)
    with get_session() as session:
        return jsonify([_serialize(v) for v in list_videos(session, game_id=game_id)])


@bp.get("/<int:video_id>")
def get_one_video(video_id):
    require_user_email()
    with get_session() as session:
        video = get_video(session, video_id)
        if video is None:
            abort(404, "video not found")
        return jsonify(_serialize(video))


@bp.get("/<int:video_id>/calibration-frame")
def get_calibration_frame(video_id):
    """A still from this video for the SPA's basket-calibration canvas to
    draw a box on - reuses extract_calibration_frame (shared with the
    legacy /api/calibrate-frame), resolving the path server-side from the
    video's own gcs_relpath so the SPA never needs to know DATA_ROOT itself."""
    from ..app import DATA_ROOT, extract_calibration_frame

    require_user_email()
    with get_session() as session:
        video = get_video(session, video_id)
        if video is None:
            abort(404, "video not found")
        video_abs_path = DATA_ROOT / video.gcs_relpath
    if not video_abs_path.is_file():
        abort(400, f"video file not found at {video_abs_path}")
    t = request.args.get("t", type=float)
    out_path = extract_calibration_frame(video_abs_path, t)
    return send_from_directory(out_path.parent, out_path.name, conditional=True)


@bp.post("")
def register_video():
    """Called by the SPA right after /api/uploads/complete confirms the
    browser's direct-to-GCS upload actually landed - this app never sees
    the upload traffic itself, so this is the hook that turns "bytes are on
    GCS" into a row other users can see and label.

    calibration_profile_id is optional and NOT auto-selected: each video's
    basket is typically in a different spot (different camera/game setup),
    so silently reusing whatever calibration was created most recently
    would often point detection at the wrong region entirely. The caller
    picks an existing one (if the camera setup genuinely repeats) or leaves
    it unset and attaches one later via PATCH once the video exists."""
    from ..app import DATA_ROOT  # deferred: avoids a hard import-time dependency on app.py

    email = require_user_email()
    body = request.get_json(force=True) or {}
    gcs_relpath = body.get("gcs_relpath")
    original_filename = body.get("original_filename")
    game_id = body.get("game_id")
    calibration_profile_id = body.get("calibration_profile_id")
    if not gcs_relpath or not original_filename:
        abort(400, "missing gcs_relpath or original_filename")
    if not game_id:
        abort(400, "missing game_id")

    with get_session() as session:
        # Idempotent on gcs_relpath: a retried "upload finished" call (flaky
        # network, double-click) re-registering the same GCS object should
        # hand back the video that already exists, not 500 on the unique
        # constraint or create a second row pointing at the same bytes.
        existing = get_by_gcs_relpath(session, gcs_relpath)
        if existing is not None:
            return jsonify({**_serialize(existing), "job_id": None}), 200

        if get_game(session, game_id) is None:
            abort(400, "game not found")
        profile = get_profile(session, calibration_profile_id) if calibration_profile_id else None
        if calibration_profile_id and profile is None:
            abort(400, "basket calibration not found")
        user = get_or_create_user(session, email)
        video = create_video(session, user.id, game_id, gcs_relpath, original_filename,
                              calibration_profile_id=profile.id if profile else None)
        job_id = _queue_detect(session, video, profile, DATA_ROOT) if profile else None
        payload = _serialize(video)
    payload["job_id"] = job_id
    return jsonify(payload), 201


@bp.patch("/<int:video_id>")
def patch_video(video_id):
    """Attach/replace a basket calibration and/or move a video to a
    different game - either field may be given, at least one is required.
    Attaching a calibration (re-)triggers detection."""
    from ..app import DATA_ROOT

    require_user_email()
    body = request.get_json(force=True) or {}
    calibration_profile_id = body.get("calibration_profile_id")
    game_id = body.get("game_id")
    if not calibration_profile_id and not game_id:
        abort(400, "missing calibration_profile_id or game_id")

    with get_session() as session:
        video = get_video(session, video_id)
        if video is None:
            abort(404, "video not found")

        if game_id:
            if get_game(session, game_id) is None:
                abort(400, "game not found")
            video = set_game(session, video_id, game_id)

        job_id = None
        if calibration_profile_id:
            profile = get_profile(session, calibration_profile_id)
            if profile is None:
                abort(400, "basket calibration not found")
            video = attach_calibration_profile(session, video_id, calibration_profile_id)
            job_id = _queue_detect(session, video, profile, DATA_ROOT)

        payload = _serialize(video)
    payload["job_id"] = job_id
    return jsonify(payload)
