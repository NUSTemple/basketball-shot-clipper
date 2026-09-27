"""v2 video registry - the metadata half of an uploaded video (bytes stay
on GCS, registered separately via the existing /api/uploads +
/api/uploads/complete flow, unchanged). Visibility is fully open
(docs/REQUIREMENTS_V2.md #2): GET routes list every video, not just the
caller's own.
"""
import json
from pathlib import Path

from flask import Blueprint, abort, jsonify, request

from ...db.repositories.calibration_profiles import get_profile
from ...db.repositories.users import get_or_create_user
from ...db.repositories.videos import (
    attach_calibration_profile,
    create_video,
    get_by_gcs_relpath,
    get_video,
    list_videos,
)
from ...db.session import get_session
from .. import jobs
from ..auth import require_user_email

bp = Blueprint("api_v2_videos", __name__, url_prefix="/api/v2/videos")

# Where a calibration profile gets materialized into the JSON shape
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
    """Materialize the video's calibration profile into a transient config
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
    with get_session() as session:
        return jsonify([_serialize(v) for v in list_videos(session)])


@bp.get("/<int:video_id>")
def get_one_video(video_id):
    require_user_email()
    with get_session() as session:
        video = get_video(session, video_id)
        if video is None:
            abort(404, "video not found")
        return jsonify(_serialize(video))


@bp.post("")
def register_video():
    """Called by the SPA right after /api/uploads/complete confirms the
    browser's direct-to-GCS upload actually landed - this app never sees
    the upload traffic itself, so this is the hook that turns "bytes are on
    GCS" into a row other users can see and label."""
    from ..app import DATA_ROOT  # deferred: avoids a hard import-time dependency on app.py

    email = require_user_email()
    body = request.get_json(force=True) or {}
    gcs_relpath = body.get("gcs_relpath")
    original_filename = body.get("original_filename")
    if not gcs_relpath or not original_filename:
        abort(400, "missing gcs_relpath or original_filename")

    calibration_profile_id = body.get("calibration_profile_id")
    with get_session() as session:
        # Idempotent on gcs_relpath: a retried "upload finished" call (flaky
        # network, double-click) re-registering the same GCS object should
        # hand back the video that already exists, not 500 on the unique
        # constraint or create a second row pointing at the same bytes.
        existing = get_by_gcs_relpath(session, gcs_relpath)
        if existing is not None:
            return jsonify({**_serialize(existing), "job_id": None}), 200

        profile = get_profile(session, calibration_profile_id) if calibration_profile_id else None
        if calibration_profile_id and profile is None:
            abort(400, "calibration profile not found")
        user = get_or_create_user(session, email)
        video = create_video(session, user.id, gcs_relpath, original_filename,
                              calibration_profile_id=calibration_profile_id)
        job_id = _queue_detect(session, video, profile, DATA_ROOT) if profile else None
        payload = _serialize(video)
    payload["job_id"] = job_id
    return jsonify(payload), 201


@bp.patch("/<int:video_id>")
def patch_video(video_id):
    """Attach (or replace) a calibration profile on an already-uploaded
    video and trigger detection - for a video uploaded before any profile
    existed yet, or to re-detect against a corrected profile."""
    from ..app import DATA_ROOT

    require_user_email()
    body = request.get_json(force=True) or {}
    calibration_profile_id = body.get("calibration_profile_id")
    if not calibration_profile_id:
        abort(400, "missing calibration_profile_id")

    with get_session() as session:
        if get_video(session, video_id) is None:
            abort(404, "video not found")
        profile = get_profile(session, calibration_profile_id)
        if profile is None:
            abort(400, "calibration profile not found")
        video = attach_calibration_profile(session, video_id, calibration_profile_id)
        job_id = _queue_detect(session, video, profile, DATA_ROOT)
        payload = _serialize(video)
    payload["job_id"] = job_id
    return jsonify(payload)
