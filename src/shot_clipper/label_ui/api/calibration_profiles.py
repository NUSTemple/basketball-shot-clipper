"""Reusable per-camera/court calibration profiles (docs/REQUIREMENTS_V2.md
#3) - replaces per-video-filename calibration. The browser side of drawing
the box is unchanged: this only replaces where /api/save-calibration used
to write (a JSON file) with a Postgres row a video can reference by id."""
from flask import Blueprint, abort, jsonify, request

from ...db.repositories.calibration_profiles import (
    create_profile,
    get_profile,
    list_profiles,
    update_bbox,
)
from ...db.repositories.users import get_or_create_user
from ...db.session import get_session
from ..auth import require_user_email

bp = Blueprint("api_v2_calibration_profiles", __name__, url_prefix="/api/v2/calibration-profiles")


def _serialize(profile):
    return {
        "id": profile.id,
        "name": profile.name,
        "hoop_bbox_norm": profile.hoop_bbox_norm,
        "frame_width": profile.frame_width,
        "frame_height": profile.frame_height,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
    }


def _validate_bbox(body: dict) -> list:
    bbox = body.get("hoop_bbox_norm")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
        abort(400, "hoop_bbox_norm must be a 4-element [x1, y1, x2, y2] list")
    return bbox


@bp.get("")
def list_calibration_profiles():
    require_user_email()  # any allowlisted user can see the global list
    with get_session() as session:
        return jsonify([_serialize(p) for p in list_profiles(session)])


@bp.post("")
def create_calibration_profile():
    email = require_user_email()
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, "missing name")
    bbox = _validate_bbox(body)
    with get_session() as session:
        user = get_or_create_user(session, email)
        try:
            profile = create_profile(
                session, name, bbox,
                body.get("frame_width"), body.get("frame_height"),
                created_by_user_id=user.id,
            )
        except Exception:
            abort(409, f"a calibration profile named {name!r} already exists")
        return jsonify(_serialize(profile)), 201


@bp.patch("/<int:profile_id>")
def patch_calibration_profile(profile_id):
    require_user_email()
    body = request.get_json(force=True) or {}
    bbox = _validate_bbox(body)
    with get_session() as session:
        profile = update_bbox(session, profile_id, bbox, body.get("frame_width"), body.get("frame_height"))
        if profile is None:
            abort(404, "calibration profile not found")
        return jsonify(_serialize(profile))
