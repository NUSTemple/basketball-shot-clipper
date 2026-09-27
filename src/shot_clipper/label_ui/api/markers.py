"""Marker timeline API (docs/REQUIREMENTS_V2.md #3) - a marker is a bare
timestamp with a source/state, never a category/label itself (Phase 3 adds
labels/comments on top of these). Two URL shapes on one blueprint:
/videos/<id>/markers (list/create, scoped to a video) and /markers/<id>
(read/update/delete a single marker, since a marker id alone is enough to
find it - no need to also carry its video id in the URL).
"""
from flask import Blueprint, abort, jsonify, request

from ...db.repositories.markers import (
    create_manual_marker,
    delete_marker,
    get_marker,
    list_for_video,
    set_state,
    set_timestamp,
)
from ...db.repositories.users import get_or_create_user
from ...db.repositories.videos import get_video
from ...db.session import get_session
from ..auth import require_user_email

bp = Blueprint("api_v2_markers", __name__, url_prefix="/api/v2")

VALID_STATES = {"confirmed", "dismissed", "unconfirmed"}


def _serialize(marker):
    return {
        "id": marker.id,
        "video_id": marker.video_id,
        "timestamp_s": float(marker.timestamp_s),
        "source": marker.source,
        "state": marker.state,
        "created_by_user_id": marker.created_by_user_id,
        "confirmed_by_user_id": marker.confirmed_by_user_id,
        "confirmed_at": marker.confirmed_at.isoformat() if marker.confirmed_at else None,
        "created_at": marker.created_at.isoformat() if marker.created_at else None,
    }


@bp.get("/videos/<int:video_id>/markers")
def list_markers(video_id):
    require_user_email()
    with get_session() as session:
        if get_video(session, video_id) is None:
            abort(404, "video not found")
        return jsonify([_serialize(m) for m in list_for_video(session, video_id)])


@bp.post("/videos/<int:video_id>/markers")
def create_marker(video_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    timestamp_s = body.get("timestamp_s")
    if not isinstance(timestamp_s, (int, float)) or timestamp_s < 0:
        abort(400, "timestamp_s must be a non-negative number")
    with get_session() as session:
        if get_video(session, video_id) is None:
            abort(404, "video not found")
        user = get_or_create_user(session, email)
        marker = create_manual_marker(session, video_id, timestamp_s, user.id)
        return jsonify(_serialize(marker)), 201


@bp.patch("/markers/<int:marker_id>")
def patch_marker(marker_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    with get_session() as session:
        marker = get_marker(session, marker_id)
        if marker is None:
            abort(404, "marker not found")
        if "state" in body:
            state = body["state"]
            if state not in VALID_STATES:
                abort(400, f"state must be one of {sorted(VALID_STATES)}")
            confirmed_by = get_or_create_user(session, email).id if state == "confirmed" else None
            marker = set_state(session, marker_id, state, confirmed_by_user_id=confirmed_by)
        if "timestamp_s" in body:
            ts = body["timestamp_s"]
            if not isinstance(ts, (int, float)) or ts < 0:
                abort(400, "timestamp_s must be a non-negative number")
            marker = set_timestamp(session, marker_id, ts)
        return jsonify(_serialize(marker))


@bp.delete("/markers/<int:marker_id>")
def remove_marker(marker_id):
    require_user_email()
    with get_session() as session:
        if get_marker(session, marker_id) is None:
            abort(404, "marker not found")
        if not delete_marker(session, marker_id):
            abort(409, "marker has labels or comments - delete those first")
        return "", 204
