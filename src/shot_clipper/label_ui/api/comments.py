"""Flat, timestamp-pinned comments on a marker (docs/REQUIREMENTS_V2.md #5).
Same author/uploader/admin permission model as labels."""
from flask import Blueprint, abort, jsonify, request

from ...db.repositories.comments import (
    add_comment,
    delete_comment,
    get_comment,
    list_for_marker,
    update_comment,
)
from ...db.repositories.markers import get_marker
from ...db.repositories.users import get_or_create_user
from ...db.session import get_session
from ..auth import is_admin, require_user_email

bp = Blueprint("api_v2_comments", __name__, url_prefix="/api/v2")


def _serialize(comment):
    return {
        "id": comment.id,
        "marker_id": comment.marker_id,
        "user_id": comment.user_id,
        "text": comment.text,
        "created_at": comment.created_at.isoformat() if comment.created_at else None,
        "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
    }


def _can_moderate(comment, email: str, user_id: int) -> bool:
    if comment.user_id == user_id or is_admin(email):
        return True
    return comment.marker.video.owner_user_id == user_id


@bp.get("/markers/<int:marker_id>/comments")
def list_comments(marker_id):
    require_user_email()
    with get_session() as session:
        if get_marker(session, marker_id) is None:
            abort(404, "marker not found")
        return jsonify([_serialize(c) for c in list_for_marker(session, marker_id)])


@bp.post("/markers/<int:marker_id>/comments")
def create_comment(marker_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        abort(400, "missing text")
    with get_session() as session:
        if get_marker(session, marker_id) is None:
            abort(404, "marker not found")
        user = get_or_create_user(session, email)
        comment = add_comment(session, marker_id, user.id, text)
        return jsonify(_serialize(comment)), 201


@bp.patch("/comments/<int:comment_id>")
def patch_comment(comment_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        abort(400, "missing text")
    with get_session() as session:
        comment = get_comment(session, comment_id)
        if comment is None:
            abort(404, "comment not found")
        user = get_or_create_user(session, email)
        if comment.user_id != user.id:
            abort(403, "only the author can edit a comment")
        comment = update_comment(session, comment_id, text)
        return jsonify(_serialize(comment))


@bp.delete("/comments/<int:comment_id>")
def remove_comment(comment_id):
    email = require_user_email()
    with get_session() as session:
        comment = get_comment(session, comment_id)
        if comment is None:
            abort(404, "comment not found")
        user = get_or_create_user(session, email)
        if not _can_moderate(comment, email, user.id):
            abort(403, "only the author, the video's uploader, or an admin can delete this comment")
        delete_comment(session, comment_id)
        return "", 204
