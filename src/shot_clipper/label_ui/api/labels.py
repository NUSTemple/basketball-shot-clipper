"""Per-user-attributed labels on a marker (docs/REQUIREMENTS_V2.md #4) - each
user's (category, player) tag is its own row, never overwritten by another
user's. Permissions: the author can edit/delete their own; the marker's
video owner or a platform admin can delete (not edit) anyone's, for
moderation.
"""
from flask import Blueprint, abort, jsonify, request

from ...db.repositories.labels import (
    add_label,
    delete_label,
    get_label,
    list_for_marker,
    update_label,
)
from ...db.repositories.markers import get_marker
from ...db.repositories.users import get_or_create_user
from ...db.session import get_session
from ..auth import is_admin, require_user_email

bp = Blueprint("api_v2_labels", __name__, url_prefix="/api/v2")


def _serialize(label):
    return {
        "id": label.id,
        "marker_id": label.marker_id,
        "user_id": label.user_id,
        "category_id": label.category_id,
        "player_name": label.player_name,
        "created_at": label.created_at.isoformat() if label.created_at else None,
    }


def _can_moderate(session, label, email: str, user_id: int) -> bool:
    if label.user_id == user_id or is_admin(email):
        return True
    return label.marker.video.owner_user_id == user_id


@bp.get("/markers/<int:marker_id>/labels")
def list_labels(marker_id):
    require_user_email()
    with get_session() as session:
        if get_marker(session, marker_id) is None:
            abort(404, "marker not found")
        return jsonify([_serialize(l) for l in list_for_marker(session, marker_id)])


@bp.post("/markers/<int:marker_id>/labels")
def create_label(marker_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    category_id = body.get("category_id")
    if not category_id:
        abort(400, "missing category_id")
    player_name = (body.get("player_name") or "").strip() or None
    with get_session() as session:
        if get_marker(session, marker_id) is None:
            abort(404, "marker not found")
        user = get_or_create_user(session, email)
        try:
            label, created = add_label(session, marker_id, user.id, category_id, player_name)
        except Exception:
            abort(400, "invalid category_id")
        return jsonify(_serialize(label)), (201 if created else 200)


@bp.patch("/labels/<int:label_id>")
def patch_label(label_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    with get_session() as session:
        label = get_label(session, label_id)
        if label is None:
            abort(404, "label not found")
        user = get_or_create_user(session, email)
        if label.user_id != user.id:
            abort(403, "only the author can edit a label")
        try:
            label = update_label(
                session, label_id,
                category_id=body.get("category_id"),
                player_name=(body.get("player_name") or "").strip() or None,
                player_name_given="player_name" in body,
            )
        except Exception:
            abort(409, "you already have an identical label on this marker")
        return jsonify(_serialize(label))


@bp.delete("/labels/<int:label_id>")
def remove_label(label_id):
    email = require_user_email()
    with get_session() as session:
        label = get_label(session, label_id)
        if label is None:
            abort(404, "label not found")
        user = get_or_create_user(session, email)
        if not _can_moderate(session, label, email, user.id):
            abort(403, "only the author, the video's uploader, or an admin can delete this label")
        delete_label(session, label_id)
        return "", 204
