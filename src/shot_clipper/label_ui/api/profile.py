from flask import Blueprint, jsonify, request

from ...db.repositories.users import update_profile
from ...db.session import get_session
from ..auth import require_user_email

bp = Blueprint("api_v2_profile", __name__, url_prefix="/api/v2")


def _serialize(user):
    return {"email": user.email, "display_name": user.display_name, "avatar_url": user.avatar_url}


@bp.get("/profile")
def get_profile():
    email = require_user_email()
    with get_session() as session:
        user = update_profile(session, email, display_name=None, avatar_url=None)
        return jsonify(_serialize(user))


@bp.patch("/profile")
def patch_profile():
    email = require_user_email()
    body = request.get_json(force=True) or {}
    with get_session() as session:
        user = update_profile(
            session, email,
            display_name=body.get("display_name"),
            avatar_url=body.get("avatar_url"),
        )
        return jsonify(_serialize(user))
