"""Global, flat player roster on Postgres (docs/REQUIREMENTS_V2.md #4) - any
allowlisted user can add a player, matching the legacy roster.py's
add_player() being open to anyone in the Review panel today."""
from flask import Blueprint, abort, jsonify, request

from ...db.repositories.roster import add_player, list_players
from ...db.repositories.users import get_or_create_user
from ...db.session import get_session
from ..auth import require_user_email

bp = Blueprint("api_v2_roster", __name__, url_prefix="/api/v2/roster")


@bp.get("")
def get_roster():
    require_user_email()
    with get_session() as session:
        return jsonify([p.name for p in list_players(session)])


@bp.post("")
def post_roster_player():
    email = require_user_email()
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, "missing name")
    with get_session() as session:
        user = get_or_create_user(session, email)
        add_player(session, name, user.id)
        return jsonify([p.name for p in list_players(session)]), 201
