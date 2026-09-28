"""Games (docs/REQUIREMENTS_V2.md #8) - the primary organizing unit for the
v2 workflow. A game tags which subset of the global roster participated;
adding a player reuses the same get-or-create pattern as the existing
/api/v2/roster (a typed name either matches an existing roster player or
creates a new one)."""
from datetime import datetime

from flask import Blueprint, abort, jsonify, request

from ...db.repositories.games import (
    add_player_to_game,
    create_game,
    get_game,
    list_game_players,
    list_games,
    remove_player_from_game,
)
from ...db.repositories.roster import add_player
from ...db.repositories.users import get_or_create_user
from ...db.session import get_session
from ..auth import require_user_email

bp = Blueprint("api_v2_games", __name__, url_prefix="/api/v2/games")


def _serialize(game, players=None):
    return {
        "id": game.id,
        "name": game.name,
        "location": game.location,
        "game_date": game.game_date.isoformat() if game.game_date else None,
        "created_at": game.created_at.isoformat() if game.created_at else None,
        "players": [{"id": p.id, "name": p.name} for p in players] if players is not None else None,
    }


def _parse_game_date(raw) -> datetime:
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        abort(400, "game_date must be an ISO 8601 date/datetime string")


@bp.get("")
def list_all_games():
    require_user_email()
    with get_session() as session:
        games = list_games(session)
        return jsonify([_serialize(g, list_game_players(session, g.id)) for g in games])


@bp.post("")
def create_new_game():
    email = require_user_email()
    body = request.get_json(force=True) or {}
    location = (body.get("location") or "").strip()
    if not location:
        abort(400, "missing location")
    game_date = _parse_game_date(body.get("game_date"))
    name = (body.get("name") or "").strip() or None
    with get_session() as session:
        user = get_or_create_user(session, email)
        game = create_game(session, user.id, location, game_date, name=name)
        return jsonify(_serialize(game, [])), 201


@bp.get("/<int:game_id>")
def get_one_game(game_id):
    require_user_email()
    with get_session() as session:
        game = get_game(session, game_id)
        if game is None:
            abort(404, "game not found")
        return jsonify(_serialize(game, list_game_players(session, game_id)))


@bp.post("/<int:game_id>/players")
def add_game_player(game_id):
    email = require_user_email()
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, "missing name")
    with get_session() as session:
        game = get_game(session, game_id)
        if game is None:
            abort(404, "game not found")
        user = get_or_create_user(session, email)
        player, _created = add_player(session, name, user.id)
        add_player_to_game(session, game_id, player.id)
        return jsonify(_serialize(game, list_game_players(session, game_id))), 201


@bp.delete("/<int:game_id>/players/<int:roster_player_id>")
def remove_game_player(game_id, roster_player_id):
    require_user_email()
    with get_session() as session:
        if get_game(session, game_id) is None:
            abort(404, "game not found")
        if not remove_player_from_game(session, game_id, roster_player_id):
            abort(404, "player not in this game")
        return "", 204
