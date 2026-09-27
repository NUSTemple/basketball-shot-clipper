"""Dev/ops connectivity check for the v2 Postgres store - confirms
SHOT_CLIPPER_DATABASE_URL is set and reachable from wherever this process is
running (local shell, label-ui container, worker container). Returns no
data beyond a timestamp, so it's harmless to leave reachable."""
from flask import Blueprint, jsonify
from sqlalchemy import text

from ...db.session import get_session

bp = Blueprint("api_v2_dbcheck", __name__, url_prefix="/api/v2")


@bp.route("/_dbcheck")
def dbcheck():
    with get_session() as session:
        now = session.execute(text("SELECT now()")).scalar()
    return jsonify({"ok": True, "server_time": str(now)})
