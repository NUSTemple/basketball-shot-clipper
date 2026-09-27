"""Admin-managed label category list (docs/REQUIREMENTS_V2.md #4), seeded
with Goal/Assist/Block. Read is open to any allowlisted user; create/rename/
soft-delete require admin."""
from flask import Blueprint, abort, jsonify, request

from ...db.repositories.categories import create_category, list_categories, update_category
from ...db.repositories.users import get_or_create_user
from ...db.session import get_session
from ..auth import require_admin, require_user_email

bp = Blueprint("api_v2_categories", __name__, url_prefix="/api/v2/label-categories")


def _serialize(category):
    return {"id": category.id, "name": category.name, "active": category.active}


@bp.get("")
def list_label_categories():
    require_user_email()
    include_inactive = request.args.get("all") == "1"
    with get_session() as session:
        return jsonify([_serialize(c) for c in list_categories(session, include_inactive=include_inactive)])


@bp.post("")
def create_label_category():
    email = require_admin()
    body = request.get_json(force=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        abort(400, "missing name")
    with get_session() as session:
        user = get_or_create_user(session, email)
        try:
            category = create_category(session, name, user.id)
        except Exception:
            abort(409, f"a label category named {name!r} already exists")
        return jsonify(_serialize(category)), 201


@bp.patch("/<int:category_id>")
def patch_label_category(category_id):
    require_admin()
    body = request.get_json(force=True) or {}
    with get_session() as session:
        category = update_category(
            session, category_id,
            active=body.get("active"),  # None (omitted) leaves it unchanged
            name=(body.get("name") or "").strip() or None,
        )
        if category is None:
            abort(404, "label category not found")
        return jsonify(_serialize(category))
