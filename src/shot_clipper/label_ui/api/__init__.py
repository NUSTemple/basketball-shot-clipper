"""v2 API blueprints, mounted under /api/v2/ - deliberately separate from
today's /api/* routes in app.py so the new platform's backend can be built
incrementally without touching (or risking) the existing single-user-per-
clip label/review/export flow those routes serve.
"""
from flask import Flask

from . import dbcheck


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(dbcheck.bp)
