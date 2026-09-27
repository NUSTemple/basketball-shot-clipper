"""v2 API blueprints, mounted under /api/v2/ - deliberately separate from
today's /api/* routes in app.py so the new platform's backend can be built
incrementally without touching (or risking) the existing single-user-per-
clip label/review/export flow those routes serve.
"""
from flask import Flask

from . import calibration_profiles, dbcheck, jobs, markers, profile, videos


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(dbcheck.bp)
    app.register_blueprint(profile.bp)
    app.register_blueprint(calibration_profiles.bp)
    app.register_blueprint(videos.bp)
    app.register_blueprint(markers.bp)
    app.register_blueprint(jobs.bp)
