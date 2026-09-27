"""Thin read-only wrapper over the existing job queue (label_ui/jobs.py,
reused unmodified) - lets the SPA poll a detect_markers/cut_markers job's
progress the same way Job Status already polls detect/cut jobs today."""
from flask import Blueprint, abort, jsonify

from .. import jobs
from ..auth import require_user_email

bp = Blueprint("api_v2_jobs", __name__, url_prefix="/api/v2/jobs")


@bp.get("/<job_id>")
def get_job_status(job_id):
    require_user_email()
    job = jobs.get_job(job_id)
    if job is None:
        abort(404, "job not found")
    job = dict(job)
    job["queue_position"] = jobs.get_queue_position(job_id)
    return jsonify(job)
