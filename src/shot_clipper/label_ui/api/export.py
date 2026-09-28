"""Cross-video export/cut (docs/REQUIREMENTS_V2.md #6) - filter markers by
label category + player + comment keyword across every video (visibility is
fully open), then cut matching ranges into real clip files on demand. One
job spans every matched video (same multi-video-per-job shape as the
legacy /api/process-batch), so there's a single job_id to poll and a
single zip to download once it's done - see worker.py's _run_cut_markers.
"""
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, abort, jsonify, request, send_file

from ...db.repositories.cut_clips import find_matching_markers, get_cut_clip
from ...db.repositories.games import get_game
from ...db.repositories.users import get_or_create_user
from ...db.repositories.videos import get_video
from ...db.session import get_session
from .. import jobs
from ..auth import require_user_email

bp = Blueprint("api_v2_export", __name__, url_prefix="/api/v2/export")

DEFAULT_PRE_S = 5.0
DEFAULT_POST_S = 2.0
_V2_CLIPS_DIR = "_v2_clips"


@bp.post("")
def create_export():
    from ..app import DATA_ROOT

    email = require_user_email()
    body = request.get_json(force=True) or {}
    categories = body.get("categories") or None
    player = (body.get("player") or "").strip() or None
    comment_keyword = (body.get("comment_keyword") or "").strip() or None
    game_id = body.get("game_id")
    # game_id alone is a valid, complete request ("export this game's
    # highlights") - see find_matching_markers - so it's included in the
    # "at least one filter" check, not just an AND-narrowing extra.
    if not (categories or player or comment_keyword or game_id):
        abort(400, "provide at least one of categories, player, comment_keyword, or game_id")

    pre = float(body.get("pre", DEFAULT_PRE_S))
    post = float(body.get("post", DEFAULT_POST_S))
    if pre < 0 or post < 0:
        abort(400, "pre/post must be non-negative")

    with get_session() as session:
        if game_id and get_game(session, game_id) is None:
            abort(400, "game not found")
        markers = find_matching_markers(session, category_names=categories, player=player,
                                         comment_keyword=comment_keyword, game_id=game_id)
        if not markers:
            return jsonify({"job_id": None, "video_count": 0, "marker_count": 0})

        by_video: dict[int, list[dict]] = {}
        for m in markers:
            by_video.setdefault(m.video_id, []).append({"id": m.id, "timestamp_s": float(m.timestamp_s)})

        video_specs = []
        for video_id, marker_list in by_video.items():
            video = get_video(session, video_id)
            if video is None:
                continue  # shouldn't happen, but a matched marker's video was deleted concurrently
            video_specs.append({
                "video_id": video_id,
                "video_path": str(DATA_ROOT / video.gcs_relpath),
                "out_dir": str(DATA_ROOT / _V2_CLIPS_DIR / f"video_{video_id}"),
                "gcs_relpath_prefix": f"{_V2_CLIPS_DIR}/video_{video_id}/",
                "markers": marker_list,
            })
        user = get_or_create_user(session, email)

    if not video_specs:
        return jsonify({"job_id": None, "video_count": 0, "marker_count": 0})

    spec = {
        "kind": "cut_markers",
        "videos": video_specs,
        "pre": pre,
        "post": post,
        "created_by_user_id": user.id,
        "user": email,
    }
    job_id = jobs.start_job(spec)
    return jsonify({
        "job_id": job_id,
        "video_count": len(video_specs),
        "marker_count": len(markers),
    })


@bp.get("/<job_id>/download")
def download_export(job_id):
    from ..app import DATA_ROOT

    require_user_email()
    job = jobs.get_job(job_id)
    if job is None:
        abort(404, "export job not found")
    if job.get("kind") != "cut_markers":
        abort(400, "not an export job")
    if job.get("state") != "done":
        abort(409, f"export isn't finished yet (state: {job.get('state')})")

    cut_clip_ids = job.get("cut_clip_ids") or []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        with get_session() as session:
            for cut_clip_id in cut_clip_ids:
                cut = get_cut_clip(session, cut_clip_id)
                if cut is None:
                    continue
                src = DATA_ROOT / cut.gcs_relpath
                if src.is_file():
                    zf.write(src, Path(cut.gcs_relpath).name)
    buf.seek(0)
    filename = f"shot-clipper-export-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.zip"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=filename)
