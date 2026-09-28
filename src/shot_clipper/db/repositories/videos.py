"""CRUD for `videos` rows - the metadata half of an uploaded video (the
bytes stay on GCS at gcs_relpath, relative to DATA_ROOT). Visibility is
fully open (docs/REQUIREMENTS_V2.md #2), so list_videos has no owner
filter."""
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Video


def create_video(session: Session, owner_user_id: int, game_id: int, gcs_relpath: str, original_filename: str,
                  duration_s: float | None = None,
                  calibration_profile_id: int | None = None) -> Video:
    video = Video(
        owner_user_id=owner_user_id,
        game_id=game_id,
        gcs_relpath=gcs_relpath,
        original_filename=original_filename,
        duration_s=duration_s,
        calibration_profile_id=calibration_profile_id,
        status="detecting" if calibration_profile_id else "uploaded",
    )
    session.add(video)
    session.flush()
    return video


def get_video(session: Session, video_id: int) -> Video | None:
    return session.get(Video, video_id)


def get_by_gcs_relpath(session: Session, gcs_relpath: str) -> Video | None:
    return session.scalar(select(Video).where(Video.gcs_relpath == gcs_relpath))


def list_videos(session: Session, game_id: int | None = None) -> list[Video]:
    stmt = select(Video).order_by(Video.created_at.desc())
    if game_id is not None:
        stmt = stmt.where(Video.game_id == game_id)
    return list(session.scalars(stmt))


def set_game(session: Session, video_id: int, game_id: int) -> Video | None:
    """Move a video to a different game - a manual escape hatch for when a
    video ends up under the wrong one (see the gcs_relpath-collision fix in
    app.py's api_create_upload for the bug this covers for)."""
    video = session.get(Video, video_id)
    if video is None:
        return None
    video.game_id = game_id
    session.flush()
    return video


def attach_calibration_profile(session: Session, video_id: int, calibration_profile_id: int) -> Video | None:
    video = session.get(Video, video_id)
    if video is None:
        return None
    video.calibration_profile_id = calibration_profile_id
    video.status = "detecting"
    video.detect_error = None
    session.flush()
    return video


def set_status(session: Session, video_id: int, status: str, detect_error: str | None = None,
               duration_s: float | None = None) -> Video | None:
    video = session.get(Video, video_id)
    if video is None:
        return None
    video.status = status
    video.detect_error = detect_error
    if duration_s is not None:
        video.duration_s = duration_s
    if status == "ready":
        video.detected_at = datetime.now(timezone.utc)
    session.flush()
    return video
