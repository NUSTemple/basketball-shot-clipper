"""CRUD for `markers` rows - a timestamp of interest on a video's timeline,
either an unconfirmed YOLO suggestion (source='auto') or user-placed
(source='manual'). No category/label lives here - see labels.py (Phase 3)."""
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Marker


def bulk_insert_auto_markers(session: Session, video_id: int, timestamps_s: list[float]) -> list[Marker]:
    """One row per detected candidate timestamp, all unconfirmed - see
    label_ui/worker.py's detect_markers job kind."""
    markers = [Marker(video_id=video_id, timestamp_s=t, source="auto", state="unconfirmed") for t in timestamps_s]
    session.add_all(markers)
    session.flush()
    return markers


def create_manual_marker(session: Session, video_id: int, timestamp_s: float, created_by_user_id: int) -> Marker:
    """Manual markers start *confirmed*, not unconfirmed - a human placing
    one by hand is itself the confirmation, unlike an auto-detected
    suggestion that still needs a human to accept/reject it. This is what
    makes a manual marker eligible for "export this game's confirmed
    markers" (docs/REQUIREMENTS_V2.md #8) without an extra do-nothing
    confirm click."""
    now = datetime.now(timezone.utc)
    marker = Marker(
        video_id=video_id, timestamp_s=timestamp_s, source="manual",
        state="confirmed", created_by_user_id=created_by_user_id,
        confirmed_by_user_id=created_by_user_id, confirmed_at=now,
    )
    session.add(marker)
    session.flush()
    return marker


def list_for_video(session: Session, video_id: int) -> list[Marker]:
    return list(session.scalars(select(Marker).where(Marker.video_id == video_id).order_by(Marker.timestamp_s)))


def get_marker(session: Session, marker_id: int) -> Marker | None:
    return session.get(Marker, marker_id)


def set_state(session: Session, marker_id: int, state: str, confirmed_by_user_id: int | None = None) -> Marker | None:
    marker = session.get(Marker, marker_id)
    if marker is None:
        return None
    marker.state = state
    if state == "confirmed":
        marker.confirmed_by_user_id = confirmed_by_user_id
        marker.confirmed_at = datetime.now(timezone.utc)
    session.flush()
    return marker


def set_timestamp(session: Session, marker_id: int, timestamp_s: float) -> Marker | None:
    marker = session.get(Marker, marker_id)
    if marker is None:
        return None
    marker.timestamp_s = timestamp_s
    session.flush()
    return marker


def delete_marker(session: Session, marker_id: int) -> bool:
    """False (no-op) if the marker has any labels/comments already, so a
    delete can't silently orphan another user's attribution - callers
    should surface that as a 409, not a 404."""
    marker = session.get(Marker, marker_id)
    if marker is None:
        return True
    if marker.labels or marker.comments:
        return False
    session.delete(marker)
    session.flush()
    return True
