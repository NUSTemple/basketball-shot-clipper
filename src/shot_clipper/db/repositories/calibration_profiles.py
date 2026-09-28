"""CRUD for reusable calibration profiles (docs/REQUIREMENTS_V2.md #3) - the
replacement for per-video-filename calibration files. Visibility is fully
open (see REQUIREMENTS_V2.md #2), so list/get have no owner filter;
created_by is provenance only."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import CalibrationProfile


def create_profile(session: Session, name: str, hoop_bbox_norm: list,
                    frame_width: int | None, frame_height: int | None,
                    created_by_user_id: int) -> CalibrationProfile:
    profile = CalibrationProfile(
        name=name,
        hoop_bbox_norm=hoop_bbox_norm,
        frame_width=frame_width,
        frame_height=frame_height,
        created_by=created_by_user_id,
    )
    session.add(profile)
    session.flush()
    return profile


def list_profiles(session: Session) -> list[CalibrationProfile]:
    return list(session.scalars(select(CalibrationProfile).order_by(CalibrationProfile.name)))


def get_profile(session: Session, profile_id: int) -> CalibrationProfile | None:
    return session.get(CalibrationProfile, profile_id)


def get_most_recent_profile(session: Session) -> CalibrationProfile | None:
    """Auto-attach target for a new video's upload (Phase 7) - most
    recently *created*, not most recently *used* (no usage-tracking column
    exists; that's a possible future refinement, not built now)."""
    return session.scalar(select(CalibrationProfile).order_by(CalibrationProfile.created_at.desc()).limit(1))


def update_bbox(session: Session, profile_id: int, hoop_bbox_norm: list,
                 frame_width: int | None, frame_height: int | None) -> CalibrationProfile | None:
    profile = session.get(CalibrationProfile, profile_id)
    if profile is None:
        return None
    profile.hoop_bbox_norm = hoop_bbox_norm
    if frame_width is not None:
        profile.frame_width = frame_width
    if frame_height is not None:
        profile.frame_height = frame_height
    session.flush()
    return profile
