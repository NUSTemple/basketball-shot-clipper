"""CRUD for `labels` rows - one per (marker, user, category, player) combo,
never overwritten by another user's row (docs/REQUIREMENTS_V2.md #4). A
marker's "3x Goal, 1x Block" view is computed by the caller from
list_for_marker's rows, not stored anywhere."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Label


def add_label(session: Session, marker_id: int, user_id: int, category_id: int,
              player_name: str | None) -> tuple[Label, bool]:
    """Returns (label, created) - a duplicate (marker, user, category,
    player) vote is idempotent (returns the existing row, created=False)
    rather than erroring, same reasoning as videos.get_by_gcs_relpath."""
    existing = session.scalar(
        select(Label).where(
            Label.marker_id == marker_id,
            Label.user_id == user_id,
            Label.category_id == category_id,
            Label.player_name == player_name,
        )
    )
    if existing is not None:
        return existing, False
    label = Label(marker_id=marker_id, user_id=user_id, category_id=category_id, player_name=player_name)
    session.add(label)
    session.flush()
    return label, True


def list_for_marker(session: Session, marker_id: int) -> list[Label]:
    return list(session.scalars(select(Label).where(Label.marker_id == marker_id).order_by(Label.created_at)))


def get_label(session: Session, label_id: int) -> Label | None:
    return session.get(Label, label_id)


def update_label(session: Session, label_id: int, category_id: int | None,
                  player_name: str | None, player_name_given: bool) -> Label | None:
    label = session.get(Label, label_id)
    if label is None:
        return None
    if category_id is not None:
        label.category_id = category_id
    if player_name_given:
        label.player_name = player_name
    session.flush()
    return label


def delete_label(session: Session, label_id: int) -> bool:
    label = session.get(Label, label_id)
    if label is None:
        return False
    session.delete(label)
    session.flush()
    return True
