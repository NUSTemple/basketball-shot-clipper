"""CRUD for `comments` rows - flat (no threading), pinned to a marker's
timestamp (docs/REQUIREMENTS_V2.md #5)."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Comment


def add_comment(session: Session, marker_id: int, user_id: int, text: str) -> Comment:
    comment = Comment(marker_id=marker_id, user_id=user_id, text=text)
    session.add(comment)
    session.flush()
    return comment


def list_for_marker(session: Session, marker_id: int) -> list[Comment]:
    return list(session.scalars(select(Comment).where(Comment.marker_id == marker_id).order_by(Comment.created_at)))


def get_comment(session: Session, comment_id: int) -> Comment | None:
    return session.get(Comment, comment_id)


def update_comment(session: Session, comment_id: int, text: str) -> Comment | None:
    comment = session.get(Comment, comment_id)
    if comment is None:
        return None
    comment.text = text
    session.flush()
    return comment


def delete_comment(session: Session, comment_id: int) -> bool:
    comment = session.get(Comment, comment_id)
    if comment is None:
        return False
    session.delete(comment)
    session.flush()
    return True
