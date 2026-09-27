"""Every v2 route/job identifies its caller by email (see label_ui.auth) but
every other table FKs a users.id - get_or_create_user is the one place that
gap gets closed, so no route/job needs to duplicate the upsert logic."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import User


def get_or_create_user(session: Session, email: str) -> User:
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email)
        session.add(user)
        session.flush()  # populates user.id without waiting for commit
    return user


def update_profile(session: Session, email: str, display_name: str | None, avatar_url: str | None) -> User:
    user = get_or_create_user(session, email)
    if display_name is not None:
        user.display_name = display_name
    if avatar_url is not None:
        user.avatar_url = avatar_url
    session.flush()
    return user
