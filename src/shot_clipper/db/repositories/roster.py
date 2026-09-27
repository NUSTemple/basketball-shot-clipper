"""Global, flat player roster on Postgres (docs/REQUIREMENTS_V2.md #4) - the
v2 equivalent of the top-level roster.py's roster.json, kept separate since
that file backs the untouched legacy goal/no_goal dataset flow."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import RosterPlayer


def add_player(session: Session, name: str, created_by_user_id: int) -> tuple[RosterPlayer, bool]:
    existing = session.scalar(select(RosterPlayer).where(RosterPlayer.name == name))
    if existing is not None:
        return existing, False
    player = RosterPlayer(name=name, created_by=created_by_user_id)
    session.add(player)
    session.flush()
    return player, True


def list_players(session: Session) -> list[RosterPlayer]:
    return list(session.scalars(select(RosterPlayer).order_by(RosterPlayer.name)))
