"""CRUD for `games` - the primary organizing unit for the v2 workflow
(docs/REQUIREMENTS_V2.md #8). A game tags which subset of the global
roster_players list participated; it doesn't fork a separate roster."""
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Game, GamePlayer, RosterPlayer


def create_game(session: Session, created_by_user_id: int, location: str, game_date: datetime,
                 name: str | None = None) -> Game:
    game = Game(name=name, location=location, game_date=game_date, created_by_user_id=created_by_user_id)
    session.add(game)
    session.flush()
    return game


def get_game(session: Session, game_id: int) -> Game | None:
    return session.get(Game, game_id)


def list_games(session: Session) -> list[Game]:
    return list(session.scalars(select(Game).order_by(Game.game_date.desc())))


def add_player_to_game(session: Session, game_id: int, roster_player_id: int) -> tuple[GamePlayer, bool]:
    existing = session.get(GamePlayer, (game_id, roster_player_id))
    if existing is not None:
        return existing, False
    link = GamePlayer(game_id=game_id, roster_player_id=roster_player_id)
    session.add(link)
    session.flush()
    return link, True


def remove_player_from_game(session: Session, game_id: int, roster_player_id: int) -> bool:
    link = session.get(GamePlayer, (game_id, roster_player_id))
    if link is None:
        return False
    session.delete(link)
    session.flush()
    return True


def list_game_players(session: Session, game_id: int) -> list[RosterPlayer]:
    return list(
        session.scalars(
            select(RosterPlayer)
            .join(GamePlayer, GamePlayer.roster_player_id == RosterPlayer.id)
            .where(GamePlayer.game_id == game_id)
            .order_by(RosterPlayer.name)
        )
    )
