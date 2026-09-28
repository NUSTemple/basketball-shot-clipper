"""Games: a real-world game/session that groups videos - every video now
belongs to exactly one game, and a game tags which subset of the global
roster participated. Pre-existing videos are backfilled into a placeholder
game rather than deleted (see docs/REQUIREMENTS_V2.md #8).

Revision ID: 0002_games
Revises: 0001_initial_schema
Create Date: 2026-09-28

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_games"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "games",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String()),
        sa.Column("location", sa.String(), nullable=False),
        sa.Column("game_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_user_id", sa.BigInteger(), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "game_players",
        sa.Column("game_id", sa.BigInteger(), sa.ForeignKey("games.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("roster_player_id", sa.BigInteger(), sa.ForeignKey("roster_players.id", ondelete="CASCADE"),
                  primary_key=True),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.add_column("videos", sa.Column("game_id", sa.BigInteger(), sa.ForeignKey("games.id"), nullable=True))

    # Backfill any pre-existing videos into one placeholder game rather than
    # deleting rows in a schema migration - trivially reversible, unlike a
    # DDL-script delete. Safe to manually clean up the placeholder + its
    # videos afterward if a clean slate is wanted; that's a data decision,
    # not a migration-safety one.
    games = sa.table(
        "games", sa.column("id", sa.BigInteger()), sa.column("name", sa.String()),
        sa.column("location", sa.String()), sa.column("game_date", sa.DateTime(timezone=True)),
    )
    conn = op.get_bind()
    existing_video_count = conn.execute(sa.text("SELECT count(*) FROM videos WHERE game_id IS NULL")).scalar()
    if existing_video_count:
        placeholder_id = conn.execute(
            sa.insert(games).values(name="Pre-Games Import", location="Unknown", game_date=sa.func.now())
            .returning(games.c.id)
        ).scalar()
        conn.execute(sa.text("UPDATE videos SET game_id = :gid WHERE game_id IS NULL"), {"gid": placeholder_id})

    op.alter_column("videos", "game_id", nullable=False)


def downgrade() -> None:
    op.drop_column("videos", "game_id")
    op.drop_table("game_players")
    op.drop_table("games")
