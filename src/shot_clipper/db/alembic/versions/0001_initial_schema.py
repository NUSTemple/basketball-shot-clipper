"""v2 initial schema: users, calibration profiles, videos, markers, labels,
comments, label categories, roster, cut clips.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-27

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False, unique=True),
        sa.Column("display_name", sa.String()),
        sa.Column("avatar_url", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "calibration_profiles",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("hoop_bbox_norm", postgresql.JSONB(), nullable=False),
        sa.Column("frame_width", sa.Integer()),
        sa.Column("frame_height", sa.Integer()),
        sa.Column("created_by", sa.BigInteger(), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "videos",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("owner_user_id", sa.BigInteger(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("gcs_relpath", sa.String(), nullable=False, unique=True),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("calibration_profile_id", sa.BigInteger(), sa.ForeignKey("calibration_profiles.id")),
        sa.Column("duration_s", sa.Numeric()),
        sa.Column("status", sa.String(), nullable=False, server_default="uploaded"),
        sa.Column("detect_error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("detected_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "markers",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("video_id", sa.BigInteger(), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("timestamp_s", sa.Numeric(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False, server_default="unconfirmed"),
        sa.Column("created_by_user_id", sa.BigInteger(), sa.ForeignKey("users.id")),
        sa.Column("confirmed_by_user_id", sa.BigInteger(), sa.ForeignKey("users.id")),
        sa.Column("confirmed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_markers_video_id", "markers", ["video_id"])
    op.create_index("ix_markers_video_id_state", "markers", ["video_id", "state"])

    op.create_table(
        "label_categories",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_by", sa.BigInteger(), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "roster_players",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, unique=True),
        sa.Column("created_by", sa.BigInteger(), sa.ForeignKey("users.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "labels",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("marker_id", sa.BigInteger(), sa.ForeignKey("markers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("category_id", sa.BigInteger(), sa.ForeignKey("label_categories.id"), nullable=False),
        sa.Column("player_name", sa.String()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("marker_id", "user_id", "category_id", "player_name", name="uq_label_vote"),
    )
    op.create_index("ix_labels_marker_id", "labels", ["marker_id"])
    op.create_index("ix_labels_category_id", "labels", ["category_id"])
    op.create_index("ix_labels_player_name", "labels", ["player_name"])

    op.create_table(
        "comments",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("marker_id", sa.BigInteger(), sa.ForeignKey("markers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.BigInteger(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_comments_marker_id", "comments", ["marker_id"])

    op.create_table(
        "cut_clips",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("video_id", sa.BigInteger(), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("gcs_relpath", sa.String(), nullable=False),
        sa.Column("start_s", sa.Numeric(), nullable=False),
        sa.Column("end_s", sa.Numeric(), nullable=False),
        sa.Column("created_by_user_id", sa.BigInteger(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "cut_clip_markers",
        sa.Column("cut_clip_id", sa.BigInteger(), sa.ForeignKey("cut_clips.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("marker_id", sa.BigInteger(), sa.ForeignKey("markers.id", ondelete="CASCADE"), primary_key=True),
    )

    # Seed the admin-managed label category list per docs/REQUIREMENTS_V2.md #4
    op.bulk_insert(
        sa.table(
            "label_categories",
            sa.column("name", sa.String()),
            sa.column("active", sa.Boolean()),
        ),
        [
            {"name": "Goal", "active": True},
            {"name": "Assist", "active": True},
            {"name": "Block", "active": True},
        ],
    )


def downgrade() -> None:
    op.drop_table("cut_clip_markers")
    op.drop_table("cut_clips")
    op.drop_index("ix_comments_marker_id", table_name="comments")
    op.drop_table("comments")
    op.drop_index("ix_labels_player_name", table_name="labels")
    op.drop_index("ix_labels_category_id", table_name="labels")
    op.drop_index("ix_labels_marker_id", table_name="labels")
    op.drop_table("labels")
    op.drop_table("roster_players")
    op.drop_table("label_categories")
    op.drop_index("ix_markers_video_id_state", table_name="markers")
    op.drop_index("ix_markers_video_id", table_name="markers")
    op.drop_table("markers")
    op.drop_table("videos")
    op.drop_table("calibration_profiles")
    op.drop_table("users")
