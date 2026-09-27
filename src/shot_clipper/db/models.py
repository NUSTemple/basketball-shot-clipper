"""SQLAlchemy models for the v2 relational store.

Video files stay on GCS exactly as today (see label_ui/app.py's
DATA_ROOT/gcsfuse helpers) - only structured metadata lives here: users,
reusable calibration profiles, per-video markers, per-user-attributed
labels/comments on those markers, the admin-managed label category list,
the global player roster, and cut-clip provenance.

`dataset_labels.py`/`roster.py` (top-level, JSON-backed) are untouched by
this - they back the separate goal/no_goal + stars ML training dataset,
not this platform's markers/labels.
"""
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str | None] = mapped_column(String)
    avatar_url: Mapped[str | None] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # Admin-ness is NOT a column here - it stays env-based (ADMIN_EMAILS /
    # require_admin() in app.py), matching today's behavior and avoiding a
    # "who's allowed to grant admin" bootstrap problem this redesign doesn't
    # need to solve.


class CalibrationProfile(Base):
    """A reusable per-camera/court hoop calibration - selectable by anyone
    (visibility is fully open, same as videos). Replaces per-video-filename
    calibration (data/configs/<video>.json); detect_shots.load_config()'s
    JSON schema is unchanged, a profile row is just materialized into that
    same shape at job-dispatch time (see label_ui/jobs.py's detect_markers
    kind)."""

    __tablename__ = "calibration_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    hoop_bbox_norm: Mapped[list] = mapped_column(JSONB, nullable=False)  # [x1, y1, x2, y2], 0..1
    frame_width: Mapped[int | None]
    frame_height: Mapped[int | None]
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    gcs_relpath: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    original_filename: Mapped[str] = mapped_column(String, nullable=False)
    calibration_profile_id: Mapped[int | None] = mapped_column(ForeignKey("calibration_profiles.id"))
    duration_s: Mapped[float | None] = mapped_column(Numeric)
    # uploaded -> detecting -> ready, or -> error
    status: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    detect_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    detected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    markers: Mapped[list["Marker"]] = relationship(back_populates="video", cascade="all, delete-orphan")


class Marker(Base):
    """A single timestamp of interest on a video's timeline - either an
    unconfirmed suggestion from the YOLO detector (source='auto') or one a
    user placed by hand (source='manual'). Holds no category/label itself;
    labels are separate rows so multiple users can independently label the
    same marker without overwriting each other (see Label below)."""

    __tablename__ = "markers"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp_s: Mapped[float] = mapped_column(Numeric, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)  # 'auto' | 'manual'
    state: Mapped[str] = mapped_column(String, nullable=False, default="unconfirmed")  # unconfirmed|confirmed|dismissed
    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    confirmed_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped["Video"] = relationship(back_populates="markers")
    labels: Mapped[list["Label"]] = relationship(back_populates="marker", cascade="all, delete-orphan")
    comments: Mapped[list["Comment"]] = relationship(back_populates="marker", cascade="all, delete-orphan")


class LabelCategory(Base):
    """Admin-managed lookup table (start seeded with Goal/Assist/Block) -
    replaces dataset_labels.VALID_LABELS' hardcoded {"goal","no_goal"} for
    the v2 marker/label model only; the legacy dataset_labels.py enum is
    untouched. 'Removing' a category is a soft-delete (active=false) since
    existing labels FK it and shouldn't become orphaned."""

    __tablename__ = "label_categories"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RosterPlayer(Base):
    """Global, flat player list (no teams) - the v2 equivalent of
    roster.py's roster.json, but shared via Postgres instead of a per-user
    JSON file."""

    __tablename__ = "roster_players"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Label(Base):
    """One user's tag of one category (+ optional player) on one marker.
    Never overwritten by another user's label on the same marker - a
    marker's "3x Goal, 1x Block" view is a GROUP BY over these rows at
    query time, not a stored value. The unique constraint only blocks the
    same user re-submitting the identical (category, player) vote twice."""

    __tablename__ = "labels"
    __table_args__ = (
        UniqueConstraint("marker_id", "user_id", "category_id", "player_name", name="uq_label_vote"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    marker_id: Mapped[int] = mapped_column(ForeignKey("markers.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    category_id: Mapped[int] = mapped_column(ForeignKey("label_categories.id"), nullable=False, index=True)
    player_name: Mapped[str | None] = mapped_column(String, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    marker: Mapped["Marker"] = relationship(back_populates="labels")


class Comment(Base):
    """A flat (non-threaded) comment pinned to a marker's timestamp."""

    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True)
    marker_id: Mapped[int] = mapped_column(ForeignKey("markers.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())

    marker: Mapped["Marker"] = relationship(back_populates="comments")


class CutClip(Base):
    """A real, ffmpeg-cut clip file produced by an explicit, on-demand
    export/cut action - never automatic. Lands under the source video
    owner's clip dir (not the requester who triggered the cut), since it
    becomes part of that video's shared clip library."""

    __tablename__ = "cut_clips"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    gcs_relpath: Mapped[str] = mapped_column(String, nullable=False)
    start_s: Mapped[float] = mapped_column(Numeric, nullable=False)
    end_s: Mapped[float] = mapped_column(Numeric, nullable=False)
    created_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CutClipMarker(Base):
    """Many-to-many: a merged cut (two markers closer together than
    pre+post padding) spans more than one marker - see
    clip_shots.cluster_timestamps usage in the cut_markers job."""

    __tablename__ = "cut_clip_markers"

    cut_clip_id: Mapped[int] = mapped_column(ForeignKey("cut_clips.id", ondelete="CASCADE"), primary_key=True)
    marker_id: Mapped[int] = mapped_column(ForeignKey("markers.id", ondelete="CASCADE"), primary_key=True)
