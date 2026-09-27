"""Cross-video export query (docs/REQUIREMENTS_V2.md #6) and cut-clip
provenance. No per-user visibility filter anywhere here - access is fully
open (see REQUIREMENTS_V2.md #2), so a search spans every video."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Comment, CutClip, CutClipMarker, Label, LabelCategory, Marker


def find_matching_markers(session: Session, category_names: list[str] | None = None,
                           player: str | None = None, comment_keyword: str | None = None) -> list[Marker]:
    """Markers matching (category AND/OR player) AND comment_keyword, each
    combined only when actually supplied - a marker matching either half
    alone is enough when the other filter is omitted."""
    label_ids = None
    if category_names or player:
        stmt = select(Marker.id).join(Label, Label.marker_id == Marker.id)
        if category_names:
            stmt = stmt.join(LabelCategory, LabelCategory.id == Label.category_id).where(
                LabelCategory.name.in_(category_names))
        if player:
            stmt = stmt.where(Label.player_name.ilike(f"%{player}%"))
        label_ids = set(session.scalars(stmt.distinct()))

    comment_ids = None
    if comment_keyword:
        stmt = select(Marker.id).join(Comment, Comment.marker_id == Marker.id).where(
            Comment.text.ilike(f"%{comment_keyword}%")).distinct()
        comment_ids = set(session.scalars(stmt))

    if label_ids is not None and comment_ids is not None:
        marker_ids = label_ids & comment_ids
    elif label_ids is not None:
        marker_ids = label_ids
    elif comment_ids is not None:
        marker_ids = comment_ids
    else:
        marker_ids = set()

    if not marker_ids:
        return []
    return list(session.scalars(
        select(Marker).where(Marker.id.in_(marker_ids)).order_by(Marker.video_id, Marker.timestamp_s)
    ))


def record_cut(session: Session, video_id: int, gcs_relpath: str, start_s: float, end_s: float,
                created_by_user_id: int, marker_ids: list[int]) -> CutClip:
    cut = CutClip(video_id=video_id, gcs_relpath=gcs_relpath, start_s=start_s, end_s=end_s,
                  created_by_user_id=created_by_user_id)
    session.add(cut)
    session.flush()
    session.add_all(CutClipMarker(cut_clip_id=cut.id, marker_id=mid) for mid in marker_ids)
    session.flush()
    return cut


def get_cut_clip(session: Session, cut_clip_id: int) -> CutClip | None:
    return session.get(CutClip, cut_clip_id)
