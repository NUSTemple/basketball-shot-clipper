"""Admin-managed label category list (docs/REQUIREMENTS_V2.md #4) - seeded
with Goal/Assist/Block in the initial migration. "Removing" a category is a
soft-delete (active=false) since existing labels FK it."""
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import LabelCategory


def list_categories(session: Session, include_inactive: bool = False) -> list[LabelCategory]:
    stmt = select(LabelCategory).order_by(LabelCategory.name)
    if not include_inactive:
        stmt = stmt.where(LabelCategory.active.is_(True))
    return list(session.scalars(stmt))


def create_category(session: Session, name: str, created_by_user_id: int) -> LabelCategory:
    category = LabelCategory(name=name, created_by=created_by_user_id)
    session.add(category)
    session.flush()
    return category


def update_category(session: Session, category_id: int, active: bool | None = None,
                     name: str | None = None) -> LabelCategory | None:
    category = session.get(LabelCategory, category_id)
    if category is None:
        return None
    if active is not None:
        category.active = active
    if name is not None:
        category.name = name
    session.flush()
    return category
