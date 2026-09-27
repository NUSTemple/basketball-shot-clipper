"""Engine/session setup for the v2 Postgres store (Supabase-hosted).

Plain SQLAlchemy, not Flask-SQLAlchemy: worker.py is a standalone subprocess,
not a Flask app, so a session factory that isn't bound to `app` works
identically whether it's called from a label_ui.api blueprint or from a
background job in worker.py.
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Supabase's transaction pooler (pgbouncer, port 6543) - not the direct
# connection - because Cloud Run's scale-to-zero plus two separate services
# means connections open/close in bursts, which a direct connection's much
# lower cap tolerates poorly. Required format:
# postgresql+psycopg://<user>:<password>@<host>:6543/<database>
DATABASE_URL = os.environ.get("SHOT_CLIPPER_DATABASE_URL", "")

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        if not DATABASE_URL:
            raise RuntimeError(
                "SHOT_CLIPPER_DATABASE_URL is not set - required for any v2 "
                "(Postgres-backed) route or job."
            )
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=2,
            pool_pre_ping=True,
            # pgbouncer in transaction-pooling mode gives a different backend
            # connection per statement, which breaks psycopg's server-side
            # prepared-statement cache (it would try to reuse a prepared
            # statement on a connection that never prepared it) - disabling
            # it trades a little per-statement overhead for correctness.
            connect_args={"prepare_threshold": None},
        )
    return _engine


def get_sessionmaker() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_session() -> Session:
    """`with get_session() as session:` - commits on clean exit, rolls back
    and re-raises on any exception."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
