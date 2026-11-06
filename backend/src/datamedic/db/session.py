"""Engine/session factories and the Row-Level Security hook.

Every transaction on an RLS-protected table runs with ``app.user_id`` set via
``set_config(..., is_local => true)``. If it is not set, RLS policies see NULL and
return zero rows (fail closed).
"""

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.orm import Session, SessionTransaction, sessionmaker

_SET_USER = text("SELECT set_config('app.user_id', :uid, true)")
_USER_KEY = "rls_user_id"


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def make_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@event.listens_for(Session, "after_begin")
def _apply_rls_user(session: Session, transaction: SessionTransaction, conn: Connection) -> None:
    uid = session.info.get(_USER_KEY)
    if uid is not None:
        conn.execute(_SET_USER, {"uid": str(uid)})


def bind_user(session: Session, user_id: uuid.UUID) -> None:
    """Scope this session to ``user_id`` for RLS, including the current transaction."""
    session.info[_USER_KEY] = user_id
    if session.in_transaction():
        session.execute(_SET_USER, {"uid": str(user_id)})


def bound_user(session: Session) -> uuid.UUID | None:
    value = session.info.get(_USER_KEY)
    return value if isinstance(value, uuid.UUID) else None


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
