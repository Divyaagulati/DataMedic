"""FastAPI dependencies: DB session, current user (RLS-bound), CSRF."""

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.context import AppContext
from datamedic.db.models import User, UserSession
from datamedic.db.session import bind_user
from datamedic.errors import AuthRequired, CsrfFailed, Forbidden
from datamedic.security.tokens import constant_time_equals, token_digest

CSRF_HEADER = "X-CSRF-Token"
_LAST_SEEN_RESOLUTION = timedelta(minutes=5)


def get_ctx(request: Request) -> AppContext:
    ctx: AppContext = request.app.state.ctx
    return ctx


def get_settings_dep(ctx: Annotated[AppContext, Depends(get_ctx)]) -> Settings:
    return ctx.settings


def get_db(ctx: Annotated[AppContext, Depends(get_ctx)]) -> Iterator[Session]:
    """One session per request. Services commit explicitly; anything uncommitted is rolled back."""
    db = ctx.sessions()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


DbDep = Annotated[Session, Depends(get_db)]
CtxDep = Annotated[AppContext, Depends(get_ctx)]


@dataclass(frozen=True)
class Principal:
    user: User
    session: UserSession


def _load_principal(request: Request, db: Session, ctx: AppContext) -> Principal | None:
    token = request.cookies.get(ctx.settings.session_cookie_name)
    if not token:
        return None
    sess = db.scalar(select(UserSession).where(UserSession.token_hash == token_digest(token)))
    now = datetime.now(UTC)
    idle_limit = timedelta(days=ctx.settings.session_idle_days)
    if (
        sess is None
        or sess.revoked_at is not None
        or sess.expires_at <= now
        or sess.last_seen_at + idle_limit <= now
    ):
        return None
    user = db.get(User, sess.user_id)
    if user is None or user.deleted_at is not None or user.status != "active":
        return None
    if now - sess.last_seen_at > _LAST_SEEN_RESOLUTION:
        sess.last_seen_at = now
        db.commit()
    # Identity comes only from the server-side session, never from client-supplied IDs.
    bind_user(db, user.id)
    return Principal(user=user, session=sess)


def current_principal(request: Request, db: DbDep, ctx: CtxDep) -> Principal:
    principal = _load_principal(request, db, ctx)
    if principal is None:
        raise AuthRequired()
    if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
        supplied = request.headers.get(CSRF_HEADER, "")
        if not supplied or not constant_time_equals(supplied, principal.session.csrf_token):
            raise CsrfFailed("Missing or invalid X-CSRF-Token header.")
    return principal


def verified_principal(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if principal.user.email_verified_at is None:
        raise Forbidden("Verify your email address before uploading data.")
    return principal


PrincipalDep = Annotated[Principal, Depends(current_principal)]
VerifiedPrincipalDep = Annotated[Principal, Depends(verified_principal)]
