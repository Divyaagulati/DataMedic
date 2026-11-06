import uuid

from fastapi import APIRouter, Request, Response, status

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.auth import service
from datamedic.auth.schemas import (
    AcceptedOut,
    AuthStateOut,
    CsrfOut,
    LoginIn,
    PasswordResetConfirmIn,
    PasswordResetRequestIn,
    SessionListOut,
    SessionOut,
    SignupIn,
    TokenIn,
    UserOut,
)
from datamedic.config import Settings
from datamedic.db.models import User
from datamedic.errors import NotFound

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        email_verified=user.email_verified_at is not None,
        plan=user.plan,
    )


def _set_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(
        settings.session_cookie_name,
        token,
        max_age=settings.session_absolute_days * 86400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
        path="/",
    )


@router.post("/signup", status_code=status.HTTP_202_ACCEPTED)
def signup(body: SignupIn, db: DbDep, ctx: CtxDep) -> AcceptedOut:
    service.signup(db, ctx.settings, ctx.email, body.email, body.password, body.display_name)
    return AcceptedOut()


@router.post("/verify-email", status_code=status.HTTP_204_NO_CONTENT)
def verify_email(body: TokenIn, db: DbDep) -> None:
    service.verify_email(db, body.token)


@router.post("/verify-email/resend", status_code=status.HTTP_202_ACCEPTED)
def resend_verification(principal: PrincipalDep, db: DbDep, ctx: CtxDep) -> AcceptedOut:
    service.resend_verification(db, ctx.settings, ctx.email, principal.user)
    return AcceptedOut()


@router.post("/login")
def login(
    body: LoginIn, request: Request, response: Response, db: DbDep, ctx: CtxDep
) -> AuthStateOut:
    user = service.authenticate(db, ctx.settings, body.email, body.password)
    token, sess = service.create_session(db, ctx.settings, user, request.headers.get("user-agent"))
    _set_cookie(response, ctx.settings, token)
    return AuthStateOut(user=_user_out(user), csrf_token=sess.csrf_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(principal: PrincipalDep, response: Response, db: DbDep, ctx: CtxDep) -> None:
    service.revoke_session(db, principal.user.id, principal.session.id)
    response.delete_cookie(
        ctx.settings.session_cookie_name,
        path="/",
        secure=ctx.settings.cookie_secure,
        httponly=True,
        samesite="lax",
    )


@router.get("/me")
def me(principal: PrincipalDep) -> AuthStateOut:
    return AuthStateOut(user=_user_out(principal.user), csrf_token=principal.session.csrf_token)


@router.get("/csrf")
def csrf(principal: PrincipalDep) -> CsrfOut:
    return CsrfOut(csrf_token=principal.session.csrf_token)


@router.get("/sessions")
def sessions(principal: PrincipalDep, db: DbDep) -> SessionListOut:
    rows = service.list_sessions(db, principal.user.id)
    return SessionListOut(
        items=[
            SessionOut(
                id=s.id,
                created_at=s.created_at,
                last_seen_at=s.last_seen_at,
                user_agent=s.user_agent,
                current=s.id == principal.session.id,
            )
            for s in rows
        ]
    )


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke(session_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> None:
    if not service.revoke_session(db, principal.user.id, session_id):
        raise NotFound()


@router.post("/password-reset/request", status_code=status.HTTP_202_ACCEPTED)
def reset_request(body: PasswordResetRequestIn, db: DbDep, ctx: CtxDep) -> AcceptedOut:
    service.request_password_reset(db, ctx.settings, ctx.email, body.email)
    return AcceptedOut()


@router.post("/password-reset/confirm", status_code=status.HTTP_204_NO_CONTENT)
def reset_confirm(body: PasswordResetConfirmIn, db: DbDep) -> None:
    service.confirm_password_reset(db, body.token, body.new_password)
