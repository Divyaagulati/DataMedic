"""Account and session logic. Users/sessions/tokens are only touched from this module."""

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.db.models import AuthToken, User, UserSession
from datamedic.email import EmailSender, OutgoingEmail
from datamedic.errors import AuthInvalid, RateLimited, ValidationFailed
from datamedic.security.passwords import (
    hash_password,
    needs_rehash,
    password_problems,
    verify_password,
)
from datamedic.security.tokens import new_token, token_digest

log = logging.getLogger(__name__)

VERIFY_TTL = timedelta(hours=24)
RESET_TTL = timedelta(minutes=30)


def _now() -> datetime:
    return datetime.now(UTC)


def _check_password_policy(password: str) -> None:
    problems = password_problems(password)
    if problems:
        raise ValidationFailed(
            "Password does not meet the policy.",
            errors=[
                {"pointer": "/password", "code": "PASSWORD_POLICY", "message": p} for p in problems
            ],
        )


def _issue_token(db: Session, user_id: uuid.UUID, purpose: str, ttl: timedelta) -> str:
    token = new_token()
    db.add(
        AuthToken(
            user_id=user_id,
            purpose=purpose,
            token_hash=token_digest(token),
            expires_at=_now() + ttl,
        )
    )
    return token


def _consume_token(db: Session, token: str, purpose: str) -> AuthToken:
    row = db.scalar(
        select(AuthToken)
        .where(AuthToken.token_hash == token_digest(token), AuthToken.purpose == purpose)
        .with_for_update()
    )
    if row is None or row.used_at is not None or row.expires_at <= _now():
        raise AuthInvalid("The link is invalid or has expired.")
    row.used_at = _now()
    return row


def signup(
    db: Session,
    settings: Settings,
    mailer: EmailSender,
    email: str,
    password: str,
    display_name: str | None,
) -> None:
    """Same observable outcome whether or not the email is registered (no enumeration)."""
    _check_password_policy(password)
    email = email.lower()
    password_hash = hash_password(password)  # always hash: equalises timing
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        db.rollback()
        return
    user = User(email=email, password_hash=password_hash, display_name=display_name)
    db.add(user)
    db.flush()
    token = _issue_token(db, user.id, "verify_email", VERIFY_TTL)
    db.commit()
    _send_verification(settings, mailer, email, token)


def _send_verification(settings: Settings, mailer: EmailSender, email: str, token: str) -> None:
    _safe_send(
        mailer,
        OutgoingEmail(
            to=email,
            subject="Verify your DataMedic account",
            body=(
                "Confirm your email address:\n"
                f"{settings.public_web_url}/verify-email?token={token}\n"
            ),
        ),
    )


def _safe_send(mailer: EmailSender, email: OutgoingEmail) -> None:
    # A mail outage must not change the API response (that would leak account existence).
    try:
        mailer.send(email)
    except Exception:  # noqa: BLE001
        log.warning("account email delivery failed")


def resend_verification(db: Session, settings: Settings, mailer: EmailSender, user: User) -> None:
    if user.email_verified_at is not None:
        return
    token = _issue_token(db, user.id, "verify_email", VERIFY_TTL)
    db.commit()
    _send_verification(settings, mailer, user.email, token)


def verify_email(db: Session, token: str) -> None:
    row = _consume_token(db, token, "verify_email")
    user = db.get(User, row.user_id)
    if user is not None and user.email_verified_at is None:
        user.email_verified_at = _now()
    db.commit()


def _login_delay_seconds(settings: Settings, failed: int) -> int:
    over = failed - settings.login_free_attempts
    if over < 0:
        return 0
    return int(min(2**over, settings.login_max_delay_seconds))


def authenticate(db: Session, settings: Settings, email: str, password: str) -> User:
    email = email.lower()
    user = db.scalar(select(User).where(User.email == email).with_for_update())
    if user is not None and user.last_failed_login_at is not None:
        delay = _login_delay_seconds(settings, user.failed_login_count)
        retry_at = user.last_failed_login_at + timedelta(seconds=delay)
        if delay and retry_at > _now():
            db.rollback()
            wait = max(1, int((retry_at - _now()).total_seconds()) + 1)
            raise RateLimited(
                "Too many failed attempts. Try again later.", headers={"Retry-After": str(wait)}
            )
    ok = verify_password(user.password_hash if user else None, password)
    if user is None or not ok or user.status != "active" or user.deleted_at is not None:
        if user is not None:
            user.failed_login_count += 1
            user.last_failed_login_at = _now()
            db.commit()
        log.info("login failed")
        raise AuthInvalid("Invalid email or password.")
    user.failed_login_count = 0
    user.last_failed_login_at = None
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    return user


def create_session(
    db: Session, settings: Settings, user: User, user_agent: str | None
) -> tuple[str, UserSession]:
    token = new_token()
    sess = UserSession(
        user_id=user.id,
        token_hash=token_digest(token),
        csrf_token=new_token(),
        expires_at=_now() + timedelta(days=settings.session_absolute_days),
        user_agent=(user_agent or "")[:300] or None,
    )
    db.add(sess)
    db.commit()
    return token, sess


def revoke_session(db: Session, user_id: uuid.UUID, session_id: uuid.UUID) -> bool:
    result = db.execute(
        update(UserSession)
        .where(
            UserSession.id == session_id,
            UserSession.user_id == user_id,
            UserSession.revoked_at.is_(None),
        )
        .values(revoked_at=_now())
    )
    db.commit()
    return bool(result.rowcount)  # type: ignore[attr-defined]


def list_sessions(db: Session, user_id: uuid.UUID) -> list[UserSession]:
    now = _now()
    return list(
        db.scalars(
            select(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
            .order_by(UserSession.created_at.desc())
        )
    )


def request_password_reset(
    db: Session, settings: Settings, mailer: EmailSender, email: str
) -> None:
    user = db.scalar(select(User).where(User.email == email.lower()))
    if user is None or user.deleted_at is not None:
        return
    token = _issue_token(db, user.id, "reset_password", RESET_TTL)
    db.commit()
    _safe_send(
        mailer,
        OutgoingEmail(
            to=user.email,
            subject="Reset your DataMedic password",
            body=f"Reset your password (valid 30 minutes):\n"
            f"{settings.public_web_url}/reset-password?token={token}\n",
        ),
    )


def confirm_password_reset(db: Session, token: str, new_password: str) -> None:
    _check_password_policy(new_password)
    row = _consume_token(db, token, "reset_password")
    user = db.get(User, row.user_id)
    if user is None:
        raise AuthInvalid("The link is invalid or has expired.")
    user.password_hash = hash_password(new_password)
    user.failed_login_count = 0
    user.last_failed_login_at = None
    db.execute(
        update(UserSession)
        .where(UserSession.user_id == user.id, UserSession.revoked_at.is_(None))
        .values(revoked_at=_now())
    )
    db.commit()
