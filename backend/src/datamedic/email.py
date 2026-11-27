"""Outbound account email (verification, password reset). Never carries dataset content."""

import logging
import smtplib
import ssl
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Protocol

from datamedic.config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutgoingEmail:
    to: str
    subject: str
    body: str


class EmailSender(Protocol):
    def send(self, email: OutgoingEmail) -> None: ...


class SmtpEmailSender:
    def __init__(
        self,
        host: str,
        port: int,
        sender: str,
        security: str = "none",
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        self._host, self._port, self._sender = host, port, sender
        self._security, self._username, self._password = security, username, password

    def send(self, email: OutgoingEmail) -> None:
        msg = EmailMessage()
        msg["From"] = self._sender
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg.set_content(email.body)
        context = ssl.create_default_context()
        smtp: smtplib.SMTP = (
            smtplib.SMTP_SSL(self._host, self._port, timeout=10, context=context)
            if self._security == "ssl"
            else smtplib.SMTP(self._host, self._port, timeout=10)
        )
        with smtp:
            if self._security == "starttls":
                smtp.starttls(context=context)
            if self._username:
                smtp.login(self._username, self._password or "")
            smtp.send_message(msg)


@dataclass
class MemoryEmailSender:
    """Test double: keeps messages in memory."""

    outbox: list[OutgoingEmail] = field(default_factory=list)

    def send(self, email: OutgoingEmail) -> None:
        self.outbox.append(email)


class ConsoleEmailSender:
    """Development only (refused in production by Settings): prints the message."""

    def send(self, email: OutgoingEmail) -> None:
        print(f"--- email to {email.to}: {email.subject}\n{email.body}\n---")  # noqa: T201


def build_email_sender(settings: Settings) -> EmailSender:
    if settings.email_backend == "memory":
        return MemoryEmailSender()
    if settings.email_backend == "console":
        return ConsoleEmailSender()
    return SmtpEmailSender(
        settings.smtp_host,
        settings.smtp_port,
        settings.smtp_from,
        settings.smtp_security,
        settings.smtp_username,
        settings.smtp_password.get_secret_value() if settings.smtp_password else None,
    )
