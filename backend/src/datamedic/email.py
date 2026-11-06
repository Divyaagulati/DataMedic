"""Outbound account email (verification, password reset). Never carries dataset content."""

import logging
import smtplib
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
    def __init__(self, host: str, port: int, sender: str) -> None:
        self._host, self._port, self._sender = host, port, sender

    def send(self, email: OutgoingEmail) -> None:
        msg = EmailMessage()
        msg["From"] = self._sender
        msg["To"] = email.to
        msg["Subject"] = email.subject
        msg.set_content(email.body)
        with smtplib.SMTP(self._host, self._port, timeout=10) as smtp:
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
    return SmtpEmailSender(settings.smtp_host, settings.smtp_port, settings.smtp_from)
