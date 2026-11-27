"""Engine error type. Messages are user-facing and must never contain data values."""

from typing import Any

PARSE_FAILED = "PARSE_FAILED"
LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"


class EngineError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        # Structured context (e.g. step index, column name). Never data values; never logged.
        self.details = dict(details or {})


def parse_failed(message: str) -> EngineError:
    return EngineError(PARSE_FAILED, message)


def limit_exceeded(message: str) -> EngineError:
    return EngineError(LIMIT_EXCEEDED, message)
