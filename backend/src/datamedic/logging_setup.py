"""Structured logging with a scrubber that keeps dataset content out of logs.

Rule: never pass dataset values, rows, column names or file contents to a logger.
The scrubber is a safety net, not a licence to log data.
"""

import json
import logging
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# `extra=` keys that must never reach a log sink.
DENIED_KEYS = frozenset(
    {
        "value",
        "values",
        "row",
        "rows",
        "cell",
        "cells",
        "sample",
        "samples",
        "data",
        "content",
        "body",
        "columns",
        "column_names",
        "password",
        "token",
    }
)
REDACTED = "[redacted]"

_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()) | {
    "message",
    "asctime",
}


class ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key in list(record.__dict__):
            if key in DENIED_KEYS:
                record.__dict__[key] = REDACTED
        if isinstance(record.args, dict):
            record.args = {k: (REDACTED if k in DENIED_KEYS else v) for k, v in record.args.items()}
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        for key, val in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.addFilter(ScrubFilter())
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
