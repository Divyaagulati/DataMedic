"""Domain errors and RFC 9457 problem+json rendering (ARCHITECTURE §13)."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from datamedic.logging_setup import request_id_var

log = logging.getLogger(__name__)

PROBLEM_TYPE_BASE = "https://docs.datamedic.app/errors/"


class DomainError(Exception):
    """An expected, user-facing error with a stable code."""

    status: int = 400
    code: str = "BAD_REQUEST"
    title: str = "Bad request"

    def __init__(
        self,
        detail: str | None = None,
        *,
        headers: dict[str, str] | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(detail or self.title)
        self.detail = detail
        self.headers = headers
        self.errors = errors


def _err(name: str, status: int, code: str, title: str) -> type[DomainError]:
    return type(name, (DomainError,), {"status": status, "code": code, "title": title})


AuthRequired = _err("AuthRequired", 401, "AUTH_REQUIRED", "Authentication required")
AuthInvalid = _err("AuthInvalid", 401, "AUTH_INVALID", "Invalid credentials or token")
CsrfFailed = _err("CsrfFailed", 403, "CSRF_FAILED", "CSRF validation failed")
Forbidden = _err("Forbidden", 403, "FORBIDDEN", "Action not allowed")
NotFound = _err("NotFound", 404, "NOT_FOUND", "Resource not found")
AnalyticsBusy = _err("AnalyticsBusy", 429, "ANALYTICS_BUSY", "Analytics is busy")
Conflict = _err("Conflict", 409, "CONFLICT", "Conflicting state")
QuotaExceeded = _err("QuotaExceeded", 409, "QUOTA_EXCEEDED", "Quota exceeded")
DatasetNotReady = _err("DatasetNotReady", 409, "DATASET_NOT_READY", "Dataset is not ready")
LengthRequired = _err("LengthRequired", 411, "LENGTH_REQUIRED", "Content-Length required")
UploadTooLarge = _err("UploadTooLarge", 413, "UPLOAD_TOO_LARGE", "Upload too large")
UnsupportedFormat = _err("UnsupportedFormat", 415, "UNSUPPORTED_FORMAT", "Unsupported format")
ValidationFailed = _err("ValidationFailed", 422, "VALIDATION_FAILED", "Validation failed")
RateLimited = _err("RateLimited", 429, "RATE_LIMITED", "Too many attempts")
JobLimitExceeded = _err("JobLimitExceeded", 429, "JOB_LIMIT_EXCEEDED", "Too many active jobs")
ApprovalRequired = _err("ApprovalRequired", 409, "APPROVAL_REQUIRED", "Approval required")
ApprovalStale = _err("ApprovalStale", 409, "APPROVAL_STALE", "Approval is out of date")
EvaluationBlocked = _err(
    "EvaluationBlocked", 409, "EVALUATION_BLOCKED", "Leakage-controlled evaluation is not possible"
)

# Error codes recorded on failed jobs (not raised over HTTP).
FORMAT_MISMATCH = "FORMAT_MISMATCH"
MALICIOUS_FILE = "MALICIOUS_FILE"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
INTEGRITY_FAILED = "INTEGRITY_FAILED"
LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
JOB_CANCELLED = "JOB_CANCELLED"
JOB_STALE = "JOB_STALE"
INTERNAL = "INTERNAL"


def problem(
    request: Request,
    status: int,
    code: str,
    title: str,
    detail: str | None = None,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "type": PROBLEM_TYPE_BASE + code.lower().replace("_", "-"),
        "title": title,
        "status": status,
        "code": code,
        "instance": request.url.path,
        "request_id": request_id_var.get(),
    }
    if detail:
        body["detail"] = detail
    if errors:
        body["errors"] = errors
    return JSONResponse(
        body, status_code=status, media_type="application/problem+json", headers=headers
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def _domain(request: Request, exc: DomainError) -> JSONResponse:
        return problem(
            request, exc.status, exc.code, exc.title, exc.detail, exc.errors, exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "pointer": "/" + "/".join(str(p) for p in err.get("loc", ())[1:]),
                "code": str(err.get("type", "invalid")).upper(),
                "message": err.get("msg"),
            }
            for err in exc.errors()
        ]
        return problem(request, 422, "VALIDATION_FAILED", "Validation failed", errors=errors)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return problem(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path})
        return problem(request, 500, INTERNAL, "Internal error")
