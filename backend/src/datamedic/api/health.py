import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from datamedic import __version__
from datamedic.api.deps import CtxDep

log = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up. No dependency checks, no details."""
    return {"status": "ok"}


@router.get("/readyz")
def readyz(ctx: CtxDep) -> JSONResponse:
    """Readiness: database reachable and storage writable."""
    checks = {"database": False, "storage": False}
    try:
        with ctx.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:  # noqa: BLE001
        log.warning("readiness: database check failed")
    checks["storage"] = ctx.storage.check_writable()
    ok = all(checks.values())
    return JSONResponse(
        {"status": "ok" if ok else "unavailable", "checks": checks}, status_code=200 if ok else 503
    )


@router.get("/api/v1/meta/version")
def version() -> dict[str, str]:
    return {"api_version": "v1", "backend_version": __version__}
