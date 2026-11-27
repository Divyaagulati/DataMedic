"""FastAPI application factory. Run with:
uvicorn datamedic.api.app:create_app --factory
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from datamedic import __version__
from datamedic.analyses.router import router as analyses_router
from datamedic.analytics.router import router as analytics_router
from datamedic.api import health
from datamedic.api.deps import CSRF_HEADER
from datamedic.api.middleware import NoStoreMiddleware, OriginCheckMiddleware, RequestIdMiddleware
from datamedic.auth.router import router as auth_router
from datamedic.biomedical.router import router as biomedical_router
from datamedic.config import Settings, get_settings
from datamedic.context import AppContext, build_context
from datamedic.datasets.router import router as datasets_router
from datamedic.errors import install_error_handlers
from datamedic.experiments.router import router as experiments_router
from datamedic.jobs.router import router as jobs_router
from datamedic.logging_setup import configure_logging
from datamedic.pipelines.router import router as pipelines_router
from datamedic.preprocessing.router import router as preprocessing_router
from datamedic.recommendations.router import router as recommendations_router
from datamedic.reports.router import router as reports_router
from datamedic.training.router import router as training_router
from datamedic.versions.router import router as versions_router


def create_app(settings: Settings | None = None, ctx: AppContext | None = None) -> FastAPI:
    settings = settings or (ctx.settings if ctx else get_settings())
    if ctx is None:
        configure_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owns_ctx = ctx is None
        app.state.ctx = ctx or build_context(settings)
        try:
            yield
        finally:
            if owns_ctx:
                app.state.ctx.dispose()

    app = FastAPI(
        title="DataMedic API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs" if settings.environment != "production" else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.environment != "production" else None,
    )
    if ctx is not None:  # tests inject a context without running the lifespan
        app.state.ctx = ctx

    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(auth_router)
    app.include_router(datasets_router)
    app.include_router(jobs_router)
    app.include_router(versions_router)
    app.include_router(analyses_router)
    app.include_router(preprocessing_router)
    app.include_router(recommendations_router)
    app.include_router(experiments_router)
    app.include_router(pipelines_router)
    app.include_router(training_router)
    app.include_router(biomedical_router)
    app.include_router(reports_router)
    app.include_router(analytics_router)

    # Middleware: last added runs first.
    app.add_middleware(NoStoreMiddleware)
    app.add_middleware(OriginCheckMiddleware, allowed_origin=settings.web_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.web_origin],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Content-Type", CSRF_HEADER, "Idempotency-Key"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RequestIdMiddleware)
    return app
