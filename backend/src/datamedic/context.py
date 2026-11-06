"""Process-wide service container shared by the API and the workers."""

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from datamedic.config import Settings
from datamedic.db.session import make_engine, make_sessionmaker
from datamedic.email import EmailSender, build_email_sender
from datamedic.jobs.publisher import CeleryJobPublisher, EagerJobPublisher, JobPublisher
from datamedic.storage import LocalFilesystemStorage, ObjectStorage


@dataclass
class AppContext:
    settings: Settings
    engine: Engine
    sessions: sessionmaker[Session]
    maintenance_engine: Engine
    maintenance_sessions: sessionmaker[Session]
    storage: ObjectStorage
    email: EmailSender
    publisher: JobPublisher | None = None

    def require_publisher(self) -> JobPublisher:
        if self.publisher is None:
            raise RuntimeError("job publisher not configured")
        return self.publisher

    def dispose(self) -> None:
        self.engine.dispose()
        self.maintenance_engine.dispose()


def build_context(settings: Settings) -> AppContext:
    engine = make_engine(settings.database_url.get_secret_value())
    maint = make_engine(settings.maintenance_database_url.get_secret_value())
    ctx = AppContext(
        settings=settings,
        engine=engine,
        sessions=make_sessionmaker(engine),
        maintenance_engine=maint,
        maintenance_sessions=make_sessionmaker(maint),
        storage=LocalFilesystemStorage(settings.storage_root),
        email=build_email_sender(settings),
    )
    if settings.jobs_eager:
        ctx.publisher = EagerJobPublisher(ctx)
    else:
        from datamedic.workers.celery_app import create_celery

        ctx.publisher = CeleryJobPublisher(create_celery(settings))
    return ctx
