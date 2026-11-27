"""Analytics data access (API side): one hash-verified, read-only in-memory copy of an
authorised version, the privacy policy, and a concurrency cap.

Decision A1 (Phase 8, docs/ARCHITECTURE.md "Analytics Studio"): interactive queries run in the
API process so the studio responds in well under a second, instead of queueing behind
training jobs. pandas is imported lazily here only. Memory stays bounded: versions above
``analytics_max_rows`` / ``analytics_max_cells`` are refused, at most ``analytics_cache_entries``
frames are cached (least recently used), and at most ``analytics_max_concurrency`` queries run
at once (others get 429). Expensive whole-dataset analyses (automated insights) run in the
worker instead. The cached frame is never modified: the engine filters copies.
"""

import hashlib
import io
import threading
import uuid
from collections import OrderedDict
from typing import Any

from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.db.models import Dataset, DatasetVersion
from datamedic.errors import AnalyticsBusy, DomainError, QuotaExceeded, ValidationFailed
from datamedic.storage import ObjectNotFound, ObjectStorage
from datamedic.training import privacy

_lock = threading.Lock()
_cache: "OrderedDict[tuple[uuid.UUID, bytes], Any]" = OrderedDict()
_slots: threading.BoundedSemaphore | None = None
_slots_size = 0


def slots(settings: Settings) -> threading.BoundedSemaphore:
    global _slots, _slots_size
    with _lock:
        if _slots is None or _slots_size != settings.analytics_max_concurrency:
            _slots = threading.BoundedSemaphore(max(1, settings.analytics_max_concurrency))
            _slots_size = settings.analytics_max_concurrency
        return _slots


class Slot:
    """Context manager: a concurrency slot or 429."""

    def __init__(self, settings: Settings) -> None:
        self.sem = slots(settings)

    def __enter__(self) -> None:
        if not self.sem.acquire(timeout=5):
            raise AnalyticsBusy("Too many analytics queries are running; try again in a moment.")

    def __exit__(self, *exc: object) -> None:
        self.sem.release()


def check_size(settings: Settings, version: DatasetVersion) -> None:
    if (
        version.n_rows > settings.analytics_max_rows
        or version.n_rows * max(version.n_cols, 1) > settings.analytics_max_cells
    ):
        raise QuotaExceeded(
            f"Analytics Studio supports up to {settings.analytics_max_rows:,} rows and "
            f"{settings.analytics_max_cells:,} cells per version."
        )


def frame(settings: Settings, storage: ObjectStorage, version: DatasetVersion) -> Any:
    """The version's canonical frame (cached; verified against the recorded file hash)."""
    check_size(settings, version)
    key = (version.id, bytes(version.file_sha256))
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    try:
        with storage.open_reader(version.object_key) as fh:
            raw = fh.read()
    except ObjectNotFound as exc:
        raise DomainError("Version data missing.") from exc
    if hashlib.sha256(raw).digest() != version.file_sha256:
        raise DomainError("The stored version failed its integrity check.")
    import pandas as pd

    df = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    with _lock:
        _cache[key] = df
        while len(_cache) > max(1, settings.analytics_cache_entries):
            _cache.popitem(last=False)
    return df


def policy(db: Session, owner_id: uuid.UUID, ds: Dataset) -> Any:
    from datamedic.engine.analytics.query import Policy

    protected = frozenset(privacy.protected_columns(db, owner_id, ds.id))
    return Policy(protected=protected, privacy_mode=ds.domain == "biomedical" or bool(protected))


def raise_errors(errors: list[dict[str, str]]) -> None:
    if errors:
        raise ValidationFailed("The analytical query is not valid.", errors=errors)
