"""Object key scheme: ``{area}/{owner_id}/{dataset_id}/{name}``. Keys never contain filenames."""

import re
import uuid

AREAS = ("raw", "versions", "artifacts", "exports")
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
KEY_RE = re.compile(rf"^({'|'.join(AREAS)})/{_UUID}/{_UUID}/[a-z0-9][a-z0-9._-]{{0,127}}$")
PREFIX_RE = re.compile(rf"^({'|'.join(AREAS)})/{_UUID}/({_UUID}/)?$")


def raw_object_key(owner_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    return f"raw/{owner_id}/{dataset_id}/{uuid.uuid4()}"


def dataset_prefix(area: str, owner_id: uuid.UUID, dataset_id: uuid.UUID) -> str:
    if area not in AREAS:
        raise ValueError(f"unknown storage area {area!r}")
    return f"{area}/{owner_id}/{dataset_id}/"


def version_object_key(owner_id: uuid.UUID, dataset_id: uuid.UUID, version_id: uuid.UUID) -> str:
    return f"versions/{owner_id}/{dataset_id}/{version_id}.parquet"


def artifact_object_key(
    owner_id: uuid.UUID, dataset_id: uuid.UUID, artifact_id: uuid.UUID, ext: str
) -> str:
    return f"artifacts/{owner_id}/{dataset_id}/{artifact_id}.{ext}"
