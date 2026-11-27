import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from datamedic.jobs.schemas import JobOut
from datamedic.versions.schemas import VersionSummaryOut


class DatasetCreateIn(BaseModel):
    # extra="forbid": a client cannot smuggle in owner_id / user_id / status.
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class DatasetPatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    domain: str | None = Field(default=None, min_length=1, max_length=64)


class DatasetOut(BaseModel):
    id: uuid.UUID
    name: str
    domain: str
    domain_changed_at: datetime | None
    status: str
    filename: str
    size_bytes: int
    detected_format: str | None
    head_version: VersionSummaryOut | None
    created_at: datetime


class DatasetCreatedOut(BaseModel):
    dataset: DatasetOut
    max_upload_bytes: int


class DatasetListOut(BaseModel):
    items: list[DatasetOut]


class UploadAcceptedOut(BaseModel):
    dataset: DatasetOut
    job: JobOut


class ParseAcceptedOut(BaseModel):
    dataset: DatasetOut
    job: JobOut


class DeletionAcceptedOut(BaseModel):
    dataset_id: uuid.UUID
    job: JobOut
