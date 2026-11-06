import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from datamedic.jobs.schemas import JobOut


class DatasetCreateIn(BaseModel):
    # extra="forbid": a client cannot smuggle in owner_id / user_id / status.
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(gt=0)


class DatasetOut(BaseModel):
    id: uuid.UUID
    name: str
    domain: str
    status: str
    filename: str
    size_bytes: int
    detected_format: str | None
    created_at: datetime


class DatasetCreatedOut(BaseModel):
    dataset: DatasetOut
    max_upload_bytes: int


class DatasetListOut(BaseModel):
    items: list[DatasetOut]


class UploadAcceptedOut(BaseModel):
    dataset: DatasetOut
    job: JobOut


class DeletionAcceptedOut(BaseModel):
    dataset_id: uuid.UUID
    job: JobOut
