"""Response models for versions, previews and profiles (they drive the generated TS client)."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from datamedic.db.models import DatasetVersion, VersionProfile
from datamedic.jobs.schemas import JobOut

SemanticType = Literal[
    "numeric", "boolean", "datetime", "categorical", "text", "identifier", "empty"
]
Cell = bool | int | float | str | None


class VersionSummaryOut(BaseModel):
    id: uuid.UUID
    version_number: int
    kind: str
    n_rows: int
    n_cols: int
    created_at: datetime
    # Lineage (Part E): ingest | transform | revert, the parent, the version a revert
    # restored, the transformation run of a transform, and full-data fitting.
    parent_version_id: uuid.UUID | None = None
    restored_from_version_id: uuid.UUID | None = None
    transformation_run_id: uuid.UUID | None = None
    fitted_on_full_data: bool = False
    has_model_spec: bool = False

    @classmethod
    def from_version(cls, v: DatasetVersion) -> "VersionSummaryOut":
        return cls(
            id=v.id,
            version_number=v.version_number,
            kind=v.kind,
            n_rows=v.n_rows,
            n_cols=v.n_cols,
            created_at=v.created_at,
            parent_version_id=v.parent_version_id,
            restored_from_version_id=v.restored_from_version_id,
            transformation_run_id=v.transformation_run_id,
            fitted_on_full_data=v.fitted_on_full_data,
            has_model_spec=bool(v.model_pipeline_spec),
        )


class ColumnSchemaOut(BaseModel):
    name: str
    original_name: str
    dtype: str
    inferred_type: SemanticType
    n_missing: int
    n_unique: int
    notes: list[str]


class VersionOut(VersionSummaryOut):
    dataset_id: uuid.UUID
    is_head: bool
    model_pipeline_spec: list[dict[str, Any]] | None = None
    parse_options: dict[str, Any] | None
    parse_report: dict[str, Any] | None
    columns: list[ColumnSchemaOut]
    size_bytes: int
    file_sha256: str
    content_hash: str
    engine_version: str
    library_versions: dict[str, Any]

    @classmethod
    def build(cls, v: DatasetVersion, is_head: bool) -> "VersionOut":
        return cls(
            **VersionSummaryOut.from_version(v).model_dump(),
            dataset_id=v.dataset_id,
            is_head=is_head,
            model_pipeline_spec=v.model_pipeline_spec,
            parse_options=v.parse_options,
            parse_report=v.parse_report,
            columns=[ColumnSchemaOut.model_validate(c) for c in v.schema],
            size_bytes=v.size_bytes,
            file_sha256=v.file_sha256.hex(),
            content_hash=v.content_hash.hex(),
            engine_version=v.engine_version,
            library_versions=v.library_versions,
        )


class VersionListOut(BaseModel):
    items: list[VersionSummaryOut]


class PreviewColumnOut(BaseModel):
    name: str
    dtype: str
    inferred_type: SemanticType


class PreviewOut(BaseModel):
    version_id: uuid.UUID
    columns: list[PreviewColumnOut]
    rows: list[list[Cell]]
    n_rows_total: int
    truncated: bool


# ---- Profile --------------------------------------------------------------------------------


class FrequencyValueOut(BaseModel):
    value: Cell
    count: int
    pct: float


class FrequenciesOut(BaseModel):
    values: list[FrequencyValueOut]
    complete: bool
    other_count: int
    reason: str | None = None


class HistogramOut(BaseModel):
    kind: Literal["numeric", "datetime"]
    edges: list[float | str | None]
    counts: list[int]


class NumericStatsOut(BaseModel):
    count: int
    n_infinite: int
    n_zeros: int
    n_negative: int
    mean: float | None
    std: float | None
    min: float | None
    q1: float | None
    median: float | None
    q3: float | None
    max: float | None
    iqr: float | None
    skewness: float | None
    kurtosis: float | None
    skewness_note: str | None


class DatetimeStatsOut(BaseModel):
    min: str | None
    max: str | None


class TextStatsOut(BaseModel):
    min_length: int | None
    mean_length: float | None
    max_length: int | None


class ColumnProfileOut(BaseModel):
    name: str
    inferred_type: SemanticType
    dtype: str
    n_missing: int
    missing_pct: float
    n_present: int
    n_unique: int
    unique_pct: float
    is_empty: bool
    is_constant: bool
    top_value_pct: float | None
    numeric: NumericStatsOut | None
    datetime: DatetimeStatsOut | None
    text: TextStatsOut | None
    histogram: HistogramOut | None
    frequencies: FrequenciesOut | None


class ProfileSummaryOut(BaseModel):
    n_rows: int
    n_cols: int
    n_cells: int
    missing_cells: int
    missing_cells_pct: float
    rows_with_missing: int
    rows_with_missing_pct: float
    duplicate_rows: int
    duplicate_rows_pct: float
    rows_in_duplicate_groups: int
    constant_columns: list[str]
    empty_columns: list[str]
    type_counts: dict[str, int]


class CorrelationPairOut(BaseModel):
    a: str
    b: str
    pearson: float
    spearman: float | None


class CorrelationsOut(BaseModel):
    status: Literal["ok", "not_run"]
    reason: str | None
    columns: list[str]
    pearson: list[list[float | None]]
    spearman: list[list[float | None]]
    top_pairs: list[CorrelationPairOut]
    n_rows_used: int
    sampled: bool
    seed: int
    truncated_columns: int


class SamplingOut(BaseModel):
    statistics: str
    correlations: str
    seed: int


class ProfileOut(BaseModel):
    id: uuid.UUID
    version_id: uuid.UUID
    job_id: uuid.UUID
    created_at: datetime
    profiler: str
    profiler_version: str
    engine_version: str
    library_versions: dict[str, Any]
    summary: ProfileSummaryOut
    columns: list[ColumnProfileOut]
    correlations: CorrelationsOut
    sampling: SamplingOut
    limitations: list[str]

    @classmethod
    def from_record(cls, p: VersionProfile) -> "ProfileOut":
        return cls(
            id=p.id,
            version_id=p.dataset_version_id,
            job_id=p.job_id,
            created_at=p.created_at,
            profiler=p.profiler,
            profiler_version=p.profiler_version,
            engine_version=p.engine_version,
            library_versions=p.library_versions,
            summary=ProfileSummaryOut.model_validate(p.summary),
            columns=[ColumnProfileOut.model_validate(c) for c in p.columns],
            correlations=CorrelationsOut.model_validate(p.correlations),
            sampling=SamplingOut.model_validate(p.sampling),
            limitations=p.limitations,
        )


class ProfileStateOut(BaseModel):
    """The latest stored profile (if any) and the latest profiling job (for polling)."""

    profile: ProfileOut | None
    job: JobOut | None


class ProfileAcceptedOut(BaseModel):
    job: JobOut
