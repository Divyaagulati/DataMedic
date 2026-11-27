"""Parse an uploaded original into the canonical version-0 table."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from datamedic.engine.errors import EngineError, limit_exceeded, parse_failed
from datamedic.engine.hashing import content_hash
from datamedic.engine.io.readers import (
    ParseLimits,
    ReadResult,
    read_csv,
    read_excel,
    read_json,
    read_parquet,
)
from datamedic.engine.preview import build_preview
from datamedic.engine.table import MISSING_TOKENS, CanonicalTable, build_canonical

FORMATS = ("csv", "xlsx", "xls", "parquet", "json")


@dataclass
class ParsedVersion:
    table: CanonicalTable
    parse_options: dict[str, Any]
    report: dict[str, Any]
    content_hash: bytes
    preview: dict[str, Any]

    @property
    def schema(self) -> list[dict[str, Any]]:
        return [c.as_dict() for c in self.table.columns]


def read_any(path: Path, fmt: str, declared_extension: str, limits: ParseLimits) -> ReadResult:
    if fmt == "csv":
        return read_csv(path, declared_extension, limits)
    if fmt in ("xlsx", "xls"):
        return read_excel(path, fmt, limits)
    if fmt == "parquet":
        return read_parquet(path, limits)
    if fmt == "json":
        return read_json(path, declared_extension, limits)
    raise EngineError("UNSUPPORTED_FORMAT", "This file format is not supported.")


def parse_original(
    path: Path, fmt: str, declared_extension: str, limits: ParseLimits | None = None
) -> ParsedVersion:
    limits = limits or ParseLimits()
    result = read_any(path, fmt, declared_extension, limits)
    if not result.raw.columns:
        raise parse_failed("The file contains no columns.")
    table, drop_stats = build_canonical(result.raw, drop_empty=result.drop_empty)
    if table.n_rows == 0:
        raise parse_failed("The file contains a header but no data rows.")
    if table.n_cols > limits.max_columns:
        raise limit_exceeded(f"The table has more than {limits.max_columns} columns.")
    if table.n_rows * table.n_cols > limits.max_cells:
        raise limit_exceeded(f"The table has more than {limits.max_cells:,} cells.")
    schema = [c.as_dict() for c in table.columns]
    report = {
        **result.report,
        **drop_stats,
        "n_rows": table.n_rows,
        "n_cols": table.n_cols,
        "missing_tokens": sorted(MISSING_TOKENS),
        "renamed_columns": sum(1 for c in table.columns if "renamed" in c.notes),
    }
    return ParsedVersion(
        table=table,
        parse_options=result.parse_options,
        report=report,
        content_hash=content_hash(table.frame),
        preview=build_preview(table.frame, schema),
    )


def write_parquet(table: CanonicalTable, path: Path) -> None:
    """Write the canonical frame. All-missing text columns are stored as strings, not nulls."""
    write_frame(table.frame, path)


def write_frame(frame: Any, path: Any) -> None:
    """Write a canonical frame as a version Parquet file (``path`` or a binary buffer). Used
    for version 0 and for committed versions, so both are stored identically."""
    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    for i, fld in enumerate(arrow.schema):
        if pa.types.is_null(fld.type):
            arrow = arrow.set_column(
                i, pa.field(fld.name, pa.string()), arrow.column(i).cast(pa.string())
            )
    pq.write_table(arrow, path)
