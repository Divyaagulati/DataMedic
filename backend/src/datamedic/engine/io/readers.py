"""Format readers. Each returns raw, positionally aligned columns plus the parse options it
used (so a later re-parse can override them) and a small report.

Readers run inside the parse sandbox (a resource-limited subprocess). Error messages are
user-facing and never contain cell values.
"""

import codecs
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from defusedxml import DefusedXmlException

from datamedic.engine.errors import EngineError, limit_exceeded, parse_failed
from datamedic.engine.table import RawColumns, to_text

SNIFF_BYTES = 64 * 1024
CANDIDATE_DELIMITERS = ",;\t|"
JSON_SEPARATOR = "."


@dataclass(frozen=True)
class ParseLimits:
    max_columns: int = 2_000
    max_cells: int = 20_000_000
    max_json_depth: int = 10


@dataclass
class ReadResult:
    raw: RawColumns
    parse_options: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)
    drop_empty: bool = False


def _check_columns(n_cols: int, limits: ParseLimits) -> None:
    if n_cols > limits.max_columns:
        raise limit_exceeded(f"The table has more than {limits.max_columns} columns.")


def header_report(names: list[Any]) -> dict[str, Any]:
    """Row 1 is always read as the header. If most header cells are numbers, the file
    probably has no header row; this is reported (not guessed around) so the UI can say so.
    """
    cells = [s for s in (to_text(n) for n in names) if s and s.strip()]
    numeric = 0
    for cell in cells:
        try:
            float(cell.strip())
            numeric += 1
        except ValueError:
            pass
    return {"header_looks_like_data": bool(cells) and numeric * 2 >= len(cells)}


# ---- CSV / TSV --------------------------------------------------------------------------------


def detect_encoding(path: Path) -> str:
    """BOM first; otherwise strict UTF-8 over the whole file, then cp1252, then latin-1."""
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    for encoding in ("utf-8", "cp1252"):
        decoder = codecs.getincrementaldecoder(encoding)()
        try:
            with open(path, "rb") as fh:
                while chunk := fh.read(1024 * 1024):
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin-1"  # decodes any byte sequence


def detect_delimiter(path: Path, encoding: str, declared_extension: str) -> str:
    if declared_extension == "tsv":
        return "\t"
    with open(path, encoding=encoding, errors="replace", newline="") as fh:
        sample = fh.read(SNIFF_BYTES)
    if "\n" in sample:
        sample = sample[: sample.rfind("\n")]
    try:
        return csv.Sniffer().sniff(sample, delimiters=CANDIDATE_DELIMITERS).delimiter
    except csv.Error:
        return ","


_FIELDS_ERROR = re.compile(r"Expected (\d+) fields in line (\d+), saw (\d+)")


def read_csv(path: Path, declared_extension: str, limits: ParseLimits) -> ReadResult:
    encoding = detect_encoding(path)
    delimiter = detect_delimiter(path, encoding, declared_extension)
    try:
        frame = pd.read_csv(
            path,
            sep=delimiter,
            header=None,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding=encoding,
            index_col=False,
            skip_blank_lines=True,
            engine="c",
            on_bad_lines="error",
        )
    except pd.errors.EmptyDataError as exc:
        raise parse_failed("The file contains no data.") from exc
    except pd.errors.ParserError as exc:
        match = _FIELDS_ERROR.search(str(exc))
        if match:
            expected, line, saw = match.groups()
            raise parse_failed(
                f"Line {line} has {saw} fields but the header row has {expected}. "
                "Check the delimiter and quoting."
            ) from exc
        raise parse_failed("The file could not be parsed as delimited text.") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise parse_failed("The file could not be parsed as delimited text.") from exc
    _check_columns(frame.shape[1], limits)
    if len(frame) == 0:
        raise parse_failed("The file contains no data.")
    names = frame.iloc[0].tolist()
    body = frame.iloc[1:]
    raw = RawColumns(names=names, columns=[body.iloc[:, i] for i in range(body.shape[1])])
    options = {
        "format": "csv",
        "encoding": encoding,
        "delimiter": delimiter,
        "header_row": 0,
    }
    return ReadResult(raw, options, header_report(names), drop_empty=True)


# ---- Excel ------------------------------------------------------------------------------------


def read_excel(path: Path, fmt: str, limits: ParseLimits) -> ReadResult:
    engine = "openpyxl" if fmt == "xlsx" else "xlrd"
    try:
        with pd.ExcelFile(path, engine=engine) as workbook:
            sheets = [str(s) for s in workbook.sheet_names]
            if not sheets:
                raise parse_failed("The workbook contains no sheets.")
            sheet = sheets[0]
            frame = pd.read_excel(
                workbook,
                sheet_name=sheet,
                header=None,
                dtype=object,
                keep_default_na=False,
                na_filter=False,
            )
    except EngineError:
        raise
    except DefusedXmlException as exc:
        raise EngineError(
            "MALICIOUS_FILE", "The workbook contains forbidden XML constructs."
        ) from exc
    except Exception as exc:  # noqa: BLE001  (xlrd/openpyxl raise many unrelated types)
        raise parse_failed("The workbook could not be read.") from exc
    _check_columns(frame.shape[1], limits)
    if len(frame) == 0:
        raise parse_failed("The first sheet is empty.")
    names = frame.iloc[0].tolist()
    body = frame.iloc[1:]
    raw = RawColumns(names=names, columns=[body.iloc[:, i] for i in range(body.shape[1])])
    options = {"format": fmt, "sheet": sheet, "header_row": 0}
    report = {"sheets": sheets[:100], "sheet_count": len(sheets), **header_report(names)}
    return ReadResult(raw, options, report, drop_empty=True)


# ---- Parquet ----------------------------------------------------------------------------------


def _typed_parquet_column(
    series: pd.Series, arrow_type: pa.DataType | None
) -> tuple[pd.Series, list[str]]:
    notes: list[str] = []
    if isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype(object)
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean"), notes
    if pd.api.types.is_integer_dtype(series):
        return series.astype("Int64"), notes
    if pd.api.types.is_float_dtype(series):
        if arrow_type is not None and pa.types.is_integer(arrow_type):
            return series.astype("Int64"), notes  # integers with nulls arrive as floats
        return series.astype("float64"), notes
    if pd.api.types.is_datetime64_any_dtype(series):
        if getattr(series.dtype, "tz", None) is not None:
            return series.dt.tz_convert("UTC"), notes
        return series, notes
    kind = pd.api.types.infer_dtype(series, skipna=True)
    if kind in ("string", "empty"):
        return series.astype(object), notes
    if kind == "date":
        return pd.to_datetime(series), ["date converted to datetime"]
    if kind == "decimal":
        return series.astype("float64"), ["decimal converted to float"]
    if kind == "boolean":
        return series.astype("boolean"), notes
    if kind == "integer":
        return series.astype("Int64"), notes
    if kind in ("floating", "mixed-integer-float"):
        return series.astype("float64"), notes
    return series.map(to_text).astype(object), ["converted to text"]


def read_parquet(path: Path, limits: ParseLimits) -> ReadResult:
    try:
        parquet = pq.ParquetFile(path)
        meta = parquet.metadata
        _check_columns(meta.num_columns, limits)
        if meta.num_rows * max(meta.num_columns, 1) > limits.max_cells:
            raise limit_exceeded(f"The table has more than {limits.max_cells:,} cells.")
        table = parquet.read()
        arrow_types = {f.name: f.type for f in table.schema}
        frame = table.to_pandas()
    except EngineError:
        raise
    except Exception as exc:  # noqa: BLE001  (pyarrow raises ArrowInvalid, OSError, ...)
        raise parse_failed("The Parquet file could not be read.") from exc
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()  # keep a stored index as ordinary columns
    _check_columns(frame.shape[1], limits)
    columns: list[pd.Series] = []
    notes: dict[int, list[str]] = {}
    for pos in range(frame.shape[1]):
        label = frame.columns[pos]
        typed, col_notes = _typed_parquet_column(
            frame.iloc[:, pos], arrow_types.get(label) if isinstance(label, str) else None
        )
        columns.append(typed)
        if col_notes:
            notes[pos] = col_notes
    raw = RawColumns(
        names=list(frame.columns), columns=columns, typed=set(range(len(columns))), notes=notes
    )
    return ReadResult(raw, {"format": "parquet"}, {"row_groups": meta.num_row_groups})


# ---- JSON -------------------------------------------------------------------------------------


def _max_depth(value: Any, limit: int) -> int:
    """Iterative nesting depth, stopping as soon as ``limit`` is exceeded."""
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, dict):
            children: Any = item.values()
        elif isinstance(item, list):
            children = item
        else:
            continue
        deepest = max(deepest, depth)
        if deepest > limit:
            return deepest
        stack.extend((child, depth + 1) for child in children)
    return deepest


def _flatten(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested objects become dotted column names ("address.city"); lists stay as values and
    are later stored as JSON text. Depth is already bounded by the nesting limit."""
    out: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{JSON_SEPARATOR}{key}" if prefix else str(key)
        nested = _flatten(value, name) if isinstance(value, dict) and value else {name: value}
        for key, item in nested.items():
            if key in out:  # e.g. {"a.b": 1, "a": {"b": 2}}: never drop a value silently
                raise parse_failed(
                    "Two JSON fields map to the same column name after flattening nested "
                    "objects (a dotted key collides with a nested key)."
                )
            out[key] = item
    return out


def _records_from(data: Any) -> tuple[list[dict[str, Any]] | dict[str, Any], str]:
    """Return records (or a column mapping) and the orientation name."""
    if isinstance(data, list):
        if not all(isinstance(r, dict) for r in data):
            raise parse_failed("A JSON array must contain only objects (one per row).")
        return data, "records"
    if isinstance(data, dict):
        values = list(data.values())
        # An API-style wrapper such as {"data": [{...}, ...], "meta": {...}}: exactly one key
        # holds a list of objects. Checked first, so it is not read as a one-column table.
        nested = [
            k for k, v in data.items() if isinstance(v, list) and v and isinstance(v[0], dict)
        ]
        if len(nested) == 1:
            records = data[nested[0]]
            if not all(isinstance(r, dict) for r in records):
                raise parse_failed("A JSON array must contain only objects (one per row).")
            return records, "records_under_key"
        if values and all(isinstance(v, list) for v in values):
            if len({len(v) for v in values}) != 1:
                raise parse_failed("JSON columns have different lengths.")
            return data, "columns"
        if values and all(isinstance(v, dict) for v in values):
            return data, "columns_indexed"
    raise parse_failed(
        "JSON must be an array of objects, JSON Lines, or an object of equal-length columns."
    )


def read_json(path: Path, declared_extension: str, limits: ParseLimits) -> ReadResult:
    try:
        with open(path, encoding="utf-8-sig") as fh:
            if declared_extension == "jsonl":
                data: Any = [json.loads(line) for line in fh if line.strip()]
                if not all(isinstance(r, dict) for r in data):
                    raise parse_failed("Every JSON Lines row must be an object.")
            else:
                data = json.load(fh)
    except RecursionError as exc:
        raise limit_exceeded(
            f"JSON nesting is deeper than the limit ({limits.max_json_depth})."
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise parse_failed("The file is not valid UTF-8 JSON.") from exc
    if _max_depth(data, limits.max_json_depth + 2) > limits.max_json_depth + 1:
        # +1: the outer array/object holding the rows is not row nesting.
        raise limit_exceeded(f"JSON nesting is deeper than the limit ({limits.max_json_depth}).")

    records, orient = _records_from(data)
    # dtype=object keeps raw JSON values (an integer column with nulls stays integers);
    # typing happens later, exactly as for CSV.
    if orient in ("records", "records_under_key"):
        assert isinstance(records, list)
        if not records:
            raise parse_failed("The file contains no data.")
        frame = pd.DataFrame([_flatten(r) for r in records], dtype=object)
    else:
        frame = pd.DataFrame(records, dtype=object)
        if orient == "columns_indexed":
            frame = frame.reset_index(drop=True)
    _check_columns(frame.shape[1], limits)
    if len(frame) == 0 or frame.shape[1] == 0:
        raise parse_failed("The file contains no data.")
    raw = RawColumns(
        names=list(frame.columns), columns=[frame.iloc[:, i] for i in range(frame.shape[1])]
    )
    options = {
        "format": "json",
        "json_orient": "lines" if declared_extension == "jsonl" else orient,
        "nested_separator": JSON_SEPARATOR,
    }
    return ReadResult(raw, options)
