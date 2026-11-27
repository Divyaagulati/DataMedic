"""Parsing of every supported format into the canonical version-0 table, plus malformed,
empty and hostile inputs. All fixtures are synthetic."""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import LIMIT_EXCEEDED, PARSE_FAILED, EngineError
from datamedic.engine.io.parse import ParsedVersion, parse_original, write_parquet
from datamedic.engine.io.readers import ParseLimits, detect_delimiter, detect_encoding
from datamedic.engine.preview import PREVIEW_ROWS

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _write(tmp_path: Path, name: str, content: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _parse(tmp_path: Path, name: str, content: bytes, fmt: str) -> ParsedVersion:
    ext = name.rsplit(".", 1)[1]
    return parse_original(_write(tmp_path, name, content), fmt, ext)


def _types(parsed: ParsedVersion) -> dict[str, tuple[str, str]]:
    return {c["name"]: (c["dtype"], c["inferred_type"]) for c in parsed.schema}


def _expect_error(code: str, tmp_path: Path, name: str, content: bytes, fmt: str) -> EngineError:
    with pytest.raises(EngineError) as info:
        _parse(tmp_path, name, content, fmt)
    assert info.value.code == code
    return info.value


SAMPLE = pd.DataFrame(
    {
        "patient_id": ["P1", "P2", "P3"],
        "age": pd.array([54, 61, None], dtype="Int64"),
        "weight": [70.5, 82.0, 64.2],
        "smoker": [True, False, True],
        "visit": ["2024-01-05", "2024-02-09", "2024-03-01"],
    }
)


def _assert_sample(parsed: ParsedVersion) -> None:
    assert parsed.table.n_rows == 3 and parsed.table.n_cols == 5
    types = _types(parsed)
    assert types["age"] == ("integer", "numeric")
    assert types["weight"] == ("float", "numeric")
    assert types["smoker"] == ("boolean", "boolean")
    assert types["visit"][1] == "datetime"
    assert types["patient_id"] == ("string", "categorical")
    frame = parsed.table.frame
    assert frame.columns[0] == ROW_ID
    assert frame[ROW_ID].tolist() == [0, 1, 2]
    assert frame["age"].isna().sum() == 1
    assert len(parsed.content_hash) == 32


# ---- Every supported format -----------------------------------------------------------------


def test_csv(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.csv", SAMPLE.to_csv(index=False).encode(), "csv")
    _assert_sample(parsed)
    assert parsed.parse_options == {
        "format": "csv",
        "encoding": "utf-8",
        "delimiter": ",",
        "header_row": 0,
    }


def test_tsv_uses_tab(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.tsv", SAMPLE.to_csv(index=False, sep="\t").encode(), "csv")
    _assert_sample(parsed)
    assert parsed.parse_options["delimiter"] == "\t"


def test_semicolon_txt_is_sniffed(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.txt", SAMPLE.to_csv(index=False, sep=";").encode(), "csv")
    _assert_sample(parsed)
    assert parsed.parse_options["delimiter"] == ";"


def test_xlsx(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        SAMPLE.to_excel(writer, sheet_name="cohort", index=False)
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="other", index=False)
    parsed = _parse(tmp_path, "a.xlsx", buf.getvalue(), "xlsx")
    _assert_sample(parsed)
    assert parsed.parse_options["sheet"] == "cohort"
    assert parsed.report["sheets"] == ["cohort", "other"]


def test_xls_fixture() -> None:
    parsed = parse_original(FIXTURES / "cohort.xls", "xls", "xls")
    types = _types(parsed)
    assert parsed.table.n_rows == 4
    assert types["visit"][1] == "datetime"
    assert types["smoker"] == ("boolean", "boolean")
    assert types["age"] == ("float", "numeric")
    assert parsed.report["sheets"] == ["cohort", "notes"]


def test_parquet_keeps_source_types(tmp_path: Path) -> None:
    frame = SAMPLE.assign(visit=pd.to_datetime(SAMPLE["visit"]), age=SAMPLE["age"].astype("Int64"))
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    parsed = _parse(tmp_path, "a.parquet", buf.getvalue(), "parquet")
    _assert_sample(parsed)
    assert parsed.parse_options == {"format": "parquet"}


def test_parquet_strings_are_not_retyped(tmp_path: Path) -> None:
    buf = io.BytesIO()
    pd.DataFrame({"code": ["1", "2", "3"]}).to_parquet(buf, index=False)
    parsed = _parse(tmp_path, "a.parquet", buf.getvalue(), "parquet")
    assert _types(parsed)["code"] == ("string", "categorical")


def test_json_records(tmp_path: Path) -> None:
    content = SAMPLE.to_json(orient="records").encode()
    _assert_sample(_parse(tmp_path, "a.json", content, "json"))


def test_json_lines(tmp_path: Path) -> None:
    content = SAMPLE.to_json(orient="records", lines=True).encode()
    parsed = _parse(tmp_path, "a.jsonl", content, "json")
    _assert_sample(parsed)
    assert parsed.parse_options["json_orient"] == "lines"


def test_json_column_oriented(tmp_path: Path) -> None:
    columns = {c: [None if pd.isna(v) else v for v in SAMPLE[c].tolist()] for c in SAMPLE}
    parsed = _parse(tmp_path, "a.json", json.dumps(columns).encode(), "json")
    _assert_sample(parsed)
    assert parsed.parse_options["json_orient"] == "columns"


def test_json_nested_objects_are_flattened(tmp_path: Path) -> None:
    content = json.dumps(
        {"data": [{"id": 1, "addr": {"city": "X", "geo": {"lat": 1.5}}, "tags": ["a", "b"]}]}
    ).encode()
    parsed = _parse(tmp_path, "a.json", content, "json")
    assert set(_types(parsed)) == {"id", "addr.city", "addr.geo.lat", "tags"}
    assert parsed.parse_options["json_orient"] == "records_under_key"
    assert parsed.table.frame["tags"].iloc[0] == '["a", "b"]'


# ---- Encodings, names, empty rows -----------------------------------------------------------


@pytest.mark.parametrize(
    "encoding,expected",
    [("utf-8", "utf-8"), ("utf-8-sig", "utf-8-sig"), ("utf-16", "utf-16"), ("cp1252", "cp1252")],
)
def test_encodings(tmp_path: Path, encoding: str, expected: str) -> None:
    path = _write(tmp_path, "a.csv", "city,n\nZürich,1\nGenève,2\n".encode(encoding))
    assert detect_encoding(path) == expected
    parsed = parse_original(path, "csv", "csv")
    assert parsed.table.frame["city"].tolist() == ["Zürich", "Genève"]


def test_header_names_are_made_unique_and_safe(tmp_path: Path) -> None:
    content = b"id,,id,_dm_row_id,  spaced   name \n1,2,3,4,5\n"
    parsed = _parse(tmp_path, "a.csv", content, "csv")
    names = [c["name"] for c in parsed.schema]
    assert names == ["id", "column_2", "id_2", "col_dm_row_id", "spaced name"]
    originals = [c["original_name"] for c in parsed.schema]
    assert originals == ["id", "", "id", "_dm_row_id", "  spaced   name "]
    assert parsed.report["renamed_columns"] == 4


def test_empty_rows_dropped_and_reported(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.csv", b"a,b,\n1,2,\n,,\n3,4,\n", "csv")
    assert parsed.table.n_rows == 2
    assert [c["name"] for c in parsed.schema] == ["a", "b"]
    assert parsed.report["dropped_empty_rows"] == 1
    assert parsed.report["dropped_empty_columns"] == 1


def test_preview_is_capped_and_hides_row_id(tmp_path: Path) -> None:
    rows = "\n".join(f"{i},{'x' * 300}" for i in range(50))
    parsed = _parse(tmp_path, "a.csv", f"n,long\n{rows}\n".encode(), "csv")
    preview = parsed.preview
    assert len(preview["rows"]) == PREVIEW_ROWS and preview["truncated"]
    assert preview["n_rows_total"] == 50
    assert [c["name"] for c in preview["columns"]] == ["n", "long"]
    assert len(preview["rows"][0][1]) == 200 and preview["rows"][0][1].endswith("…")


def test_parquet_roundtrip_preserves_types_and_row_ids(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.csv", SAMPLE.to_csv(index=False).encode(), "csv")
    out = tmp_path / "v0.parquet"
    write_parquet(parsed.table, out)
    back = pd.read_parquet(out)
    assert back[ROW_ID].tolist() == [0, 1, 2]
    assert str(back["age"].dtype) == "Int64"
    assert str(back["smoker"].dtype) == "boolean"


def test_all_missing_column_is_stored_as_string(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.csv", b"a,b\n1,\n2,NA\n", "csv")
    assert _types(parsed)["b"] == ("string", "empty")
    out = tmp_path / "v0.parquet"
    write_parquet(parsed.table, out)
    assert pd.read_parquet(out)["b"].isna().all()


# ---- Malformed, empty and unsupported -------------------------------------------------------


@pytest.mark.parametrize(
    "name,content,fmt",
    [
        ("a.csv", b"", "csv"),
        ("a.csv", b"\n\n\n", "csv"),
        ("a.csv", b"a,b,c\n", "csv"),  # header only
        ("a.csv", b"a,b\n,\n , \n", "csv"),  # only empty rows
        ("a.json", b"[]", "json"),
        ("a.json", b"{}", "json"),
        ("a.jsonl", b"\n", "json"),
    ],
)
def test_empty_inputs_fail_cleanly(tmp_path: Path, name: str, content: bytes, fmt: str) -> None:
    _expect_error(PARSE_FAILED, tmp_path, name, content, fmt)


def test_ragged_csv_reports_line(tmp_path: Path) -> None:
    err = _expect_error(PARSE_FAILED, tmp_path, "a.csv", b"a,b\n1,2\n3,4,5\n", "csv")
    assert "Line 3" in err.message and "3 fields" in err.message


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"[1, 2, 3]",  # array of scalars
        b'[{"a": 1}, 2]',
        b'{"a": [1, 2], "b": [1]}',  # unequal columns
        b'{"a": 1}',  # a single object is not a table
    ],
)
def test_malformed_json(tmp_path: Path, content: bytes) -> None:
    _expect_error(PARSE_FAILED, tmp_path, "a.json", content, "json")


def test_jsonl_with_non_object_line(tmp_path: Path) -> None:
    _expect_error(PARSE_FAILED, tmp_path, "a.jsonl", b'{"a": 1}\n[1]\n', "json")


def test_deep_json_is_rejected(tmp_path: Path) -> None:
    nested: object = 1
    for _ in range(12):
        nested = {"k": nested}
    content = json.dumps([{"a": nested}]).encode()
    _expect_error(LIMIT_EXCEEDED, tmp_path, "a.json", content, "json")


def test_pathologically_deep_json_does_not_crash(tmp_path: Path) -> None:
    content = b"[" * 100_000 + b"]" * 100_000
    err = _expect_error(LIMIT_EXCEEDED, tmp_path, "a.json", content, "json")
    assert "nesting" in err.message


def test_corrupt_xlsx(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        zf.writestr("xl/workbook.xml", b"<not-a-workbook")
    _expect_error(PARSE_FAILED, tmp_path, "a.xlsx", buf.getvalue(), "xlsx")


def test_corrupt_xls(tmp_path: Path) -> None:
    content = bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 600
    _expect_error(PARSE_FAILED, tmp_path, "a.xls", content, "xls")


def test_corrupt_parquet(tmp_path: Path) -> None:
    _expect_error(PARSE_FAILED, tmp_path, "a.parquet", b"PAR1" + b"\x00" * 64 + b"PAR1", "parquet")


def test_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(EngineError) as info:
        parse_original(_write(tmp_path, "a.bin", b"x"), "exe", "bin")
    assert info.value.code == "UNSUPPORTED_FORMAT"


def test_column_limit(tmp_path: Path) -> None:
    header = ",".join(f"c{i}" for i in range(11))
    path = _write(tmp_path, "a.csv", f"{header}\n{','.join('1' * 11)}\n".encode())
    with pytest.raises(EngineError) as info:
        parse_original(path, "csv", "csv", ParseLimits(max_columns=10))
    assert info.value.code == LIMIT_EXCEEDED


def test_cell_limit(tmp_path: Path) -> None:
    rows = "\n".join("1,2" for _ in range(10))
    path = _write(tmp_path, "a.csv", f"a,b\n{rows}\n".encode())
    with pytest.raises(EngineError) as info:
        parse_original(path, "csv", "csv", ParseLimits(max_cells=19))
    assert info.value.code == LIMIT_EXCEEDED


def test_parquet_limits_checked_from_metadata(tmp_path: Path) -> None:
    buf = io.BytesIO()
    pd.DataFrame(np.zeros((10, 3))).rename(columns=str).to_parquet(buf, index=False)
    path = _write(tmp_path, "a.parquet", buf.getvalue())
    with pytest.raises(EngineError) as info:
        parse_original(path, "parquet", "parquet", ParseLimits(max_cells=29))
    assert info.value.code == LIMIT_EXCEEDED


def test_formula_like_cells_stay_inert_text(tmp_path: Path) -> None:
    parsed = _parse(tmp_path, "a.csv", b'name,f\na,"=HYPERLINK(""x"")"\nb,+1+1\n', "csv")
    assert parsed.table.frame["f"].tolist() == ['=HYPERLINK("x")', "+1+1"]


def test_delimiter_defaults_to_comma_for_single_column(tmp_path: Path) -> None:
    path = _write(tmp_path, "a.csv", b"only\n1\n2\n")
    assert detect_delimiter(path, "utf-8", "csv") == ","


def test_json_key_collision_fails_instead_of_dropping_data(tmp_path: Path) -> None:
    content = json.dumps([{"a.b": 1, "a": {"b": 2}}]).encode()
    err = _expect_error(PARSE_FAILED, tmp_path, "a.json", content, "json")
    assert "same column name" in err.message


def test_header_that_looks_like_data_is_reported(tmp_path: Path) -> None:
    no_header = _parse(tmp_path, "a.csv", b"1,54,0\n2,61,1\n", "csv")
    assert no_header.report["header_looks_like_data"] is True
    with_header = _parse(tmp_path, "b.csv", b"id,age,y\n1,54,0\n", "csv")
    assert with_header.report["header_looks_like_data"] is False


def test_error_messages_never_contain_cell_values(tmp_path: Path) -> None:
    secret = "SECRET-PATIENT-NAME"
    cases = [
        ("a.csv", f"a,b\n1,{secret}\n3,4,{secret}\n".encode(), "csv"),
        ("a.json", f'[{{"a": "{secret}"}}, 7]'.encode(), "json"),
        ("a.json", f'{{"a": ["{secret}"], "b": []}}'.encode(), "json"),
        ("a.jsonl", f'{{"a": "{secret}"}}\n"{secret}"\n'.encode(), "json"),
    ]
    for name, content, fmt in cases:
        with pytest.raises(EngineError) as info:
            _parse(tmp_path, name, content, fmt)
        assert secret not in info.value.message


def test_mixed_type_excel_column_is_not_coerced(tmp_path: Path) -> None:
    buf = io.BytesIO()
    pd.DataFrame({"v": [1, "two", 3.5], "n": [1, 2, 3]}).to_excel(buf, index=False)
    parsed = _parse(tmp_path, "a.xlsx", buf.getvalue(), "xlsx")
    assert _types(parsed)["v"] == ("string", "categorical")
    assert parsed.table.frame["v"].tolist() == ["1", "two", "3.5"]
    assert _types(parsed)["n"] == ("integer", "numeric")


def test_invalid_utf8_falls_back_without_crashing(tmp_path: Path) -> None:
    # 0x81 is undefined in cp1252, so the file decodes as latin-1; nothing is dropped.
    parsed = _parse(tmp_path, "a.csv", b"name,n\nA\x81B,1\n", "csv")
    assert parsed.parse_options["encoding"] == "latin-1"
    assert parsed.table.n_rows == 1
