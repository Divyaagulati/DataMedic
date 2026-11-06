import io
import zipfile

import pytest

from datamedic.datasets.file_validation import extension_of, validate_file
from datamedic.errors import FORMAT_MISMATCH, MALICIOUS_FILE, UNSUPPORTED_FORMAT

OLE2 = bytes.fromhex("D0CF11E0A1B11AE1")


def check(content: bytes, ext: str):  # type: ignore[no-untyped-def]
    return validate_file(io.BytesIO(content), ext, len(content))


def xlsx(extra: dict[str, bytes] | None = None, content_types: bytes = b"<Types/>") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("a.csv", "csv"),
        ("A.CSV", "csv"),
        ("x.tar.json", "json"),
        ("x.jsonl", "jsonl"),
        ("noext", None),
        ("evil.exe", None),
        ("macro.xlsm", None),
    ],
)
def test_extension_of(name: str, expected: str | None) -> None:
    assert extension_of(name) == expected


def test_csv_utf8_ok() -> None:
    r = check("id,name\n1,Zoë\n".encode(), "csv")
    assert r.ok and r.detected_format == "csv"


def test_csv_utf16_bom_ok() -> None:
    r = check("id,name\n1,a\n".encode("utf-16"), "csv")
    assert r.ok


def test_csv_cp1252_ok() -> None:
    assert check("id,city\n1,Zürich\n".encode("cp1252"), "csv").ok


def test_binary_declared_as_csv_rejected() -> None:
    r = check(b"MZ\x90\x00" + b"\x00" * 100, "csv")
    assert not r.ok and r.code == FORMAT_MISMATCH


def test_nul_bytes_rejected_as_text() -> None:
    assert check(b"a,b\n\x00\x00\x01\x02", "csv").code == FORMAT_MISMATCH


def test_json_ok_and_mismatch() -> None:
    assert check(b'  [{"a": 1}]', "json").ok
    assert check(b'{"a": 1}\n{"a": 2}\n', "jsonl").ok
    assert check(b"[1,2]", "jsonl").code == FORMAT_MISMATCH
    assert check(b"a,b\n1,2", "json").code == FORMAT_MISMATCH


def test_parquet_requires_magic_both_ends() -> None:
    assert check(b"PAR1" + b"\x00" * 20 + b"PAR1", "parquet").ok
    assert check(b"PAR1" + b"\x00" * 20, "parquet").code == FORMAT_MISMATCH


def test_xls_signature() -> None:
    assert check(OLE2 + b"\x00" * 100, "xls").ok
    assert check(b"id,a\n1,2", "xls").code == FORMAT_MISMATCH


def test_xlsx_valid() -> None:
    r = check(xlsx(), "xlsx")
    assert r.ok and r.detected_format == "xlsx"


def test_xlsx_extension_but_csv_content() -> None:
    assert check(b"a,b\n1,2\n", "xlsx").code == FORMAT_MISMATCH


def test_zip_that_is_not_a_workbook() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", b"<doc/>")
    assert check(buf.getvalue(), "xlsx").code == FORMAT_MISMATCH


def test_xlsx_with_macros_rejected() -> None:
    assert check(xlsx({"xl/vbaProject.bin": b"\x00" * 10}), "xlsx").code == MALICIOUS_FILE


def test_xlsx_macro_content_type_rejected() -> None:
    ct = b'<Types><Override ContentType="application/vnd.ms-excel.sheet.macroEnabled.main+xml"/>'
    assert check(xlsx(content_types=ct), "xlsx").code == MALICIOUS_FILE


def test_xlsx_external_links_rejected() -> None:
    r = check(xlsx({"xl/externalLinks/externalLink1.xml": b"<x/>"}), "xlsx")
    assert r.code == MALICIOUS_FILE


def test_xlsx_zip_bomb_ratio_rejected() -> None:
    r = check(xlsx({"xl/worksheets/sheet1.xml": b"\x00" * (8 * 1024 * 1024)}), "xlsx")
    assert r.code == MALICIOUS_FILE and "ratio" in (r.reason or "")


def test_xlsx_path_traversal_rejected() -> None:
    assert check(xlsx({"../evil.xml": b"x"}), "xlsx").code == MALICIOUS_FILE


def test_corrupt_zip_rejected() -> None:
    assert check(b"PK\x03\x04" + b"garbage" * 10, "xlsx").code == MALICIOUS_FILE


def test_empty_and_unsupported() -> None:
    assert check(b"", "csv").code == FORMAT_MISMATCH
    assert check(b"abc", "exe").code == UNSUPPORTED_FORMAT
