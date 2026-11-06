"""Content validation of uploaded files (ARCHITECTURE §7.2 steps 1–2).

Type is decided by magic bytes, never by the extension alone; the declared extension must
agree with the detected type. XLSX containers are inspected *before* any extraction.
This module never parses table content (that is Phase 3) and never logs file bytes.

Limitation: ZIP headers declare uncompressed sizes that an attacker can falsify. The
ratio/size checks here are a first gate; the Phase 3 parser must also enforce streaming
decompression limits.
"""

import codecs
import zipfile
from dataclasses import dataclass
from typing import BinaryIO

from datamedic.errors import FORMAT_MISMATCH, MALICIOUS_FILE, UNSUPPORTED_FORMAT

# Declared extension -> canonical format
EXTENSION_FORMATS: dict[str, str] = {
    "csv": "csv",
    "tsv": "csv",
    "txt": "csv",
    "xlsx": "xlsx",
    "xls": "xls",
    "parquet": "parquet",
    "json": "json",
    "jsonl": "json",
}

HEAD_BYTES = 64 * 1024
OLE2_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")
ZIP_SIGNATURE = b"PK\x03\x04"
PARQUET_MAGIC = b"PAR1"
OTHER_BINARY_SIGNATURES = (
    b"MZ",  # Windows executable
    b"\x7fELF",  # ELF executable
    b"%PDF",
    b"\x1f\x8b",  # gzip
    b"Rar!",
    b"7z\xbc\xaf\x27\x1c",
    b"\x89PNG",
    b"\xff\xd8\xff",  # JPEG
)

# XLSX container limits
XLSX_MAX_ENTRIES = 10_000
XLSX_MAX_TOTAL_UNCOMPRESSED = 2 * 1024 * 1024 * 1024
XLSX_MAX_RATIO = 200
XLSX_RATIO_MIN_SIZE = 1024 * 1024
XLSX_MAX_CONTENT_TYPES_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    detected_format: str | None = None
    code: str | None = None
    reason: str | None = None


def _fail(code: str, reason: str) -> ValidationResult:
    return ValidationResult(False, None, code, reason)


def extension_of(filename: str) -> str | None:
    if "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[1].lower()
    return ext if ext in EXTENSION_FORMATS else None


def sniff(head: bytes, tail: bytes) -> str:
    """Return one of: parquet, ole2, zip, binary, text."""
    if head.startswith(PARQUET_MAGIC) and tail.endswith(PARQUET_MAGIC):
        return "parquet"
    if head.startswith(OLE2_SIGNATURE):
        return "ole2"
    if head.startswith(ZIP_SIGNATURE):
        return "zip"
    if any(head.startswith(sig) for sig in OTHER_BINARY_SIGNATURES):
        return "binary"
    return "text" if _looks_like_text(head) else "binary"


def _decode_head(head: bytes) -> str | None:
    for bom, enc in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if head.startswith(bom):
            return head.decode(enc, errors="ignore")
    if b"\x00" in head:
        return None
    try:
        # The head may end mid-character; ignore only a truncated tail sequence.
        return head.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start >= len(head) - 3:
            return head[: exc.start].decode("utf-8")
    text = head.decode("cp1252", errors="replace")
    printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
    return text if text and printable / len(text) >= 0.95 else None


def _looks_like_text(head: bytes) -> bool:
    return _decode_head(head) is not None


def _first_significant_char(head: bytes) -> str | None:
    decoded = _decode_head(head)
    if decoded is None:
        return None
    stripped = decoded.lstrip()
    return stripped[0] if stripped else None


def inspect_xlsx(fh: BinaryIO) -> ValidationResult:
    try:
        with zipfile.ZipFile(fh) as zf:
            infos = zf.infolist()
            if len(infos) > XLSX_MAX_ENTRIES:
                return _fail(MALICIOUS_FILE, "Workbook archive has too many entries.")
            names = {info.filename for info in infos}
            if "[Content_Types].xml" not in names or "xl/workbook.xml" not in names:
                return _fail(FORMAT_MISMATCH, "File is a ZIP archive but not an Excel workbook.")
            total = 0
            for info in infos:
                name = info.filename.replace("\\", "/")
                lowered = name.lower()
                if name.startswith("/") or ".." in name.split("/"):
                    return _fail(MALICIOUS_FILE, "Workbook archive contains unsafe paths.")
                if info.flag_bits & 0x1:
                    return _fail(UNSUPPORTED_FORMAT, "Encrypted workbooks are not supported.")
                if lowered.endswith("vbaproject.bin"):
                    return _fail(MALICIOUS_FILE, "Macro-enabled workbooks are not accepted.")
                if lowered.startswith("xl/externallinks/"):
                    return _fail(MALICIOUS_FILE, "Workbooks with external links are not accepted.")
                total += info.file_size
                if (
                    info.file_size > XLSX_RATIO_MIN_SIZE
                    and info.file_size / max(info.compress_size, 1) > XLSX_MAX_RATIO
                ):
                    return _fail(MALICIOUS_FILE, "Workbook compression ratio is unsafe.")
            if total > XLSX_MAX_TOTAL_UNCOMPRESSED:
                return _fail(MALICIOUS_FILE, "Workbook expands beyond the allowed size.")
            ct_info = zf.getinfo("[Content_Types].xml")
            if ct_info.file_size > XLSX_MAX_CONTENT_TYPES_BYTES:
                return _fail(MALICIOUS_FILE, "Workbook content-type manifest is too large.")
            # Substring check only; no XML parsing at this stage.
            content_types = zf.read(ct_info).lower()
            if b"macroenabled" in content_types:
                return _fail(MALICIOUS_FILE, "Macro-enabled workbooks are not accepted.")
    except zipfile.BadZipFile:
        return _fail(MALICIOUS_FILE, "Workbook archive is corrupt.")
    return ValidationResult(True, "xlsx")


def validate_file(fh: BinaryIO, declared_extension: str, size: int) -> ValidationResult:
    """Validate an uploaded object. ``fh`` must be seekable."""
    expected = EXTENSION_FORMATS.get(declared_extension)
    if expected is None:
        return _fail(UNSUPPORTED_FORMAT, "File extension is not supported.")
    if size == 0:
        return _fail(FORMAT_MISMATCH, "File is empty.")
    fh.seek(0)
    head = fh.read(HEAD_BYTES)
    fh.seek(max(0, size - 8))
    tail = fh.read(8)
    fh.seek(0)
    kind = sniff(head, tail)

    if expected == "parquet":
        return (
            ValidationResult(True, "parquet")
            if kind == "parquet"
            else _fail(FORMAT_MISMATCH, "File content is not Parquet.")
        )
    if expected == "xls":
        return (
            ValidationResult(True, "xls")
            if kind == "ole2"
            else _fail(FORMAT_MISMATCH, "File content is not a legacy Excel (.xls) workbook.")
        )
    if expected == "xlsx":
        if kind != "zip":
            return _fail(FORMAT_MISMATCH, "File content is not an Excel (.xlsx) workbook.")
        return inspect_xlsx(fh)
    if kind != "text":
        return _fail(FORMAT_MISMATCH, "File content is not text.")
    if expected == "json":
        first = _first_significant_char(head)
        allowed = "{" if declared_extension == "jsonl" else "{["
        if first is None or first not in allowed:
            return _fail(FORMAT_MISMATCH, "File content is not JSON.")
        return ValidationResult(True, "json")
    return ValidationResult(True, "csv")
