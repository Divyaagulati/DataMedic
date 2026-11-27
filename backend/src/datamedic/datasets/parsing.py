"""Worker side of the parse sandbox: copy the original to a private temp dir, run the
resource-limited child process, collect its outputs.

The child's stdout/stderr are discarded, never logged: tracebacks from parsers can contain
cell values (CLAUDE.md invariant 2).
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datamedic.config import Settings
from datamedic.storage import ObjectStorage

log = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
PARSE_FAILED = "PARSE_FAILED"


@dataclass
class SandboxOutcome:
    ok: bool
    code: str | None = None
    message: str | None = None
    result: dict[str, Any] | None = None
    parquet_path: Path | None = None


def _child_env(workdir: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "TMP": str(workdir),
        "TEMP": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        # The system allocator does not reserve large virtual ranges, so RLIMIT_AS measures
        # real use rather than allocator reservations.
        "ARROW_DEFAULT_MEMORY_POOL": "system",
        # Single-threaded native libraries: OpenBLAS otherwise reserves per-CPU buffers at
        # import and aborts the process (not a Python error) under a memory limit.
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENPYXL_DEFUSEDXML": "True",
    }
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
    return env


def copy_original(storage: ObjectStorage, object_key: str, dest: Path) -> None:
    with storage.open_reader(object_key) as src, open(dest, "xb") as out:
        while chunk := src.read(_CHUNK):
            out.write(chunk)


def run_parse_sandbox(
    settings: Settings, source: Path, workdir: Path, fmt: str, declared_extension: str
) -> SandboxOutcome:
    outdir = workdir / "out"
    outdir.mkdir()
    limits = {
        "memory_mb": settings.parse_memory_limit_mb,
        "cpu_seconds": settings.parse_timeout_seconds,
        "max_output_bytes": settings.parse_max_output_bytes,
        "max_columns": settings.max_columns,
        "max_cells": settings.max_cells,
        "max_json_depth": settings.max_json_depth,
    }
    cmd = [
        sys.executable,
        "-I",
        "-m",
        "datamedic.workers.parse_sandbox",
        str(source),
        str(outdir),
        fmt,
        declared_extension,
        json.dumps(limits),
    ]
    try:
        proc = subprocess.run(  # noqa: S603  (fixed argv, no shell, no user strings)
            cmd,
            env=_child_env(workdir),
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=settings.parse_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxOutcome(
            False,
            LIMIT_EXCEEDED,
            f"Parsing took longer than the allowed {settings.parse_timeout_seconds} seconds.",
        )
    result_path = outdir / "result.json"
    if proc.returncode != 0 or not result_path.is_file():
        log.warning("parse sandbox exited abnormally", extra={"returncode": proc.returncode})
        if proc.returncode < 0:  # killed by a signal (memory or CPU limit)
            return SandboxOutcome(
                False, LIMIT_EXCEEDED, "Parsing this file exceeded the allowed resources."
            )
        return SandboxOutcome(False, PARSE_FAILED, "The file could not be parsed.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("ok"):
        return SandboxOutcome(False, result.get("code", PARSE_FAILED), result.get("message"))
    parquet = outdir / "table.parquet"
    if not parquet.is_file():
        return SandboxOutcome(False, PARSE_FAILED, "The file could not be parsed.")
    return SandboxOutcome(True, result=result, parquet_path=parquet)
