"""Parse sandbox: the child process that parses an untrusted original (ARCHITECTURE §7.2 #4).

Run as:  python -I -m datamedic.workers.parse_sandbox <input> <outdir> <fmt> <ext> <limits>

Resource limits (address space, CPU time, output file size, no core dumps) are applied
before pandas or any parser is imported. They are enforced on Linux (the worker image); on
other platforms only the parent's wall-clock timeout applies.

Writes ``table.parquet`` and ``result.json`` to <outdir>. Expected failures (bad files,
limits) are reported in ``result.json`` with exit code 0. Anything else crashes the process;
the parent reports a generic error and never logs this process's output, which could
contain data values.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_USAGE = 2


def apply_limits(memory_mb: int, cpu_seconds: int, max_output_bytes: int) -> bool:
    if sys.platform == "win32":  # no rlimits on Windows; only the parent's timeout applies
        return False
    import resource

    memory = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return True


def _write_result(outdir: Path, payload: dict[str, Any]) -> None:
    tmp = outdir / "result.json.part"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, outdir / "result.json")


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        return EXIT_USAGE
    source, outdir_arg, fmt, declared_extension, limits_json = argv
    limits = json.loads(limits_json)
    outdir = Path(outdir_arg)
    limited = apply_limits(
        int(limits["memory_mb"]), int(limits["cpu_seconds"]), int(limits["max_output_bytes"])
    )

    # The heavy libraries are imported only after the limits are in place, inside the guarded
    # block: under a tight memory limit even importing pandas can fail.
    from datamedic.engine.errors import LIMIT_EXCEEDED, EngineError  # stdlib-only module

    memory_message = "Parsing this file needs more memory than allowed."
    try:
        from datamedic.engine.io.parse import parse_original, write_parquet
        from datamedic.engine.io.readers import ParseLimits
    except (MemoryError, ImportError):
        if not limited:
            raise  # a genuine installation problem, not a resource limit
        _write_result(outdir, {"ok": False, "code": LIMIT_EXCEEDED, "message": memory_message})
        return EXIT_OK

    parse_limits = ParseLimits(
        max_columns=int(limits["max_columns"]),
        max_cells=int(limits["max_cells"]),
        max_json_depth=int(limits["max_json_depth"]),
    )
    try:
        parsed = parse_original(Path(source), fmt, declared_extension, parse_limits)
        write_parquet(parsed.table, outdir / "table.parquet")
        payload: dict[str, Any] = {
            "ok": True,
            "schema": parsed.schema,
            "parse_options": parsed.parse_options,
            "report": {**parsed.report, "resource_limits_enforced": limited},
            "content_hash": parsed.content_hash.hex(),
            "preview": parsed.preview,
            "n_rows": parsed.table.n_rows,
            "n_cols": parsed.table.n_cols,
        }
    except EngineError as exc:
        payload = {"ok": False, "code": exc.code, "message": exc.message}
    except MemoryError:
        payload = {"ok": False, "code": LIMIT_EXCEEDED, "message": memory_message}
    _write_result(outdir, payload)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
