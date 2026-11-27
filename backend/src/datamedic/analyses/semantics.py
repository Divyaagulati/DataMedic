"""Semantics and check-set hashes (D8). A run whose semantics hash differs from the version's
current one is stale: the domain or roles changed after it ran."""

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from datamedic.engine.checks.spec import CheckSpec


def roles_snapshot(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Canonical, sorted list of {column, role, task_type} from ColumnRole-like rows."""
    return sorted(
        ({"column": r.column_name, "role": r.role, "task_type": r.task_type} for r in rows),
        key=lambda r: (r["column"], r["role"]),
    )


def semantics_hash(domain: str, snapshot: list[dict[str, Any]]) -> bytes:
    payload = json.dumps({"domain": domain, "roles": snapshot}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).digest()


def check_set_hash(specs: Iterable[CheckSpec]) -> bytes:
    payload = "\n".join(sorted(f"{s.id}@{s.version}" for s in specs))
    return hashlib.sha256(payload.encode("utf-8")).digest()
