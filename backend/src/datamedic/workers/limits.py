"""Per-process memory cap for Celery worker processes (Linux).

Analysis jobs (profiling, diagnostics) run inside the worker process. RLIMIT_DATA bounds the
memory a process can allocate (heap and anonymous mappings, which is where NumPy/pandas data
lives), so a job that grows too large gets a Python MemoryError, which the runner turns into
a LIMIT_EXCEEDED job error, instead of taking down the container. RLIMIT_DATA rather than
RLIMIT_AS: shared libraries and allocator address-space reservations do not count, so the cap
measures real use.
"""

import sys
from typing import Any

MIB = 1024 * 1024


def apply_worker_memory_limit(limit_mb: int, resource_module: Any = None) -> bool:
    """Apply the cap to the current process. Returns True when a limit was set."""
    if limit_mb <= 0:
        return False
    if resource_module is None:
        if sys.platform == "win32":
            return False
        import resource as resource_module
    limit = limit_mb * MIB
    resource_module.setrlimit(resource_module.RLIMIT_DATA, (limit, limit))
    return True
