"""The Celery worker must keep DataMedic's logging (JSON + scrubber, no tracebacks).

By default Celery replaces the root handlers at worker start; tasks.py connects a
``setup_logging`` receiver so it does not. Run in a subprocess: importing the worker module
configures global logging state.
"""

import json
import os
import subprocess
import sys

SCRIPT = r"""
import logging
from datamedic.workers.tasks import celery_app
celery_app.log.setup(loglevel="INFO")  # what `celery worker` does at start-up
log = logging.getLogger("datamedic.workers.runner")
try:
    raise ValueError("could not convert string to float: 'SECRET-CELL-VALUE'")
except ValueError:
    log.exception("job crashed", extra={"job_id": "j1", "values": ["SECRET-CELL-VALUE"]})
"""


def test_worker_logs_stay_json_without_values() -> None:
    env = {
        **os.environ,
        "DATAMEDIC_DATABASE_URL": "postgresql+psycopg://u:p@localhost/x",
        "DATAMEDIC_MAINTENANCE_DATABASE_URL": "postgresql+psycopg://u:p@localhost/x",
        "DATAMEDIC_REDIS_URL": "memory://",
    }
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, check=True
    )
    logs = out.stdout + out.stderr
    assert "SECRET-CELL-VALUE" not in logs
    record = json.loads(next(line for line in logs.splitlines() if "job crashed" in line))
    assert record["exc_type"] == "ValueError" and record["values"] == "[redacted]"
