"""DataMedic analysis engine.

PURE package: it must not import FastAPI, SQLAlchemy, Celery, boto3 or any other part of
``datamedic`` outside ``datamedic.engine`` (enforced by tests/unit/test_engine_boundary.py).
The API and the workers are thin adapters around it.
"""

ENGINE_VERSION = "0.3.0"

ROW_ID = "_dm_row_id"
"""Hidden stable row identifier (int64, 0..n-1 in source order). Never a feature."""
