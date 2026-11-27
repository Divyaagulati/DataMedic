"""Semantic content hash of a version (ARCHITECTURE §9.3, D4).

SHA-256 over (1) ordered column names and dtypes and (2) pandas row hashes taken after
sorting by ``_dm_row_id`` and excluding it. Independent of Parquet writer metadata.
"""

import hashlib

import pandas as pd

from datamedic.engine import ROW_ID


def content_hash(frame: pd.DataFrame) -> bytes:
    data = frame.sort_values(ROW_ID, kind="stable").drop(columns=[ROW_ID])
    digest = hashlib.sha256()
    for name, dtype in zip(data.columns, data.dtypes, strict=True):
        digest.update(f"{name}\x1f{dtype}\x1e".encode())
    if len(data.columns):
        rows = pd.util.hash_pandas_object(data, index=False)
        digest.update(rows.to_numpy().tobytes())
    return digest.digest()
