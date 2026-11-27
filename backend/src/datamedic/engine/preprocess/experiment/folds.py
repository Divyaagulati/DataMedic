"""Stored fold assignments (the frozen split every candidate shares).

A split is stored as JSON: the sorted ``_dm_row_id`` values and, per row, the fold in which
it is a test row (``-1``: never tested, earliest block of a time split). Training rows are
derived from that map:
- ``partition`` schemes (k-fold, stratified, group): training rows of fold k are every
  cohort row not tested in fold k;
- ``forward_chaining`` (time): training rows of fold k are the never-tested rows plus the
  rows tested in an earlier fold.

The JSON's SHA-256 is recorded with the split definition, so every experiment that reuses
it can prove it evaluated exactly the same rows.
"""

import hashlib
import json
from typing import Any

import numpy as np

from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.split import Fold, Splits

FORMAT_VERSION = 1


def encode(splits: Splits) -> bytes:
    assignment = splits.assignment()
    row_ids = sorted(assignment)
    payload = {
        "format": FORMAT_VERSION,
        "method": splits.method,
        "n_splits": splits.n_splits,
        "seed": splits.seed,
        "scheme": "forward_chaining" if splits.method == "time" else "partition",
        "row_ids": row_ids,
        "fold": [assignment[r] for r in row_ids],
    }
    data = json.dumps(payload, separators=(",", ":")).encode()
    # The encoding must reproduce the folds exactly, or it is not a faithful record.
    rebuilt = decode(data)
    for a, b in zip(rebuilt.folds, splits.folds, strict=True):
        if not (
            np.array_equal(a.train_ids, b.train_ids) and np.array_equal(a.test_ids, b.test_ids)
        ):
            raise EngineError(
                "POSTCONDITION_FAILED", "The fold assignment cannot be stored faithfully."
            )
    return data


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def decode(data: bytes) -> Splits:
    payload: dict[str, Any] = json.loads(data)
    if payload.get("format") != FORMAT_VERSION:
        raise EngineError("SPLIT_INVALID", "Unsupported fold assignment format.")
    row_ids = np.asarray(payload["row_ids"], dtype=np.int64)
    fold = np.asarray(payload["fold"], dtype=np.int64)
    n_splits = int(payload["n_splits"])
    forward = payload["scheme"] == "forward_chaining"
    folds = []
    for k in range(n_splits):
        test = row_ids[fold == k]
        if forward:
            train = row_ids[(fold == -1) | ((fold >= 0) & (fold < k))]
        else:
            train = row_ids[fold != k]
        folds.append(Fold(k, np.sort(train), np.sort(test)))
    return Splits(str(payload["method"]), n_splits, int(payload["seed"]), folds)
