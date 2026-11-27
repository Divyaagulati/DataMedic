"""Content hash (ARCHITECTURE §9.3): semantic, row-order independent, value sensitive."""

import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.hashing import content_hash


def frame(values: list[int], ids: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame({ROW_ID: ids or list(range(len(values))), "x": values})


def test_same_content_same_hash() -> None:
    assert content_hash(frame([1, 2, 3])) == content_hash(frame([1, 2, 3]))


def test_physical_row_order_does_not_matter() -> None:
    a = frame([1, 2, 3])
    b = a.iloc[::-1]
    assert content_hash(a) == content_hash(b)


def test_values_names_and_dtypes_matter() -> None:
    base = content_hash(frame([1, 2, 3]))
    assert content_hash(frame([1, 2, 4])) != base
    assert content_hash(frame([1, 2, 3]).rename(columns={"x": "y"})) != base
    assert content_hash(frame([1, 2, 3]).astype({"x": "float64"})) != base
