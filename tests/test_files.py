from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, 28, 22],
        "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
        "department": [
            "Engineering",
            "Sales",
            "Engineering",
            "Sales",
            "Engineering",
        ],
        "is_active": [True, True, False, True, True],
    })


def test_sink_parquet(sample_df: pl.DataFrame) -> None:
    with NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        lf = pql.LazyFrame(sample_df)
        lf.sink_parquet(tmp_path)
        assert tmp_path.exists()
        read_back = pl.read_parquet(tmp_path)
        assert_eq(read_back, sample_df)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_sink_csv(sample_df: pl.DataFrame) -> None:
    with NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        lf = pql.LazyFrame(sample_df)
        lf.sink_csv(tmp_path, separator=",", include_header=True)
        assert tmp_path.exists()
        read_back = pl.read_csv(tmp_path)
        assert_eq(read_back, sample_df, check_dtypes=False)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_sink_ndjson(sample_df: pl.DataFrame) -> None:
    with NamedTemporaryFile(
        encoding="utf-8", suffix=".ndjson", delete=False, mode="w"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        lf = pql.LazyFrame(sample_df)
        lf.sink_ndjson(tmp_path)
        assert tmp_path.exists()
        read_back = pl.read_ndjson(tmp_path)
        assert_eq(read_back, sample_df, check_dtypes=False)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
