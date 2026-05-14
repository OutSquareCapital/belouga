from functools import partial
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pyochain import Iter

import belugas as bl
import belugas.typing as t

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)
DATA = Path("tests", "data", "foo")
CSV = DATA.with_suffix(".csv")
JSON = DATA.with_suffix(".json")
PARQUET = DATA.with_suffix(".parquet")
type TestData = dict[str, Any]  # pyright: ignore[reportExplicitAny]


@pytest.fixture
def data() -> TestData:
    return _get_data()


def _get_data() -> TestData:
    return {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "sex": ["F", "M", "M", "M", "F"],
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
        "value": [10.0, None, 30.0, None, 50.0],
        "category": ["A", "B", None, "A", "B"],
    }


PL_DF = pl.DataFrame(_get_data())
REL = duckdb.from_arrow(PL_DF)


def test_protocols() -> None:
    df = pl.DataFrame([[1, 2], [3, 4]])

    assert isinstance(df, t.IntoPlDataFrame)
    assert isinstance(df.lazy(), t.IntoPlLazyFrame)
    assert isinstance(df.to_pandas(), t.IntoArrowStream)
    assert isinstance(df.to_numpy(), t.NPArrayLike)
    assert isinstance(df.to_arrow(), t.IntoArrowStream)


def test_from_duckdb_relation() -> None:
    assert_eq(bl.LazyFrame(REL).collect(), REL.pl())


def test_from_arrow() -> None:
    assert_eq(bl.from_arrow(PL_DF).collect(), REL.pl())


def test_from_table_function() -> None:
    assert_eq(
        bl.from_table_function("duckdb_functions").collect(),
        duckdb.table_function("duckdb_functions").pl(),
    )


def test_from_table() -> None:
    REL.create("test_table")
    assert_eq(bl.from_table("test_table").collect(), PL_DF)


def test_from_polars() -> None:
    assert_eq(bl.LazyFrame(PL_DF.lazy()).collect(), PL_DF)
    assert_eq(bl.LazyFrame(PL_DF).collect(), PL_DF)
    assert_eq(bl.from_polars(PL_DF.lazy()).collect(), PL_DF)
    assert_eq(bl.from_polars(PL_DF).collect(), PL_DF)


def test_from_pd_dataframe(data: TestData) -> None:
    import pandas as pd

    pd_df = pd.DataFrame(data)

    assert_eq(bl.LazyFrame(pd_df).collect(), PL_DF)
    assert_eq(bl.from_arrow(pd_df).collect(), PL_DF)
    assert_eq(bl.from_pandas(pd_df).collect(), PL_DF)


def test_from_pl_dataframe() -> None:
    assert_eq(bl.LazyFrame(PL_DF).collect(), PL_DF)
    assert_eq(bl.from_arrow(PL_DF).collect(), PL_DF)


@pytest.mark.skip(reason="TODO, need investigation.")
def test_from_arrow_sort_limit() -> None:
    """Reused `arrow` data sources don't seem to work well with our current query execution.

    `belugas` generate empty result, but ONLY with this specific setup.
    """
    df = pl.DataFrame({"x": [3, 1, 2]})
    assert_eq(
        df.lazy().sort("x").limit(1).collect(),
        bl.from_arrow(df).sort("x").limit(1).collect(),
    )


def test_from_dict_sort_limit_control() -> None:
    data = {"x": [3, 1, 2]}
    assert_eq(
        pl.DataFrame(data).lazy().sort("x").limit(1).collect(),
        bl.from_dict(data).sort("x").limit(1).collect(),
    )


def test_from_dict(data: TestData) -> None:
    assert_eq(bl.LazyFrame(data).collect(), pl.DataFrame(data, orient="col"))
    assert_eq(bl.from_dict(data).collect(), pl.from_dict(data))


def test_from_numpy_1d() -> None:

    data = [1, 2, 3, 4]

    arr1d = np.array(data)
    assert_eq(bl.LazyFrame(arr1d).collect(), pl.DataFrame(arr1d, orient="col"))
    assert_eq(bl.from_numpy(arr1d, "col").collect(), pl.from_numpy(arr1d, orient="col"))


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_numpy_2d(orient: t.Orientation) -> None:

    data = [1, 2, 3, 4]
    arr2d = np.array([data, data, data, data, data, data, data, data])
    assert_eq(
        bl.LazyFrame(arr2d, orient=orient).collect(),
        pl.DataFrame(arr2d, orient=orient),
    )
    assert_eq(
        bl.from_numpy(arr2d, orient).collect(), pl.from_numpy(arr2d, orient=orient)
    )


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_numpy_3d(orient: t.Orientation) -> None:

    arr3d = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    expected = arr3d if orient == "row" else arr3d.T
    assert_eq(
        bl.from_numpy(arr3d, orient).collect(),
        pl.DataFrame(expected.tolist(), orient=orient),
    )


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_numpy_4d(orient: t.Orientation) -> None:

    arr4d = np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    expected = arr4d if orient == "row" else arr4d.T
    assert_eq(
        bl.from_numpy(arr4d, orient).collect(),
        pl.DataFrame(expected.tolist(), orient=orient),  # pyright: ignore[reportAny]
    )


def test_from_seq_of_dicts() -> None:
    dicts = Iter(range(10)).map(lambda _: _get_data()).collect()
    assert_eq(bl.LazyFrame(dicts).collect(), pl.DataFrame(dicts))
    assert_eq(bl.from_records(dicts).collect(), pl.from_records(dicts))
    assert_eq(bl.from_dicts(dicts).collect(), pl.from_dicts(dicts))


def test_from_seq_of_seqs() -> None:
    seqs = Iter(range(10)).map(lambda _: tuple(range(5))).collect()
    assert_eq(bl.LazyFrame(seqs).collect(), pl.DataFrame(seqs))
    assert_eq(bl.from_records(seqs).collect(), pl.from_records(seqs))


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_seq_of_seqs_orient(orient: t.Orientation) -> None:
    seqs = ((1, 2, 3), (4, 5, 6))
    assert_eq(
        bl.LazyFrame(seqs, orient=orient).collect(),
        pl.DataFrame(seqs, orient=orient),
    )
    assert_eq(
        bl.from_records(seqs, orient=orient).collect(),
        pl.from_records(seqs, orient=orient),
    )


def test_from_seq_of_vals() -> None:
    vals = Iter(range(10)).map(lambda _: 42).collect()
    assert_eq(bl.LazyFrame(vals).collect(), pl.DataFrame(vals))
    assert_eq(bl.from_records(vals).collect(), pl.from_records(vals))


def test_from_seq_of_str_vals() -> None:
    vals = ("x", "y", "z")
    assert_eq(bl.LazyFrame(vals).collect(), pl.DataFrame(vals))
    assert_eq(bl.from_records(vals).collect(), pl.from_records(vals))


def test_from_csv() -> None:
    bl_df = bl.scan_csv(CSV).collect()
    pl_df = duckdb.from_csv_auto(CSV).pl()
    assert_eq(bl_df, pl_df)


def test_from_json() -> None:
    bl_df = bl.scan_json(JSON).collect()
    pl_df = duckdb.read_json(JSON).pl()
    assert_eq(bl_df, pl_df)


def test_from_parquet() -> None:
    bl_df = bl.scan_parquet(PARQUET).collect()
    pl_df = duckdb.read_parquet(str(PARQUET)).pl()
    assert_eq(bl_df, pl_df)
