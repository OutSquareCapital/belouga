from functools import partial
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyochain as pc
import pytest
import sqlglot
from polars.testing import assert_frame_equal

import pql

assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)

type TestData = dict[str, Any]  # pyright: ignore[reportExplicitAny]


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


@pytest.fixture
def data() -> TestData:
    return _get_data()


def test_from_query(data: TestData) -> None:
    df = pl.DataFrame(data)
    qry = """--sql
    SELECT *
    FROM df
    """
    pql_df = pql.from_query(sqlglot.parse_one(qry), df=df).collect()
    pl_df = duckdb.from_query(qry).pl()
    assert_eq(pql_df, pl_df)


def test_from_duckdb_relation(data: TestData) -> None:
    rel = duckdb.from_arrow(pl.DataFrame(data))
    assert_eq(pql.LazyFrame(rel).collect(), rel.pl())


def test_from_table_function() -> None:
    rel = duckdb.table_function("duckdb_functions")
    assert_eq(pql.from_table_function("duckdb_functions").collect(), rel.pl())


def test_from_table(data: TestData) -> None:
    duckdb.from_arrow(pl.DataFrame(data)).create("test_table")
    assert_eq(pql.from_table("test_table").collect(), pl.DataFrame(data))


def test_from_pl_lazyframe(data: TestData) -> None:
    assert_eq(pql.LazyFrame(pl.LazyFrame(data)).collect(), pl.DataFrame(data))
    assert_eq(pql.from_df(pl.LazyFrame(data)).collect(), pl.DataFrame(data))


def test_from_pd_dataframe(data: TestData) -> None:
    import pandas as pd

    assert_eq(pql.LazyFrame(pd.DataFrame(data)).collect(), pl.DataFrame(data))
    assert_eq(pql.from_df(pd.DataFrame(data)).collect(), pl.DataFrame(data))


def test_from_pl_dataframe(data: TestData) -> None:
    assert_eq(pql.LazyFrame(pl.DataFrame(data)).collect(), pl.DataFrame(data))
    assert_eq(pql.from_df(pl.DataFrame(data)).collect(), pl.DataFrame(data))


def test_from_dict(data: TestData) -> None:
    assert_eq(pql.LazyFrame(data).collect(), pl.DataFrame(data, orient="col"))
    assert_eq(pql.from_dict(data).collect(), pl.from_dict(data))


def test_from_numpy_1d() -> None:

    data = [1, 2, 3, 4]

    arr1d = np.array(data)
    assert_eq(pql.LazyFrame(arr1d).collect(), pl.DataFrame(arr1d, orient="col"))
    assert_eq(
        pql.from_numpy(arr1d, "col").collect(), pl.from_numpy(arr1d, orient="col")
    )


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_numpy_2d(orient: pql.sql.typing.Orientation) -> None:

    data = [1, 2, 3, 4]
    arr2d = np.array([data, data, data, data, data, data, data, data])
    assert_eq(
        pql.LazyFrame(arr2d, orient=orient).collect(),
        pl.DataFrame(arr2d, orient=orient),
    )
    assert_eq(
        pql.from_numpy(arr2d, orient).collect(), pl.from_numpy(arr2d, orient=orient)
    )


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_numpy_3d(orient: pql.sql.typing.Orientation) -> None:

    arr3d = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    expected = arr3d if orient == "row" else arr3d.T
    assert_eq(
        pql.from_numpy(arr3d, orient).collect(),
        pl.DataFrame(expected.tolist(), orient=orient),
    )


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_numpy_4d(orient: pql.sql.typing.Orientation) -> None:

    arr4d = np.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
    expected = arr4d if orient == "row" else arr4d.T
    assert_eq(
        pql.from_numpy(arr4d, orient).collect(),
        pl.DataFrame(expected.tolist(), orient=orient),  # pyright: ignore[reportAny]
    )


def test_from_pql_expr() -> None:
    expr = duckdb.ConstantExpression(42)
    rel = duckdb.values(expr)
    assert_eq(pql.LazyFrame(pql.lit(42).inner).collect(), rel.pl())


def test_from_seq_of_dicts() -> None:
    dicts = pc.Iter(range(10)).map(lambda _: _get_data()).collect()
    assert_eq(pql.LazyFrame(dicts).collect(), pl.DataFrame(dicts))
    assert_eq(pql.from_records(dicts).collect(), pl.from_records(dicts))
    assert_eq(pql.from_dicts(dicts).collect(), pl.from_dicts(dicts))


def test_from_seq_of_seqs() -> None:
    seqs = pc.Iter(range(10)).map(lambda _: tuple(range(5))).collect()
    assert_eq(pql.LazyFrame(seqs).collect(), pl.DataFrame(seqs))
    assert_eq(pql.from_records(seqs).collect(), pl.from_records(seqs))


@pytest.mark.parametrize("orient", ["row", "col"])
def test_from_seq_of_seqs_orient(orient: pql.sql.typing.Orientation) -> None:
    seqs = ((1, 2, 3), (4, 5, 6))
    assert_eq(
        pql.LazyFrame(seqs, orient=orient).collect(),
        pl.DataFrame(seqs, orient=orient),
    )
    assert_eq(
        pql.from_records(seqs, orient=orient).collect(),
        pl.from_records(seqs, orient=orient),
    )


def test_from_seq_of_vals() -> None:
    vals = pc.Iter(range(10)).map(lambda _: 42).collect()
    assert_eq(pql.LazyFrame(vals).collect(), pl.DataFrame(vals))
    assert_eq(pql.from_records(vals).collect(), pl.from_records(vals))


def test_from_seq_of_str_vals() -> None:
    vals = ("x", "y", "z")
    assert_eq(pql.LazyFrame(vals).collect(), pl.DataFrame(vals))
    assert_eq(pql.from_records(vals).collect(), pl.from_records(vals))
