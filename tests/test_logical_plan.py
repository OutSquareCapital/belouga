"""Tests for the logical plan generation and optimization.

Note:
    - The expected results are what we currently get, not necessarily what we should/would get.
    - If the test breaks, check if the new plan is actually better than the old one before updating the expected results.
    - This is currently far from exhaustive
"""

from __future__ import annotations

import polars as pl
import pytest
from pyochain import Seq
from sqlglot import exp

import belugas as bl

from ._data import LF_TEST
from ._utils import assert_lf_eq

bl_age = bl.col("age")
bl_text = bl.col("text")
bl_salary = bl.col("salary")
pl_age = pl.col("age")
pl_salary = pl.col("salary")


@pytest.fixture
def lf() -> bl.LazyFrame:
    return LF_TEST


def test_drop_inline(lf: bl.LazyFrame) -> None:
    """This should inline the original `SELECT *` from the scan."""
    selects_nb = (
        lf
        .drop("name")
        .select(bl.all())
        .drop("age")
        .select(bl.all())
        .drop("salary")
        .query.logical()
        .pipe(lambda e: Seq(e.find_all(exp.Select)))
        .length()
    )
    assert selects_nb == 3


def test_flattens_consecutive_filters(lf: bl.LazyFrame) -> None:
    query = lf.filter(bl_age.gt(25)).filter(bl_salary.gt(50_000), department="Sales")
    assert_lf_eq(
        lf
        .lazy()
        .filter(pl_age.gt(25))
        .filter(pl_salary.gt(50_000), department="Sales"),
        query,
    )
    sql = query.query.logical()
    assert len(tuple(sql.find_all(exp.Where))) == 1


def test_flattens_consecutive_limits(lf: bl.LazyFrame) -> None:
    query = lf.limit(4).limit(2)
    assert_lf_eq(lf.lazy().limit(4).limit(2), query)
    sql = query.query.logical()
    assert len(tuple(sql.find_all(exp.Limit))) == 1


def test_flattens_consecutive_sorts(lf: bl.LazyFrame) -> None:
    query = lf.sort("age").sort("salary")
    assert_lf_eq(lf.lazy().sort("age").sort("salary"), query)
    sql = query.query.logical()
    assert len(tuple(sql.find_all(exp.Order))) == 1


def test_flattens_consecutive_drops(lf: bl.LazyFrame) -> None:
    query = lf.drop("value").drop("category")
    assert_lf_eq(lf.lazy().drop("value").drop("category"), query)


def test_flattens_consecutive_renames(lf: bl.LazyFrame) -> None:
    first = {"department": "dept"}
    second = {"dept": "team", "age": "years"}

    assert_lf_eq(
        lf.lazy().rename(first).rename(second), lf.rename(first).rename(second)
    )


def test_flattens_consecutive_slices(lf: bl.LazyFrame) -> None:
    query = lf.slice(2, 5).slice(1, 2)
    assert_lf_eq(lf.lazy().slice(2, 5).slice(1, 2), query)


def test_flattens_limit_then_slice(lf: bl.LazyFrame) -> None:
    query = lf.limit(6).slice(2, 3)
    assert_lf_eq(lf.lazy().limit(6).slice(2, 3), query)


def test_flattens_slice_then_limit(lf: bl.LazyFrame) -> None:
    query = lf.slice(2, 5).limit(3)
    assert_lf_eq(lf.lazy().slice(2, 5).limit(3), query)


def test_rename_inline() -> None:
    """This should inline the original `SELECT *` from the scan.

    ```sql
    SELECT
        "a" AS "new_a",
        "b" AS "b",
        "c" AS "c"
    FROM "foo"
    ```
    """
    lf = bl.LazyFrame({"a": [1, 1, 2], "b": [1, 2, 2], "c": [1, 2, 3]})
    selects_nb = (
        lf
        .rename({"a": "new_a"})
        .query.logical()
        .pipe(lambda e: Seq(e.find_all(exp.Select)))
        .length()
    )
    assert selects_nb == 1


def test_struct_inline() -> None:
    """This should inline the original `SELECT *` from the scan.

    ```sql
    SELECT
        UNNEST("a") AS "a",
        "b"
    FROM "bl_scan_1296621563952"
    ```
    """
    nested = {
        "a": [[1, 2], [3, 4]],
        "b": [["a", "b"], ["c", "d"]],
    }
    selects_nb = (
        bl
        .LazyFrame(nested)
        .unnest("a")
        .query.logical()
        .pipe(lambda e: Seq(e.find_all(exp.Select)))
        .length()
    )
    assert selects_nb == 1


def test_row_index_keep_same_select() -> None:
    """A new `row_index` should just be added to the original `SELECT`."""
    lf = bl.LazyFrame({"a": [1, 1, 2], "b": [1, 2, 2], "c": [1, 2, 3]})
    selects_nb = (
        lf
        .with_row_index(order_by="c")
        .with_row_index(order_by="c")
        .with_row_index(order_by="c")
        .query.logical()
        .pipe(lambda e: Seq(e.find_all(exp.Select)))
        .length()
    )
    assert selects_nb == 1
