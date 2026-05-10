from __future__ import annotations

import polars as pl
import pytest
from pyochain import Iter

import belugas as bl
import belugas.typing as t

from ._utils import assert_lf_eq

_DF = pl.DataFrame({
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
})


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return _DF


def test_unpivot() -> None:
    data = pl.DataFrame({"id": ["a", "b"], "x": [1, 3], "y": [2, 4]})
    assert_lf_eq(
        data.lazy().unpivot(on=["x", "y"], index="id"),
        bl.LazyFrame(data).unpivot(on=["x", "y"], index="id"),
    )


def test_pivot_single_value_column(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="id",
            values="salary",
        ),
        bl.LazyFrame(sample_df).pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="id",
            values="salary",
        ),
    )


def test_pivot_multiple_value_columns(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department", on_columns=["Engineering", "Sales"], index="id"
        ),
        bl.LazyFrame(sample_df).pivot(
            "department", on_columns=["Engineering", "Sales"], index="id"
        ),
    )


@pytest.mark.parametrize(
    "agg", Iter[str](t.PivotAgg.__args__).filter(lambda agg: agg != "sum").collect()
)
def test_pivot_aggregate_fns(sample_df: pl.DataFrame, agg: t.PivotAgg) -> None:
    """The `count` option is currently deprecated for polars, but `len` is NOT the right func to call in `SqlExpr`.

    We keep both `count` and `len` as options for the `aggregate_function` parameter, but they both map to `len` in polars.
    """
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="sex",
            values="salary",
            aggregate_function="len" if agg == "count" else agg,
        ),
        bl.LazyFrame(sample_df).pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="sex",
            values="salary",
            aggregate_function=agg,
        ),
    )


def test_pivot_aggregate_sum(sample_df: pl.DataFrame) -> None:
    """Sum in `polars` is at 0 for null values, but return null in `DuckDB`."""
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="sex",
            values="salary",
            aggregate_function="sum",
        ),
        bl
        .LazyFrame(sample_df)
        .pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="sex",
            values="salary",
            aggregate_function="sum",
        )
        .with_columns(bl.col("Sales").fill_null(0)),
    )


def test_pivot_custom_separator(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="id",
            separator="__",
        ),
        bl.LazyFrame(sample_df).pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="id",
            separator="__",
        ),
    )


def test_pivot_auto_detect_index(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department", on_columns=["Engineering", "Sales"], values="salary"
        ),
        bl.LazyFrame(sample_df).pivot(
            "department", on_columns=["Engineering", "Sales"], values="salary"
        ),
    )


def test_pivot_maintain_order(sample_df: pl.DataFrame) -> None:
    assert_lf_eq(
        sample_df.lazy().pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="id",
            values="salary",
            maintain_order=True,
        ),
        bl.LazyFrame(sample_df).pivot(
            "department",
            on_columns=["Engineering", "Sales"],
            index="id",
            values="salary",
            maintain_order=True,
        ),
    )


def test_pivot_integer_on_columns(sample_df: pl.DataFrame) -> None:
    cols = (1, 2, 3, 4, 5)
    assert_lf_eq(
        sample_df.lazy().pivot(
            "id",
            on_columns=cols,
            index="department",
            values="salary",
            aggregate_function="first",
        ),
        bl.LazyFrame(sample_df).pivot(
            "id",
            on_columns=cols,
            index="department",
            values="salary",
            aggregate_function="first",
        ),
    )


def test_pivot_no_index_no_values_error(sample_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"index.*or.*values"):
        _ = (
            bl
            .LazyFrame(sample_df)
            .pivot("department", on_columns=["Engineering", "Sales"])
            .collect()
        )
