from __future__ import annotations

from functools import partial

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


def test_agg(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(pql.col("salary").mean().alias("mean_salary"))
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(pl.col("salary").mean().alias("mean_salary"))
        .sort("department")
        .collect(),
    )


def test_agg_by_prefix(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(pql.col("salary").mean().name.prefix("avg_"))
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(pl.col("salary").mean().name.prefix("avg_"))
        .sort("department")
        .collect(),
    )


@pytest.mark.parametrize(
    "method_name",
    ["all", "sum", "mean", "median", "min", "max", "first", "last", "n_unique"],
)
def test_lazygroupby_simple_computations(
    sample_df: pl.DataFrame, method_name: str
) -> None:
    selected = ("department", "age", "salary")
    result: pl.DataFrame = (  # pyright: ignore[reportAny]
        getattr(  # pyright: ignore[reportAny]
            pql.LazyFrame(sample_df).select(*selected).group_by("department"),
            method_name,
        )()
        .sort("department")
        .collect()
    )
    expected: pl.DataFrame = (  # pyright: ignore[reportAny]
        getattr(  # pyright: ignore[reportAny]
            sample_df.lazy().select(*selected).group_by("department"), method_name
        )()
        .sort("department")
        .collect()
    )
    assert_eq(result, expected)


def test_len(sample_df: pl.DataFrame) -> None:
    selected = ("department", "age", "salary")
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .select(*selected)
        .group_by("department")
        .len()
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .select(*selected)
        .group_by("department")
        .len()
        .sort("department")
        .collect(),
    )
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .select(*selected)
        .group_by("department")
        .len(name="n_rows")
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .select(*selected)
        .group_by("department")
        .len(name="n_rows")
        .sort("department")
        .collect(),
    )


def test_quantile() -> None:

    qdf = pl.DataFrame({
        "department": ["A", "A", "A", "B", "B", "B"],
        "age": [10, 30, 50, 5, 25, 45],
        "salary": [100.0, 300.0, 500.0, 50.0, 250.0, 450.0],
    })
    assert_eq(
        pql
        .LazyFrame(qdf)
        .group_by("department")
        .quantile(0.5, interpolation=True)
        .sort("department")
        .collect(),
        qdf
        .lazy()
        .group_by("department")
        .quantile(0.5, interpolation="nearest")
        .sort("department")
        .collect(),
    )
    assert_eq(
        pql
        .LazyFrame(qdf)
        .group_by("department")
        .quantile(0.5, interpolation=False)
        .sort("department")
        .collect(),
        qdf
        .lazy()
        .group_by("department")
        .quantile(0.5, interpolation="equiprobable")
        .sort("department")
        .collect(),
    )


def test_agg_all_exclude(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            pql.all(exclude=["category"]),
            pql.col("category").str.join(pql.lit(", ")).alias("category"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl.all().exclude("category"),
            pl.col("category").str.join(", ").alias("category"),
        )
        .sort("department")
        .collect(),
    )


def test_agg_multi_key(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department", "sex")
        .agg(
            pql.col("salary").mean().alias("mean_salary"),
            pql.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
        sample_df
        .lazy()
        .group_by("department", "sex")
        .agg(
            pl.col("salary").mean().alias("mean_salary"),
            pl.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
    )


def test_agg_multi_exprs(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            pql.col("salary").mean().alias("mean_salary"),
            pql.col("salary").sum().alias("sum_salary"),
            pql.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl.col("salary").mean().alias("mean_salary"),
            pl.col("salary").sum().alias("sum_salary"),
            pl.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
    )


def test_agg_named_exprs(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            mean_salary=pql.col("salary").mean(),
            n=pql.col("id").count(),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            mean_salary=pl.col("salary").mean(),
            n=pl.col("id").count(),
        )
        .sort("department")
        .collect(),
    )


def test_drop_null_keys(sample_df: pl.DataFrame) -> None:
    # category has one null row — drop_null_keys must exclude it before grouping
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("category", drop_null_keys=True)
        .agg(pql.col("salary").mean().alias("mean_salary"))
        .sort("category")
        .collect(),
        sample_df
        .lazy()
        .filter(pl.col("category").is_not_null())
        .group_by("category")
        .agg(pl.col("salary").mean().alias("mean_salary"))
        .sort("category")
        .collect(),
    )


def test_agg_count_nulls(sample_df: pl.DataFrame) -> None:
    # count skips nulls (value has nulls); n_unique on null-free salary agrees across backends
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            pql.col("value").count().alias("n_values"),
            pql.col("salary").n_unique().alias("n_unique_salary"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl.col("value").count().alias("n_values"),
            pl.col("salary").n_unique().alias("n_unique_salary"),
        )
        .sort("department")
        .collect(),
    )


def test_agg_std_var(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            pql.col("salary").std().alias("std_salary"),
            pql.col("salary").var().alias("var_salary"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl.col("salary").std().alias("std_salary"),
            pl.col("salary").var().alias("var_salary"),
        )
        .sort("department")
        .collect(),
    )


def test_group_by_rollup() -> None:
    df = pl.DataFrame({"dept": ["A", "A", "B"], "val": [10, 20, 30]})
    result = (
        pql
        .LazyFrame(df)
        .group_by("dept", strategy="ROLLUP")
        .agg(pql.col("val").sum().alias("total"))
        .sort("dept", nulls_last=True)
        .collect()
    )
    # ROLLUP(dept): (A, 30), (B, 30), (NULL, 60)
    assert_eq(
        result,
        pl.DataFrame({"dept": ["A", "B", None], "total": [30, 30, 60]}),
    )


def test_group_by_cube() -> None:
    df = pl.DataFrame({
        "dept": ["A", "A", "B"],
        "cat": ["X", "X", "Y"],
        "val": [10, 20, 30],
    })
    result = (
        pql
        .LazyFrame(df)
        .group_by("dept", "cat", strategy="CUBE")
        .agg(pql.col("val").sum().alias("total"))
        .collect()
    )
    # CUBE(dept, cat): (A,X), (A,None), (B,Y), (B,None), (None,X), (None,Y), (None,None)
    assert result.height == 7
    assert (
        result.filter(pl.col("dept").is_null().and_(pl.col("cat").is_null())).height
        == 1
    )
    assert (
        result
        .filter(pl.col("dept").is_null().and_(pl.col("cat").is_null()))
        .get_column("total")
        .first()
        == 60
    )


def test_group_by_all_basic() -> None:
    df = pl.DataFrame({"dept": ["A", "A", "B"], "val": [10, 20, 30]})
    assert_eq(
        pql
        .LazyFrame(df)
        .group_by_all("dept", pql.col("val").sum().alias("total"))
        .sort("dept")
        .collect(),
        df
        .lazy()
        .group_by("dept")
        .agg(pl.col("val").sum().alias("total"))
        .sort("dept")
        .collect(),
    )


def test_group_by_all_multi_key(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by_all(
            "department",
            "sex",
            pql.col("salary").mean().alias("mean_salary"),
            pql.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
        sample_df
        .lazy()
        .group_by("department", "sex")
        .agg(
            pl.col("salary").mean().alias("mean_salary"),
            pl.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
    )


def test_group_by_all_multi_agg(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by_all(
            "department",
            pql.col("salary").mean().alias("avg_salary"),
            pql.col("salary").sum().alias("sum_salary"),
            pql.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl.col("salary").mean().alias("avg_salary"),
            pl.col("salary").sum().alias("sum_salary"),
            pl.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
    )


def test_group_by_all_named_exprs(sample_df: pl.DataFrame) -> None:
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by_all("department", mean_salary=pql.col("salary").mean())
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(mean_salary=pl.col("salary").mean())
        .sort("department")
        .collect(),
    )


def test_unique_exprs(sample_df: pl.DataFrame) -> None:
    dep = "department"
    assert_eq(
        pql
        .LazyFrame(sample_df)
        .group_by(dep)
        .agg(pql.col("sex").unique())
        .explode(pql.selectors.by_dtype(pql.List))
        .sort(dep)
        .collect(),
        sample_df
        .lazy()
        .group_by(dep)
        .agg(pl.col("sex").unique())
        .explode(pl.selectors.by_dtype(pl.List))
        .sort(dep)
        .collect(),
    )
