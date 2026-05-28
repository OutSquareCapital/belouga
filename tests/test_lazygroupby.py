from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars.lazyframe.group_by import LazyGroupBy as PlLazyGroupBy
from polars.testing import assert_frame_equal
from pyochain import Seq

import belugas as bl
from belugas._groupby import LazyGroupBy  # noqa: PLC2701

from ._utils import into_ids

if TYPE_CHECKING:
    from collections.abc import Callable

bl_salary = bl.col("salary")
pl_salary = pl.col("salary")
assert_eq = partial(assert_frame_equal, check_dtypes=False, check_row_order=False)

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


def test_agg(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(bl_salary.mean())
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(pl_salary.mean())
        .sort("department")
        .collect(),
    )


def test_agg_by_prefix(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(bl_salary.mean().name.prefix("avg_"))
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(pl_salary.mean().name.prefix("avg_"))
        .sort("department")
        .collect(),
    )


_GROUP_BY_METHODS = Seq((
    (LazyGroupBy.all, PlLazyGroupBy.all),
    (LazyGroupBy.sum, PlLazyGroupBy.sum),
    (LazyGroupBy.mean, PlLazyGroupBy.mean),
    (LazyGroupBy.median, PlLazyGroupBy.median),
    (LazyGroupBy.min, PlLazyGroupBy.min),
    (LazyGroupBy.max, PlLazyGroupBy.max),
    (LazyGroupBy.first, PlLazyGroupBy.first),
    (LazyGroupBy.last, PlLazyGroupBy.last),
    (LazyGroupBy.n_unique, PlLazyGroupBy.n_unique),
))


@pytest.mark.parametrize("fns", _GROUP_BY_METHODS, ids=_GROUP_BY_METHODS.into(into_ids))
def test_lazygroupby_simple_computations(
    sample_df: pl.DataFrame,
    fns: tuple[
        Callable[[LazyGroupBy], bl.LazyFrame], Callable[[PlLazyGroupBy], pl.LazyFrame]
    ],
) -> None:
    selected = ("department", "age", "salary")
    result = (
        (fns[0](bl.LazyFrame(sample_df).select(*selected).group_by("department")))
        .sort("department")
        .collect()
    )
    expected = (
        fns[1](sample_df.lazy().select(*selected).group_by("department"))
        .sort("department")
        .collect()
    )
    assert_eq(result, expected)


def test_len(sample_df: pl.DataFrame) -> None:
    selected = ("department", "age", "salary")
    assert_eq(
        bl
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
        bl
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
        bl
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
        bl
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
    sep = ", "
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(bl.all(exclude="category"), bl.col("category").str.join(sep))
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(pl.all().exclude("category"), pl.col("category").str.join(sep))
        .sort("department")
        .collect(),
    )


def test_agg_multi_key(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department", "sex")
        .agg(
            bl_salary.mean().alias("mean_salary"),
            bl.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
        sample_df
        .lazy()
        .group_by("department", "sex")
        .agg(
            pl_salary.mean().alias("mean_salary"),
            pl.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
    )


def test_agg_multi_exprs(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            bl_salary.mean().alias("mean_salary"),
            bl_salary.sum().alias("sum_salary"),
            bl.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl_salary.mean().alias("mean_salary"),
            pl_salary.sum().alias("sum_salary"),
            pl.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
    )


def test_agg_composed_reducer(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(bl_salary.mean().add(1).alias("mean_plus"))
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(pl_salary.mean().add(1).alias("mean_plus"))
        .sort("department")
        .collect(),
    )


def test_agg_named_exprs(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            mean_salary=bl_salary.mean(),
            n=bl.col("id").count(),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            mean_salary=pl_salary.mean(),
            n=pl.col("id").count(),
        )
        .sort("department")
        .collect(),
    )


def test_drop_null_keys(sample_df: pl.DataFrame) -> None:
    # category has one null row — drop_null_keys must exclude it before grouping
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("category", drop_null_keys=True)
        .agg(bl_salary.mean().alias("mean_salary"))
        .sort("category")
        .collect(),
        sample_df
        .lazy()
        .filter(pl.col("category").is_not_null())
        .group_by("category")
        .agg(pl_salary.mean().alias("mean_salary"))
        .sort("category")
        .collect(),
    )


def test_agg_count_nulls(sample_df: pl.DataFrame) -> None:
    # count skips nulls (value has nulls); n_unique on null-free salary agrees across backends
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            bl.col("value").count().alias("n_values"),
            bl_salary.n_unique().alias("n_unique_salary"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl.col("value").count().alias("n_values"),
            pl_salary.n_unique().alias("n_unique_salary"),
        )
        .sort("department")
        .collect(),
    )


def test_agg_std_var(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by("department")
        .agg(
            bl_salary.std().alias("std_salary"),
            bl_salary.var().alias("var_salary"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl_salary.std().alias("std_salary"),
            pl_salary.var().alias("var_salary"),
        )
        .sort("department")
        .collect(),
    )


def test_group_by_rollup() -> None:
    df = pl.DataFrame({"dept": ["A", "A", "B"], "val": [10, 20, 30]})
    result = (
        bl
        .LazyFrame(df)
        .group_by("dept", strategy="ROLLUP")
        .agg(bl.col("val").sum().alias("total"))
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
        bl
        .LazyFrame(df)
        .group_by("dept", "cat", strategy="CUBE")
        .agg(bl.col("val").sum().alias("total"))
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
        bl
        .LazyFrame(df)
        .group_by_all("dept", bl.col("val").sum().alias("total"))
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
        bl
        .LazyFrame(sample_df)
        .group_by_all(
            "department",
            "sex",
            bl_salary.mean().alias("mean_salary"),
            bl.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
        sample_df
        .lazy()
        .group_by("department", "sex")
        .agg(
            pl_salary.mean().alias("mean_salary"),
            pl.col("age").max().alias("max_age"),
        )
        .sort("department", "sex")
        .collect(),
    )


def test_group_by_all_multi_agg(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by_all(
            "department",
            bl_salary.mean().alias("avg_salary"),
            bl_salary.sum().alias("sum_salary"),
            bl.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(
            pl_salary.mean().alias("avg_salary"),
            pl_salary.sum().alias("sum_salary"),
            pl.col("id").count().alias("n"),
        )
        .sort("department")
        .collect(),
    )


def test_group_by_all_named_exprs(sample_df: pl.DataFrame) -> None:
    assert_eq(
        bl
        .LazyFrame(sample_df)
        .group_by_all("department", mean_salary=bl_salary.mean())
        .sort("department")
        .collect(),
        sample_df
        .lazy()
        .group_by("department")
        .agg(mean_salary=pl_salary.mean())
        .sort("department")
        .collect(),
    )


def test_unique_exprs(sample_df: pl.DataFrame) -> None:
    dep = "department"
    assert_eq(
        sample_df
        .lazy()
        .group_by(dep)
        .agg(pl.col("sex").unique())
        .explode(pl.selectors.by_dtype(pl.List))
        .sort(dep)
        .collect(),
        bl
        .LazyFrame(sample_df)
        .group_by(dep)
        .agg(bl.col("sex").unique())
        .explode(bl.selectors.by_dtype(bl.List))
        .sort(dep)
        .collect(),
    )


def test_selectors(sample_df: pl.DataFrame) -> None:
    dep = "department"
    assert_eq(
        sample_df
        .lazy()
        .group_by(dep)
        .agg(pl.selectors.numeric().mean())
        .sort(dep)
        .collect(),
        bl
        .LazyFrame(sample_df)
        .group_by(dep)
        .agg(bl.selectors.numeric().mean())
        .sort(dep)
        .collect(),
    )
