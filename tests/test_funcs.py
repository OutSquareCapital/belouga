import narwhals as nw
import polars as pl
import pytest

import pql

from ._utils import Fns, FnsCat, assert_eq, assert_eq_pl, assert_lf_eq_pl


def test_all_add() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    assert_lf_eq_pl(
        pql.LazyFrame(data).select(pql.all().add(1)),
        pl.LazyFrame(data).select(pl.all().add(1)),
    )


def test_all_chained() -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    assert_lf_eq_pl(
        pql.LazyFrame(data).select(pql.all().mul(2).add(1)),
        pl.LazyFrame(data).select(pl.all().mul(2).add(1)),
    )


_MULTI_FNS = FnsCat(
    (pql.sum, pl.sum),
    (pql.mean, pl.mean),
    (pql.median, pl.median),
    (pql.min, pl.min),
    (pql.max, pl.max),
    (pql.sum_horizontal, pl.sum_horizontal),
    (pql.min_horizontal, pl.min_horizontal),
    (pql.max_horizontal, pl.max_horizontal),
    (pql.mean_horizontal, pl.mean_horizontal),
    (pql.coalesce, pl.coalesce),
)

_SIMPLE_FNS = FnsCat((pql.all, pl.all), (pql.len, pl.len))


@pytest.mark.parametrize("fns", _SIMPLE_FNS, ids=_SIMPLE_FNS.into_ids())
def test_simple_fn(fns: Fns) -> None:
    assert_eq_pl(*fns.call())


@pytest.mark.parametrize("fns", _MULTI_FNS, ids=_MULTI_FNS.into_ids())
def test_multi_col(fns: Fns) -> None:
    assert_eq_pl(*fns.call("x", "n"))


def test_all_horizontal() -> None:
    assert_eq(
        pql.all_horizontal("a", "b"), nw.all_horizontal("a", "b", ignore_nulls=False)
    )


def test_any_horizontal() -> None:
    assert_eq(
        pql.any_horizontal("a", "b"), nw.any_horizontal("a", "b", ignore_nulls=False)
    )


def test_when_then_simple() -> None:
    pql_expr = (
        pql.when(pql.col("x").eq(5))
        .then(pql.lit("equal_to_5"))
        .otherwise(pql.lit("not_equal_to_5"))
    )
    pl_expr = (
        pl.when(pl.col("x").eq(5))
        .then(pl.lit("equal_to_5"))
        .otherwise(pl.lit("not_equal_to_5"))
    )
    assert_eq_pl(pql_expr, pl_expr)


def test_when_then_chained() -> None:
    pql_expr = (
        pql.when(pql.col("x") > 5)
        .then(pql.lit("high"))
        .when(pql.col("x") < 5)
        .then(pql.lit("low"))
        .when(pql.col("x") == 5)
        .then(pql.lit("equal"))
        .otherwise(pql.lit("mid"))
    )
    pl_expr = (
        pl.when(pl.col("x") > 5)
        .then(pl.lit("high"))
        .when(pl.col("x") < 5)
        .then(pl.lit("low"))
        .when(pl.col("x") == 5)
        .then(pl.lit("equal"))
        .otherwise(pl.lit("mid"))
    )
    assert_eq_pl(pql_expr, pl_expr)


def test_when_with_multiple_predicates() -> None:
    pql_expr = (
        pql.when(pql.col("a"), pql.col("b"))
        .then(pql.lit("both_true"))
        .otherwise(pql.lit("not_both_true"))
    )
    pl_expr = (
        pl.when(pl.col("a") & pl.col("b"))
        .then(pl.lit("both_true"))
        .otherwise(pl.lit("not_both_true"))
    )
    assert_eq_pl(pql_expr, pl_expr)


def test_when_without_otherwise() -> None:
    pql_expr = pql.when(pql.col("x") > 10).then(pql.lit("high"))
    pl_expr = pl.when(pl.col("x") > 10).then(pl.lit("high"))
    assert_eq_pl(pql_expr, pl_expr)


def test_when_nested_conditions() -> None:
    pql_expr = (
        pql.when(pql.col("x") > 15)
        .then(
            pql.when(pql.col("age") > 30)
            .then(pql.lit("x_high_age_high"))
            .otherwise(pql.lit("x_high_age_low"))
        )
        .otherwise(pql.lit("x_low"))
    )
    pl_expr = (
        pl.when(pl.col("x") > 15)
        .then(
            pl.when(pl.col("age") > 30)
            .then(pl.lit("x_high_age_high"))
            .otherwise(pl.lit("x_high_age_low"))
        )
        .otherwise(pl.lit("x_low"))
    )
    assert_eq_pl(pql_expr, pl_expr)
