from collections.abc import Callable

import polars as pl
import pytest

import belugas as bl

from ._data import sample_bl, sample_lf
from ._utils import Fns, FnsCat, assert_eq, assert_lf_eq

bl_x = bl.col("x")
bl_age = bl.col("age")
pl_x = pl.col("x")
pl_age = pl.col("age")
data = {"a": [1, 2], "b": [3, 4]}
pl_lf = pl.LazyFrame(data)
bl_lf = bl.LazyFrame(pl_lf)
type FnsTup = tuple[Callable[[pl.Expr], pl.Expr], Callable[[bl.Expr], bl.Expr]]
_ALL_FNS: list[FnsTup] = [
    (pl.Expr.mean, bl.Expr.mean),
    (lambda c: c.add(1), lambda c: c.add(1)),
]


@pytest.mark.parametrize("fns", _ALL_FNS)
def test_all_add(fns: FnsTup) -> None:
    assert_lf_eq(
        pl_lf.select(pl.all().pipe(fns[0])),
        bl_lf.select(bl.all().pipe(fns[1])),
    )
    assert_lf_eq(
        pl_lf.with_columns(pl.all().pipe(fns[0])),
        bl_lf.with_columns(bl.all().pipe(fns[1])),
    )


def test_all_chained() -> None:
    assert_lf_eq(
        pl_lf.select(pl.all().mul(2).add(1)), bl_lf.select(bl.all().mul(2).add(1))
    )


_MULTI_FNS = FnsCat(
    (bl.count, pl.count),
    (bl.first, pl.first),
    (bl.last, pl.last),
    (bl.sum, pl.sum),
    (bl.mean, pl.mean),
    (bl.median, pl.median),
    (bl.min, pl.min),
    (bl.max, pl.max),
    (bl.sum_horizontal, pl.sum_horizontal),
    (bl.mean_horizontal, pl.mean_horizontal),
    (bl.coalesce, pl.coalesce),
)

_SIMPLE_FNS = FnsCat((bl.all, pl.all), (bl.len, pl.len))


@pytest.mark.parametrize("fns", _SIMPLE_FNS, ids=_SIMPLE_FNS.into_ids())
def test_simple_fn(fns: Fns) -> None:
    assert_eq(*fns.call())


@pytest.mark.parametrize("fns", _MULTI_FNS, ids=_MULTI_FNS.into_ids())
def test_multi_col(fns: Fns) -> None:
    assert_eq(*fns.call("x", "n"))


_NULL_PROP_FNS = FnsCat(
    (bl.min_horizontal, pl.min_horizontal), (bl.max_horizontal, pl.max_horizontal)
)

_STD_VAR_FNS = FnsCat((bl.std, pl.std), (bl.var, pl.var))
_N_UNIQUE_FNS = FnsCat(
    (bl.approx_n_unique, pl.approx_n_unique),
    (bl.n_unique, pl.n_unique),
)


@pytest.mark.parametrize("fns", _NULL_PROP_FNS, ids=_NULL_PROP_FNS.into_ids())
def test_horizontal_minmax_propagates_null(fns: Fns) -> None:
    """DuckDB `LEAST`/`GREATEST` propagate NULL, unlike Polars which ignores them.

    We drop nulls before testing to get identical results.
    """
    bl_expr, pl_expr = fns.call("x", "n")
    assert_lf_eq(
        sample_lf().drop_nulls("n").select(pl_expr),
        sample_bl().drop_nulls("n").select(bl_expr),
    )


def test_all_horizontal() -> None:
    assert_eq(bl.all_horizontal("a", "b"), pl.all_horizontal("a", "b"))


def test_any_horizontal() -> None:
    assert_eq(bl.any_horizontal("a", "b"), pl.any_horizontal("a", "b"))


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_any(ignore_nulls: bool) -> None:
    assert_eq(
        bl.any("a", "b", ignore_nulls=ignore_nulls),
        pl.any("a", "b", ignore_nulls=ignore_nulls),
    )


def test_arctan2() -> None:
    assert_eq(bl.arctan2("x", "n"), pl.arctan2("x", "n"))


@pytest.mark.parametrize("fns", _N_UNIQUE_FNS, ids=_N_UNIQUE_FNS.into_ids())
def test_n_unique_family(fns: Fns) -> None:
    assert_eq(*fns.call("x"))


@pytest.mark.parametrize("reverse", [False, True])
def test_cum_count(reverse: bool) -> None:
    assert_eq(
        bl.cum_count("x", reverse=reverse),
        pl.cum_count("x", reverse=reverse),
    )


def test_cum_sum() -> None:
    assert_eq(bl.cum_sum("x"), pl.cum_sum("x"))


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("fns", _STD_VAR_FNS, ids=_STD_VAR_FNS.into_ids())
def test_std_var(fns: Fns, ddof: int) -> None:
    assert_eq(*fns.call("x", ddof=ddof))


def test_when_then_simple() -> None:
    bl_expr = (
        bl
        .when(bl_x.eq(5))
        .then(bl.lit("equal_to_5"))
        .otherwise(bl.lit("not_equal_to_5"))
    )
    pl_expr = (
        pl
        .when(pl_x.eq(5))
        .then(pl.lit("equal_to_5"))
        .otherwise(pl.lit("not_equal_to_5"))
    )
    assert_eq(bl_expr, pl_expr)


def test_when_then_chained() -> None:
    bl_expr = (
        bl
        .when(bl_x.gt(5))
        .then(bl.lit("high"))
        .when(bl_x.lt(5))
        .then(bl.lit("low"))
        .when(bl_x.eq(5))
        .then(bl.lit("equal"))
        .otherwise(bl.lit("mid"))
    )
    pl_expr = (
        pl
        .when(pl_x.gt(5))
        .then(pl.lit("high"))
        .when(pl_x.lt(5))
        .then(pl.lit("low"))
        .when(pl_x.eq(5))
        .then(pl.lit("equal"))
        .otherwise(pl.lit("mid"))
    )
    assert_eq(bl_expr, pl_expr)


def test_when_with_multiple_predicates() -> None:
    bl_expr = (
        bl
        .when(bl.col("a"), bl.col("b"))
        .then(bl.lit("both_true"))
        .otherwise(bl.lit("not_both_true"))
    )
    pl_expr = (
        pl
        .when(pl.col("a"), (pl.col("b")))
        .then(pl.lit("both_true"))
        .otherwise(pl.lit("not_both_true"))
    )
    assert_eq(bl_expr, pl_expr)


def test_when_without_otherwise() -> None:
    bl_expr = bl.when(bl_x.gt(10)).then(bl.lit("high"))
    pl_expr = pl.when(pl_x.gt(10)).then(pl.lit("high"))
    assert_eq(bl_expr, pl_expr)


def test_when_nested_conditions() -> None:
    bl_expr = (
        bl
        .when(bl_x.gt(15))
        .then(
            bl
            .when(bl_age.gt(30))
            .then(bl.lit("x_high_age_high"))
            .otherwise(bl.lit("x_high_age_low"))
        )
        .otherwise(bl.lit("x_low"))
    )
    pl_expr = (
        pl
        .when(pl_x.gt(15))
        .then(
            pl
            .when(pl_age.gt(30))
            .then(pl.lit("x_high_age_high"))
            .otherwise(pl.lit("x_high_age_low"))
        )
        .otherwise(pl.lit("x_low"))
    )
    assert_eq(bl_expr, pl_expr)
