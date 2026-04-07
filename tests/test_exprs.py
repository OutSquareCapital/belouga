from collections.abc import Callable
from typing import cast

import polars as pl
import pytest

import pql
from pql import _typing as t  # noqa: PLC2701

from ._utils import assert_eq, on_simple_fn

desc_param = pytest.mark.parametrize("descending", [True, False])


def test_rand() -> None:
    assert_eq((True & pql.col("a")), (True & pl.col("a")))


def test_ror() -> None:
    assert_eq((False | pql.col("a")), (False | pl.col("a")))


def test_hash() -> None:
    assert hash(pql.col("x")) == hash(pql.col("x"))


def test_xor() -> None:
    assert_eq(pql.col("x").xor(3), pl.col("x").xor(3))
    assert_eq(pql.col("x").__xor__(3), pl.col("x").xor(3))
    assert_eq(
        pql.col("x").xor(pql.col("n")),
        pl.col("x").xor(pl.col("n")),
    )


@pytest.mark.parametrize("by", ["age", "n", 2])
def test_repeat_by(by: int | str) -> None:
    assert_eq(pql.col("x").repeat_by(by), pl.col("x").repeat_by(by))


def test_mul() -> None:
    assert_eq(pql.col("x").mul(5), pl.col("x").__mul__(5))
    assert_eq(pql.col("salary").mul(2), pl.col("salary").__mul__(2))


def test_truediv() -> None:
    assert_eq((pql.col("x") / 5), (pl.col("x") / 5))

    assert_eq(pql.col("salary").truediv(1000), pl.col("salary").__truediv__(1000))


def test_replace() -> None:
    assert_eq(pql.col("x").replace(2, 99), pl.col("x").replace(2, 99))


def test_repr() -> None:
    assert "Expr" in repr(pql.col("name"))


def test_sql_list_sort_uses_array_sort_constructor() -> None:
    list_sort = pql.sql.col("arr").list.sort().inner()
    array_sort = pql.sql.col("arr").arr.sort().inner()

    assert type(list_sort) is type(array_sort)


def test_and() -> None:
    assert_eq((pql.col("a") & pql.col("b")), (pl.col("a") & pl.col("b")))


def test_or() -> None:
    assert_eq((pql.col("a") | pql.col("b")), (pl.col("a") | pl.col("b")))


def test_not() -> None:
    assert_eq((~pql.col("a")), (~pl.col("a")))


def test_radd() -> None:
    assert_eq((10 + pql.col("x")), (10 + pl.col("x")))


def test_rmul() -> None:
    assert_eq((10 * pql.col("x")), (10 * pl.col("x")))


def test_rtruediv() -> None:
    assert_eq((10 / pql.col("x")), (10 / pl.col("x")))


def test_eq() -> None:
    assert_eq((pql.col("x") == 2), (pl.col("x") == 2))


def test_lt() -> None:
    assert_eq((pql.col("x") < 3), (pl.col("x") < 3))


def test_gt() -> None:
    assert_eq((pql.col("x") > 2), (pl.col("x") > 2))


def test_ge() -> None:
    assert_eq((pql.col("x") >= 3), (pl.col("x") >= 3))


def test_rsub() -> None:
    assert_eq((10 - pql.col("x")), (10 - pl.col("x")))


def test_rfloordiv() -> None:
    assert_eq((10 // pql.col("x")), (10 // pl.col("x")))


def test_rmod() -> None:
    assert_eq((10 % pql.col("x")), (10 % pl.col("x")))


def test_rpow() -> None:
    assert_eq((2 ** pql.col("x")), (2 ** pl.col("x")))


def test_neg() -> None:
    assert_eq((-pql.col("x")), (-pl.col("x")))
    assert_eq(pql.col("x").neg(), pl.col("x").neg())


def test_ne() -> None:
    assert_eq((pql.col("x") != 2), (pl.col("x") != 2))


def test_le() -> None:
    assert_eq((pql.col("x") <= 2), (pl.col("x") <= 2))


def test_sub() -> None:
    assert_eq((pql.col("x") - 5), (pl.col("x") - 5))


def test_floordiv() -> None:
    assert_eq((pql.col("x") // 3), (pl.col("x") // 3))


_SIMPLE_FNS = {
    "sinh",
    "cosh",
    "tanh",
    "is_finite",
    "is_infinite",
    "count",
    "len",
    "min",
    "max",
    "sum",
    "mean",
    "median",
    "mode",
    "product",
    "n_unique",
    "null_count",
    "has_nulls",
    "cot",
    "degrees",
    "radians",
    "sign",
    "floor",
    "ceil",
    "cbrt",
    "abs",
    "approx_n_unique",
    "is_last_distinct",
    "exp",
    "sin",
    "cos",
    "tan",
    "arctan",
    "arccosh",
    "arcsinh",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "diff",
}


@pytest.mark.parametrize("fn", _SIMPLE_FNS)
def test_simple_methods_on_x(fn: str) -> None:
    on_simple_fn(pql.col("x"), pl.col("x"), fn)


@pytest.mark.parametrize("fn", ["null_count", "has_nulls"])
def test_simple_methods_on_age(fn: str) -> None:
    on_simple_fn(pql.col("age"), pl.col("age"), fn)


def test_uint_only_simple() -> None:
    assert_eq(pql.col("uint").log(2), pl.col("uint").log(2))
    assert_eq(pql.col("uint").log10(), pl.col("uint").log10())
    assert_eq(pql.col("uint").log1p(), pl.col("uint").log1p())
    assert_eq(pql.col("uint").sqrt(), pl.col("uint").sqrt())


def test_is_first_distinct() -> None:
    assert_eq(pql.col("a").is_first_distinct(), pl.col("a").is_first_distinct())


def test_col_getattr() -> None:
    assert_eq(pql.col.a, pl.col.a)


@pytest.mark.parametrize("mode", pql.sql.typing.RoundMode.__args__)
def test_round(mode: pql.sql.typing.RoundMode) -> None:
    assert_eq(
        pql.col("float_vals").round(2, mode=mode),
        pl.col("float_vals").round(2, mode=mode),
    )


def test_pipe() -> None:
    assert_eq(pql.col("x").pipe(lambda x: x * 2), pl.col("x").pipe(lambda x: x * 2))


def test_forward_fill() -> None:
    assert_eq(pql.col("a").forward_fill(), pl.col("a").forward_fill())


@pytest.mark.parametrize("limit", [1, 2, None])
def test_backward_fill(limit: int | None) -> None:
    assert_eq(pql.col("n").backward_fill(limit), pl.col("n").backward_fill(limit))


def test_is_nan() -> None:
    assert_eq(pql.col("nan_vals").is_nan(), pl.col("nan_vals").is_nan())


def test_is_null() -> None:
    assert_eq(pql.col("n").is_null(), pl.col("n").is_null())


def test_is_not_null() -> None:
    assert_eq(pql.col("n").is_not_null(), pl.col("n").is_not_null())


def test_is_not_nan() -> None:
    assert_eq(pql.col("nan_vals").is_not_nan(), pl.col("nan_vals").is_not_nan())


def test_fill_nan() -> None:
    assert_eq(pql.col("nan_vals").fill_nan(0.0), pl.col("nan_vals").fill_nan(0.0))


def test_is_duplicated() -> None:
    assert_eq(pql.col("a").is_duplicated(), pl.col("a").is_duplicated())


def test_arccos() -> None:
    assert_eq(
        pql.col("x").truediv(20).arccos(),
        pl.col("x").truediv(20).arccos(),
    )


def test_arcsin() -> None:
    assert_eq(
        pql.col("x").truediv(20).arcsin(),
        pl.col("x").truediv(20).arcsin(),
    )


def test_arctanh() -> None:
    assert_eq(
        pql.col("x").truediv(30).arctanh(),
        pl.col("x").truediv(30).arctanh(),
    )


def test_pow() -> None:
    assert_eq(pql.col("x").pow(2), pl.col("x").__pow__(2))
    assert_eq(pql.col("x").__pow__(2), pl.col("x").__pow__(2))


def test_add() -> None:
    assert_eq(pql.col("age").add(10), pl.col("age").__add__(10))
    assert_eq(pql.col("age").__add__(10), pl.col("age").__add__(10))


def test_mod() -> None:
    assert_eq(pql.col("age").__mod__(10), pl.col("age").__mod__(10))
    assert_eq(pql.col("age").mod(10), pl.col("age").__mod__(10))


def test_is_in() -> None:
    assert_eq(pql.col("x").is_in([2, 3]), pl.col("x").is_in([2, 3]))


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_shift(n: int) -> None:
    assert_eq(pql.col("x").shift(n), pl.col("x").shift(n))


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_pct_change(n: int) -> None:
    assert_eq(pql.col("x").pct_change(n), pl.col("x").pct_change(n))


@pytest.mark.parametrize("closed", pql.sql.typing.ClosedInterval.__args__)
def test_is_between(closed: pql.sql.typing.ClosedInterval) -> None:
    assert_eq(
        pql.col("x").is_between(2, 10, closed=closed),
        pl.col("x").is_between(2, 10, closed=closed),
    )


def test_is_unique() -> None:
    assert_eq(pql.col("a").is_unique(), pl.col("a").is_unique())


@pytest.mark.parametrize("ignore_nulls", [True, False])
def test_first(ignore_nulls: bool) -> None:
    assert_eq(
        pql.col("n").first(ignore_nulls=ignore_nulls),
        pl.col("n").first(ignore_nulls=ignore_nulls),
    )


def test_last() -> None:
    assert_eq(pql.col("n").last(), pl.col("n").last())


def test_max_by() -> None:
    assert_eq(pql.col("x").max_by("age"), pl.col("x").max_by("age"))
    assert_eq(
        pql.col("salary").max_by(pql.col("x").neg()),
        pl.col("salary").max_by(pl.col("x").neg()),
    )


def test_min_by() -> None:
    assert_eq(pql.col("x").min_by("age"), pl.col("x").min_by("age"))
    assert_eq(
        pql.col("salary").min_by(pql.col("x").neg()),
        pl.col("salary").min_by(pl.col("x").neg()),
    )


def test_implode() -> None:
    assert_eq(pql.col("x").implode(), pl.col("x").implode())


def test_unique() -> None:
    assert_eq(pql.col("x").unique(), pl.col("x").unique(), with_cols=False)
    assert_eq(
        (pql.col("x").unique(), pql.col("x").unique().add(3).alias("x_unique_right")),
        (pl.col("x").unique(), pl.col("x").unique().add(3).alias("x_unique_right")),
        with_cols=False,
    )


def test_is_close() -> None:
    assert_eq(
        pql.col("salary").is_close(
            pql.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0
        ),
        pl.col("salary").is_close(
            pl.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0
        ),
    )
    assert_eq(
        pql.col("salary").is_close(
            pql.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0, nans_equal=True
        ),
        pl.col("salary").is_close(
            pl.col("salary").add(0.001), abs_tol=0.01, rel_tol=0.0, nans_equal=True
        ),
    )


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("window_size", [2, 4])
@pytest.mark.parametrize("min_samples", [None, 1, 2])
@pytest.mark.parametrize(
    "method",
    ["rolling_mean", "rolling_sum", "rolling_min", "rolling_max", "rolling_median"],
)
def test_rolling(
    method: str, window_size: int, min_samples: int | None, center: bool
) -> None:
    pql_fn = cast(Callable[..., pql.Expr], getattr(pql.col("x"), method))
    pl_fn = cast(Callable[..., pl.Expr], getattr(pl.col("x"), method))
    assert_eq(
        pql_fn(window_size=window_size, min_samples=min_samples, center=center),
        pl_fn(window_size=window_size, min_samples=min_samples, center=center),
    )


def test_rolling_std() -> None:
    assert_eq(
        pql.col("x").rolling_std(window_size=3, min_samples=2, center=False, ddof=1),
        pl.col("x").rolling_std(window_size=3, min_samples=2, center=False, ddof=1),
    )


def test_rolling_var() -> None:
    assert_eq(
        pql.col("x").rolling_var(window_size=3, min_samples=2, center=False, ddof=1),
        pl.col("x").rolling_var(window_size=3, min_samples=2, center=False, ddof=1),
    )


@pytest.mark.parametrize("lower_bound", [None, 2, 10])
@pytest.mark.parametrize("upper_bound", [None, 10, 20])
def test_clip(lower_bound: int | None, upper_bound: int | None) -> None:
    assert_eq(
        pql.col("x").clip(lower_bound=lower_bound, upper_bound=upper_bound),
        pl.col("x").clip(lower_bound=lower_bound, upper_bound=upper_bound),
    )


bias_arg = pytest.mark.parametrize("bias", [True, False])


@pytest.mark.parametrize("fisher", [True, False])
@bias_arg
def test_kurtosis(fisher: bool, bias: bool) -> None:
    assert_eq(
        pql.col("x").kurtosis(fisher=fisher, bias=bias),
        pl.col("x").kurtosis(fisher=fisher, bias=bias),
    )


@bias_arg
def test_skew(bias: bool) -> None:
    assert_eq(pql.col("x").skew(bias=bias), pl.col("x").skew(bias=bias))


def test_quantile() -> None:
    assert_eq(
        pql.col("x").quantile(0.75, interpolation=True),
        pl.col("x").quantile(0.75, "linear"),
    )
    assert_eq(
        pql.col("x").quantile(0.75, interpolation=False),
        pl.col("x").quantile(0.75, "equiprobable"),
    )


@desc_param
@pytest.mark.parametrize("order_by", ["n", None])
def test_over(order_by: str | None, descending: bool) -> None:
    assert_eq(
        pql.col("x").sum().over("a", order_by=order_by, descending=descending),
        pl.col("x").sum().over("a", order_by=order_by, descending=descending),
    )


def test_over_nested() -> None:
    assert_eq(
        pql.col("x").rolling_max(2).over("a"), pl.col("x").rolling_max(2).over("a")
    )


@pytest.mark.parametrize("nulls_last", [True, False])
def test_over_with_nulls_last(*, nulls_last: bool) -> None:
    assert_eq(
        pql.col("n").first().over("a", order_by="x", nulls_last=nulls_last),
        pl.col("n").first().over("a", order_by="x", nulls_last=nulls_last),
    )


@pytest.mark.parametrize("strategy", pql.sql.typing.FillNullStrategy.__args__)
def test_fill_null(strategy: pql.sql.typing.FillNullStrategy) -> None:
    assert_eq(pql.col("age").fill_null(0), pl.col("age").fill_null(0))
    assert_eq(
        pql.col("age").fill_null(strategy=strategy),
        pl.col("age").fill_null(strategy=strategy),
    )


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_fill_null_limit(strategy: pql.sql.typing.FillNullStrategy, limit: int) -> None:
    assert_eq(
        pql.col("age").fill_null(strategy=strategy, limit=limit),
        pl.col("age").fill_null(strategy=strategy, limit=limit),
    )


def test_fill_null_no_value_or_strategy() -> None:
    msg = "must specify either a fill `value` or `strategy`"
    with pytest.raises(ValueError, match=msg):
        _ = pql.col("age").fill_null()
    with pytest.raises(ValueError, match=msg):
        _ = pl.col("age").fill_null()


def test_fill_null_limit_negative() -> None:
    with pytest.raises(
        ValueError, match="Can't process negative `limit` value for fill_null"
    ):
        _ = pql.col("age").fill_null(strategy="forward", limit=-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = pl.col("age").fill_null(strategy="forward", limit=-1)


def test_fill_null_limit_invalid_strategy() -> None:
    err = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
    with pytest.raises(ValueError, match=err):
        _ = pql.col("age").fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pl.col("age").fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pql.col("age").fill_null(0, limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pl.col("age").fill_null(0, limit=1)


def test_fill_val_and_strat() -> None:
    err = "cannot specify both `value` and `strategy`"
    with pytest.raises(ValueError, match=err):
        _ = pql.col("age").fill_null(value=0, strategy="min")
    with pytest.raises(ValueError, match=err):
        _ = pl.col("age").fill_null(value=0, strategy="min")


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_and_var(ddof: int) -> None:
    assert_eq(pql.col("x").std(ddof=ddof), pl.col("x").std(ddof=ddof))
    assert_eq(pql.col("x").var(ddof=ddof), pl.col("x").var(ddof=ddof))


def test_all() -> None:
    assert_eq(pql.col("x").gt(0).all(), pl.col("x").gt(0).all())


def test_any() -> None:
    assert_eq(pql.col("x").gt(10).any(), pl.col("x").gt(10).any())


@desc_param
@pytest.mark.parametrize("method", t.RankMethod.__args__)
def test_rank(method: t.RankMethod, descending: bool) -> None:
    assert_eq(
        pql.col("x").rank(method, descending=descending),
        pl.col("x").rank(method, descending=descending),
    )


def test_cum_count() -> None:
    assert_eq(pql.col("x").cum_count(), pl.col("x").cum_count())
    assert_eq(pql.col("x").cum_count(reverse=True), pl.col("x").cum_count(reverse=True))


def test_cum_sum() -> None:
    assert_eq(pql.col("x").cum_sum(), pl.col("x").cum_sum())
    assert_eq(pql.col("x").cum_sum(reverse=True), pl.col("x").cum_sum(reverse=True))


def test_cum_prod() -> None:
    assert_eq(pql.col("x").cum_prod(), pl.col("x").cum_prod())
    assert_eq(pql.col("x").cum_prod(reverse=True), pl.col("x").cum_prod(reverse=True))


def test_cum_min() -> None:
    assert_eq(pql.col("x").cum_min(), pl.col("x").cum_min())
    assert_eq(pql.col("x").cum_min(reverse=True), pl.col("x").cum_min(reverse=True))


def test_cum_max() -> None:
    assert_eq(pql.col("x").cum_max(), pl.col("x").cum_max())
    assert_eq(pql.col("x").cum_max(reverse=True), pl.col("x").cum_max(reverse=True))


@desc_param
@pytest.mark.parametrize("nulls_last", [True, False])
def test_arg_sort(descending: bool, nulls_last: bool) -> None:
    assert_eq(
        pql.col("x").arg_sort(descending=descending, nulls_last=nulls_last),
        pl.col("x").arg_sort(descending=descending, nulls_last=nulls_last),
    )


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_truncate(decimals: int) -> None:
    assert_eq(
        pql.col("float_vals").truncate(decimals),
        pl.col("float_vals").truncate(decimals),
    )


def test_dot() -> None:
    assert_eq(
        pql.col("x").dot("age"),
        pl.col("x").dot("age"),
    )


@pytest.mark.parametrize("base", [1, 2, 3])
@pytest.mark.parametrize("normalize", [True, False])
def test_entropy(base: int, *, normalize: bool) -> None:
    assert_eq(
        pql.col("x").abs().cast(pql.Float64()).entropy(base=base, normalize=normalize),
        pl.col("x").abs().cast(pl.Float64).entropy(base=base, normalize=normalize),
    )
