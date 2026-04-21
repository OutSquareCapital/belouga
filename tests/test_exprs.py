from collections.abc import Callable
from typing import Protocol

import polars as pl
import pytest
from pyochain import Iter, Seq

import pql

from ._utils import assert_eq

pql_a = pql.col("a")
pql_x = pql.col("x")
pql_n = pql.col("n")
pql_salary = pql.col("salary")
pql_uint = pql.col("uint")
pql_age = pql.col("age")
pql_nan_vals = pql.col("nan_vals")
pql_float_vals = pql.col("float_vals")
pl_a = pl.col("a")
pl_x = pl.col("x")
pl_n = pl.col("n")
pl_salary = pl.col("salary")
pl_uint = pl.col("uint")
pl_age = pl.col("age")
pl_nan_vals = pl.col("nan_vals")
pl_float_vals = pl.col("float_vals")
desc_param = pytest.mark.parametrize("descending", [True, False])
type Fns = tuple[Callable[[], pql.Expr], Callable[[], pl.Expr]]


def _ids(fns: Seq[tuple[Callable[..., pql.Expr], Callable[..., pl.Expr]]]) -> Iter[str]:
    return fns.iter().map_star(lambda f1, _f2: f1.__name__)


def test_rand() -> None:
    assert_eq(pql_a.__rand__(other=True), pl_a.__rand__(other=True))


def test_ror() -> None:
    assert_eq(pql_a.__ror__(other=False), pl_a.__ror__(other=False))


def test_hash() -> None:
    assert hash(pql_x) == hash(pql_x)


def test_xor() -> None:
    assert_eq(pql_x.xor(3), pl_x.xor(3))
    assert_eq(pql_x.__xor__(3), pl_x.xor(3))
    assert_eq(pql_x.xor("n"), pl_x.xor("n"))


@pytest.mark.parametrize("by", ["age", "n", 2])
def test_repeat_by(by: int | str) -> None:
    assert_eq(pql_x.repeat_by(by), pl_x.repeat_by(by))


def test_mul() -> None:
    assert_eq(pql_x.mul(5), pl_x.__mul__(5))
    assert_eq(pql_salary.mul(2), pl_salary.__mul__(2))


def test_truediv() -> None:
    assert_eq(pql_x.__truediv__(5), pl_x.__truediv__(5))

    assert_eq(pql_salary.truediv(1000), pl_salary.__truediv__(1000))


def test_replace() -> None:
    assert_eq(pql_x.replace(2, 99), pl_x.replace(2, 99))


def test_repr() -> None:
    txt = "this=Identifier(this=name, quoted=False))"
    assert txt in repr(pql.col("name"))


def test_list_sort_uses_array_sort_constructor() -> None:
    list_sort = pql.col("arr").list.sort().inner.__class__
    array_sort = pql.col("arr").arr.sort().inner.__class__

    assert list_sort is array_sort


def test_and() -> None:
    assert_eq(pql_a.__and__("b"), pl_a.__and__("b"))
    with pytest.raises(AssertionError, match="DataFrames are different"):
        assert_eq(pql_x.and_(1), pl_x.and_(1))


def test_or() -> None:
    assert_eq(pql_a.__or__("b"), pl_a.__or__("b"))
    with pytest.raises(AssertionError, match="DataFrames are different"):
        assert_eq(pql_x.or_(1), pl_x.or_(1))


def test_not() -> None:
    assert_eq(pql_a.__invert__(), pl_a.__invert__())
    assert_eq(pql_a.not_(), pl_a.not_())


def test_radd() -> None:
    assert_eq(pql_x.__radd__(10), pl_x.__radd__(10))


def test_lit_keeps_literal_name_when_composed() -> None:
    assert_eq(pql.lit(10).add(pql_x), pl.lit(10).add(pl_x))


def test_rmul() -> None:
    assert_eq(pql_x.__rmul__(10), pl_x.__rmul__(10))


def test_rtruediv() -> None:
    assert_eq(pql_x.__rtruediv__(10), pl_x.__rtruediv__(10))


def test_eq() -> None:
    assert_eq(pql_x.__eq__(2), pl_x.__eq__(2))


def test_lt() -> None:
    assert_eq(pql_x.__lt__(3), pl_x.__lt__(3))


def test_gt() -> None:
    assert_eq(pql_x.__gt__(2), pl_x.__gt__(2))


def test_ge() -> None:
    assert_eq(pql_x.__ge__(3), pl_x.__ge__(3))


def test_rsub() -> None:
    assert_eq(pql_x.__rsub__(10), pl_x.__rsub__(10))


def test_rfloordiv() -> None:
    assert_eq(pql_x.__rfloordiv__(10), pl_x.__rfloordiv__(10))


def test_rmod() -> None:
    assert_eq(pql_x.__rmod__(10), pl_x.__rmod__(10))


def test_rpow() -> None:
    assert_eq(pql_x.__rpow__(2.0), pl_x.__rpow__(2.0))


def test_neg() -> None:
    assert_eq(pql_x.__neg__(), pl_x.__neg__())
    assert_eq(pql_x.neg(), pl_x.neg())


def test_ne() -> None:
    assert_eq(pql_x.__ne__(2), pl_x.__ne__(2))


def test_le() -> None:
    assert_eq(pql_x.__le__(2), pl_x.__le__(2))


def test_sub() -> None:
    assert_eq(pql_x.__sub__(5), pl_x.__sub__(5))


def test_floordiv() -> None:
    assert_eq(pql_x.__floordiv__(3), pl_x.__floordiv__(3))


_SIMPLE_FNS = Seq((
    (pql_x.sinh, pl_x.sinh),
    (pql_x.cosh, pl_x.cosh),
    (pql_x.tanh, pl_x.tanh),
    (pql_x.is_finite, pl_x.is_finite),
    (pql_x.is_infinite, pl_x.is_infinite),
    (pql_x.count, pl_x.count),
    (pql_x.len, pl_x.len),
    (pql_x.min, pl_x.min),
    (pql_x.max, pl_x.max),
    (pql_x.sum, pl_x.sum),
    (pql_x.mean, pl_x.mean),
    (pql_x.median, pl_x.median),
    (pql_x.product, pl_x.product),
    (pql_x.n_unique, pl_x.n_unique),
    (pql_x.null_count, pl_x.null_count),
    (pql_x.has_nulls, pl_x.has_nulls),
    (pql_x.cot, pl_x.cot),
    (pql_x.degrees, pl_x.degrees),
    (pql_x.radians, pl_x.radians),
    (pql_x.sign, pl_x.sign),
    (pql_x.floor, pl_x.floor),
    (pql_x.ceil, pl_x.ceil),
    (pql_x.cbrt, pl_x.cbrt),
    (pql_x.abs, pl_x.abs),
    (pql_x.approx_n_unique, pl_x.approx_n_unique),
    (pql_x.is_last_distinct, pl_x.is_last_distinct),
    (pql_x.exp, pl_x.exp),
    (pql_x.sin, pl_x.sin),
    (pql_x.cos, pl_x.cos),
    (pql_x.tan, pl_x.tan),
    (pql_x.arctan, pl_x.arctan),
    (pql_x.arccosh, pl_x.arccosh),
    (pql_x.arcsinh, pl_x.arcsinh),
    (pql_x.bitwise_and, pl_x.bitwise_and),
    (pql_x.bitwise_or, pl_x.bitwise_or),
    (pql_x.bitwise_xor, pl_x.bitwise_xor),
    (pql_x.diff, pl_x.diff),
))


@pytest.mark.parametrize("fn", _SIMPLE_FNS, ids=_SIMPLE_FNS.into(_ids))
def test_simple_methods_on_x(fn: Fns) -> None:
    assert_eq(fn[0](), fn[1]())


_SIMPLE_FN_AGE = Seq((
    (pql_age.null_count, pl_age.null_count),
    (pql_age.has_nulls, pl_age.has_nulls),
))


@pytest.mark.parametrize("fn", _SIMPLE_FN_AGE, ids=_SIMPLE_FN_AGE.into(_ids))
def test_simple_methods_on_age(fn: Fns) -> None:
    assert_eq(fn[0](), fn[1]())


def test_uint_only_simple() -> None:
    assert_eq(pql_uint.log(2), pl_uint.log(2))
    assert_eq(pql_uint.log10(), pl_uint.log10())
    assert_eq(pql_uint.log1p(), pl_uint.log1p())
    assert_eq(pql_uint.sqrt(), pl_uint.sqrt())


def test_mode() -> None:
    assert_eq(pql_x.mode(), pl_x.mode(), with_cols=False)


def test_is_first_distinct() -> None:
    assert_eq(pql_a.is_first_distinct(), pl_a.is_first_distinct())


def test_col_getattr() -> None:
    assert_eq(pql.col.a, pl.col.a)


@pytest.mark.parametrize("mode", pql.sql.typing.RoundMode.__args__)
def test_round(mode: pql.sql.typing.RoundMode) -> None:
    assert_eq(pql_float_vals.round(2, mode=mode), pl_float_vals.round(2, mode=mode))


def test_pipe() -> None:
    assert_eq(pql_x.pipe(lambda x: x * 2), pl_x.pipe(lambda x: x * 2))


def test_forward_fill() -> None:
    assert_eq(pql_a.forward_fill(), pl_a.forward_fill())


@pytest.mark.parametrize("limit", [1, 2, None])
def test_backward_fill(limit: int | None) -> None:
    assert_eq(pql_n.backward_fill(limit), pl_n.backward_fill(limit))


def test_is_nan() -> None:
    assert_eq(pql_nan_vals.is_nan(), pl_nan_vals.is_nan())


def test_is_null() -> None:
    assert_eq(pql_n.is_null(), pl_n.is_null())


def test_is_not_null() -> None:
    assert_eq(pql_n.is_not_null(), pl_n.is_not_null())


def test_is_not_nan() -> None:
    assert_eq(pql_nan_vals.is_not_nan(), pl_nan_vals.is_not_nan())


def test_fill_nan() -> None:
    assert_eq(pql_nan_vals.fill_nan(0.0), pl_nan_vals.fill_nan(0.0))


def test_is_duplicated() -> None:
    assert_eq(pql_a.is_duplicated(), pl_a.is_duplicated())


def test_arccos() -> None:
    assert_eq(
        pql_x.truediv(20).arccos(),
        pl_x.truediv(20).arccos(),
    )


def test_arcsin() -> None:
    assert_eq(
        pql_x.truediv(20).arcsin(),
        pl_x.truediv(20).arcsin(),
    )


def test_arctanh() -> None:
    assert_eq(
        pql_x.truediv(30).arctanh(),
        pl_x.truediv(30).arctanh(),
    )


def test_pow() -> None:
    assert_eq(pql_x.pow(2), pl_x.__pow__(2))
    assert_eq(pql_x.__pow__(2), pl_x.__pow__(2))


def test_add() -> None:
    assert_eq(pql_age.add(10), pl_age.__add__(10))
    assert_eq(pql_age.__add__(10), pl_age.__add__(10))


def test_mod() -> None:
    assert_eq(pql_age.__mod__(10), pl_age.__mod__(10))
    assert_eq(pql_age.mod(10), pl_age.__mod__(10))


def test_is_in() -> None:
    assert_eq(pql_x.is_in([2, 3]), pl_x.is_in([2, 3]))


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_shift(n: int) -> None:
    assert_eq(pql_x.shift(n), pl_x.shift(n))


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
def test_pct_change(n: int) -> None:
    assert_eq(pql_x.pct_change(n), pl_x.pct_change(n))


@pytest.mark.parametrize("closed", pql.sql.typing.ClosedInterval.__args__)
def test_is_between(closed: pql.sql.typing.ClosedInterval) -> None:
    assert_eq(
        pql_x.is_between(2, 10, closed=closed),
        pl_x.is_between(2, 10, closed=closed),
    )


def test_is_unique() -> None:
    assert_eq(pql_a.is_unique(), pl_a.is_unique())


def test_first() -> None:
    assert_eq(pql_n.first(), pl_n.first(ignore_nulls=False))
    assert_eq(pql_n.any_value(), pl_n.first(ignore_nulls=True))


def test_last() -> None:
    assert_eq(pql_n.last(), pl_n.last())


_MIN_MAX_BY_FNS = Seq(((pql_x.min_by, pl_x.min_by), (pql_x.max_by, pl_x.max_by)))


@pytest.mark.parametrize("fns", _MIN_MAX_BY_FNS, ids=_MIN_MAX_BY_FNS.into(_ids))
def test_min_max_by(
    fns: tuple[
        Callable[[str | pql.Expr], pql.Expr], Callable[[str | pl.Expr], pl.Expr]
    ],
) -> None:
    assert_eq(fns[0]("age"), fns[1]("age"))
    assert_eq(fns[0]("age"), fns[1]("age"))


def test_implode() -> None:
    assert_eq(pql_x.implode(), pl_x.implode())


def test_unique() -> None:
    assert_eq(pql_x.unique(), pl_x.unique(), with_cols=False)


def test_composed_reducer_broadcast() -> None:
    assert_eq(pql_x.sum().add(3), pl_x.sum().add(3))


def test_is_close() -> None:
    assert_eq(
        pql_salary.is_close(pql_salary.add(0.001), abs_tol=0.01, rel_tol=0.0),
        pl_salary.is_close(pl_salary.add(0.001), abs_tol=0.01, rel_tol=0.0),
    )
    assert_eq(
        pql_salary.is_close(
            pql_salary.add(0.001), abs_tol=0.01, rel_tol=0.0, nans_equal=True
        ),
        pl_salary.is_close(
            pl_salary.add(0.001), abs_tol=0.01, rel_tol=0.0, nans_equal=True
        ),
    )


_ROLLING_FNS = Seq((
    (pql_x.rolling_mean, pl_x.rolling_mean),
    (pql_x.rolling_sum, pl_x.rolling_sum),
    (pql_x.rolling_min, pl_x.rolling_min),
    (pql_x.rolling_max, pl_x.rolling_max),
    (pql_x.rolling_median, pl_x.rolling_median),
))


class RollingFn[T: pql.Expr | pl.Expr](Protocol):
    def __call__(
        self, window_size: int, min_samples: int | None, center: bool
    ) -> T: ...


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("window_size", [2, 4])
@pytest.mark.parametrize("min_samples", [None, 1, 2])
@pytest.mark.parametrize("method", _ROLLING_FNS, ids=_ROLLING_FNS.into(_ids))
def test_rolling(
    method: tuple[RollingFn[pql.Expr], RollingFn[pl.Expr]],
    window_size: int,
    min_samples: int | None,
    center: bool,
) -> None:
    assert_eq(
        method[0](window_size=window_size, min_samples=min_samples, center=center),
        method[1](window_size=window_size, min_samples=min_samples, center=center),
    )


def test_rolling_std() -> None:
    assert_eq(
        pql_x.rolling_std(window_size=3, min_samples=2, center=False, ddof=1),
        pl_x.rolling_std(window_size=3, min_samples=2, center=False, ddof=1),
    )


def test_rolling_var() -> None:
    assert_eq(
        pql_x.rolling_var(window_size=3, min_samples=2, center=False, ddof=1),
        pl_x.rolling_var(window_size=3, min_samples=2, center=False, ddof=1),
    )


@pytest.mark.parametrize("lower_bound", [None, 2, 10])
@pytest.mark.parametrize("upper_bound", [None, 10, 20])
def test_clip(lower_bound: int | None, upper_bound: int | None) -> None:
    assert_eq(
        pql_x.clip(lower_bound=lower_bound, upper_bound=upper_bound),
        pl_x.clip(lower_bound=lower_bound, upper_bound=upper_bound),
    )


bias_arg = pytest.mark.parametrize("bias", [True, False])


@pytest.mark.parametrize("fisher", [True, False])
@bias_arg
def test_kurtosis(fisher: bool, bias: bool) -> None:
    assert_eq(
        pql_x.kurtosis(fisher=fisher, bias=bias),
        pl_x.kurtosis(fisher=fisher, bias=bias),
    )


@bias_arg
def test_skew(bias: bool) -> None:
    assert_eq(pql_x.skew(bias=bias), pl_x.skew(bias=bias))


def test_quantile() -> None:
    assert_eq(
        pql_x.quantile(0.75, interpolation=True),
        pl_x.quantile(0.75, "linear"),
    )
    assert_eq(
        pql_x.quantile(0.75, interpolation=False),
        pl_x.quantile(0.75, "equiprobable"),
    )


@desc_param
@pytest.mark.parametrize("order_by", ["n", None])
def test_over(order_by: str | None, descending: bool) -> None:
    assert_eq(
        pql_x.sum().over("a", order_by=order_by, descending=descending),
        pl_x.sum().over("a", order_by=order_by, descending=descending),
    )


def test_over_nested() -> None:
    assert_eq(pql_x.rolling_max(2).over("a"), pl_x.rolling_max(2).over("a"))


@pytest.mark.parametrize("nulls_last", [True, False])
def test_over_with_nulls_last(*, nulls_last: bool) -> None:
    assert_eq(
        pql_n.first().over("a", order_by="x", nulls_last=nulls_last),
        pl_n.first().over("a", order_by="x", nulls_last=nulls_last),
    )


@pytest.mark.parametrize("strategy", pql.sql.typing.FillNullStrategy.__args__)
def test_fill_null(strategy: pql.sql.typing.FillNullStrategy) -> None:
    assert_eq(pql_age.fill_null(0), pl_age.fill_null(0))
    assert_eq(pql_age.fill_null(strategy=strategy), pl_age.fill_null(strategy=strategy))


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_fill_null_limit(strategy: pql.sql.typing.FillNullStrategy, limit: int) -> None:
    assert_eq(
        pql_age.fill_null(strategy=strategy, limit=limit),
        pl_age.fill_null(strategy=strategy, limit=limit),
    )


def test_fill_null_no_value_or_strategy() -> None:
    msg = "must specify either a fill `value` or `strategy`"
    with pytest.raises(ValueError, match=msg):
        _ = pql_age.fill_null()
    with pytest.raises(ValueError, match=msg):
        _ = pl_age.fill_null()


def test_fill_null_limit_negative() -> None:
    with pytest.raises(
        ValueError, match="Can't process negative `limit` value for fill_null"
    ):
        _ = pql_age.fill_null(strategy="forward", limit=-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = pl_age.fill_null(strategy="forward", limit=-1)


def test_fill_null_limit_invalid_strategy() -> None:
    err = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
    with pytest.raises(ValueError, match=err):
        _ = pql_age.fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pl_age.fill_null(strategy="min", limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pql_age.fill_null(0, limit=1)
    with pytest.raises(ValueError, match=err):
        _ = pl_age.fill_null(0, limit=1)


def test_fill_val_and_strat() -> None:
    err = "cannot specify both `value` and `strategy`"
    with pytest.raises(ValueError, match=err):
        _ = pql_age.fill_null(value=0, strategy="min")
    with pytest.raises(ValueError, match=err):
        _ = pl_age.fill_null(value=0, strategy="min")


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_and_var(ddof: int) -> None:
    assert_eq(pql_x.std(ddof=ddof), pl_x.std(ddof=ddof))
    assert_eq(pql_x.var(ddof=ddof), pl_x.var(ddof=ddof))


def test_all() -> None:
    assert_eq(pql_x.gt(0).all(), pl_x.gt(0).all())


def test_any() -> None:
    assert_eq(pql_x.gt(10).any(), pl_x.gt(10).any())


@desc_param
@pytest.mark.parametrize("method", pql.sql.typing.RankMethod.__args__)
def test_rank(method: pql.sql.typing.RankMethod, descending: bool) -> None:
    assert_eq(
        pql_x.rank(method, descending=descending),
        pl_x.rank(method, descending=descending),
    )


def test_cum_count() -> None:
    assert_eq(pql_x.cum_count(), pl_x.cum_count())
    assert_eq(pql_x.cum_count(reverse=True), pl_x.cum_count(reverse=True))


def test_cum_sum() -> None:
    assert_eq(pql_x.cum_sum(), pl_x.cum_sum())
    assert_eq(pql_x.cum_sum(reverse=True), pl_x.cum_sum(reverse=True))


def test_cum_prod() -> None:
    assert_eq(pql_x.cum_prod(), pl_x.cum_prod())
    assert_eq(pql_x.cum_prod(reverse=True), pl_x.cum_prod(reverse=True))


def test_cum_min() -> None:
    assert_eq(pql_x.cum_min(), pl_x.cum_min())
    assert_eq(pql_x.cum_min(reverse=True), pl_x.cum_min(reverse=True))


def test_cum_max() -> None:
    assert_eq(pql_x.cum_max(), pl_x.cum_max())
    assert_eq(pql_x.cum_max(reverse=True), pl_x.cum_max(reverse=True))


@desc_param
@pytest.mark.parametrize("nulls_last", [True, False])
def test_arg_sort(descending: bool, nulls_last: bool) -> None:
    assert_eq(
        pql_x.arg_sort(descending=descending, nulls_last=nulls_last),
        pl_x.arg_sort(descending=descending, nulls_last=nulls_last),
    )


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_truncate(decimals: int) -> None:
    assert_eq(pql_float_vals.truncate(decimals), pl_float_vals.truncate(decimals))


def test_dot() -> None:
    assert_eq(
        pql_x.dot("age"),
        pl_x.dot("age"),
    )


@pytest.mark.parametrize("base", [1, 2, 3])
@pytest.mark.parametrize("normalize", [True, False])
def test_entropy(base: int, *, normalize: bool) -> None:
    assert_eq(
        pql_x.abs().cast(pql.Float64()).entropy(base=base, normalize=normalize),
        pl_x.abs().cast(pl.Float64).entropy(base=base, normalize=normalize),
    )
