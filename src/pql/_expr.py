from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Self, override

from pyochain import NONE, Err, Iter, NoneOption as Null, Ok, Option, Result, Some
from sqlglot import exp

from ._core import func, into_expr, into_expr_list
from ._fns import Fns
from ._meta import ExprMeta, Marker
from ._window import (
    BoundsValues,
    FrameBound,
    OverBuilder,
    get_order,
    get_partition,
    make_spec,
    rolling_agg,
)
from .datatypes import DataType
from .utils import try_iter

if TYPE_CHECKING:
    from decimal import Decimal

    from . import namespaces as nm
    from .typing import (
        ClosedInterval,
        FillNullStrategy,
        FrameMode,
        IntoDataType,
        IntoExpr,
        IntoExprColumn,
        RankMethod,
        RoundMode,
        WindowExclude,
    )
    from .utils import TryIter

_FILL_STRATEGY: dict[FillNullStrategy, Callable[[Expr], Expr]] = {
    "forward": lambda expr: expr.last_value().window(frame_end=0, ignore_nulls=True),
    "backward": lambda expr: expr.any_value().window(frame_start=0),
    "min": lambda expr: expr.coalesce(expr.min().window()),
    "max": lambda expr: expr.coalesce(expr.max().window()),
    "mean": lambda expr: expr.coalesce(expr.mean().window()),
    "zero": lambda expr: expr.coalesce(0),
    "one": lambda expr: expr.coalesce(1),
}


@dataclass(slots=True, repr=False)
class Expr(Fns):
    """A wrapper around sqlglot.exp.Expr that provides operator overloading and SQL function methods."""

    meta: ExprMeta = field(default_factory=ExprMeta)

    @classmethod
    def new(cls, value: IntoExpr, *, as_col: bool = False) -> Self:
        """Convert a value to a `Expr`.

        Args:
            value (IntoExpr): The value to convert.
            as_col (bool): Whether to treat `str` values as column names (default: `False`).

        Returns:
            Expr
        """
        return cls(into_expr(value, as_col=as_col))

    @override
    def _cls(self, value: exp.Expr) -> Self:
        return self.__class__(value, self.meta)

    def _rolling_agg(
        self,
        agg: Callable[[Expr], Expr],
        window_size: int,
        min_samples: int | None,
        *,
        center: bool,
    ) -> Self:
        from ._meta import Marker
        from ._when import when

        spec = BoundsValues.rolling(window_size, center=center)

        def _clause(e: Expr) -> Expr:
            return e.inner.pipe(rolling_agg, Marker.TEMP, spec).pipe(Expr)

        return (
            when(self.count().pipe(_clause).ge(min_samples or window_size))
            .then(self.pipe(agg).pipe(_clause))
            .otherwise(None)
            .inner.pipe(self._cls)
        )

    def rolling_max(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling max.

        Returns:
            Self: A new expression that evaluates to the rolling max.
        """
        return self._rolling_agg(Expr.max, window_size, min_samples, center=center)

    def rolling_min(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling min.

        Returns:
            Self: A new expression that evaluates to the rolling min.
        """
        return self._rolling_agg(Expr.min, window_size, min_samples, center=center)

    def rolling_mean(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling mean.

        Returns:
            Self: A new expression that evaluates to the rolling mean.
        """
        return self._rolling_agg(Expr.mean, window_size, min_samples, center=center)

    def rolling_median(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling median.

        Returns:
            Self: A new expression that evaluates to the rolling median.
        """
        return self._rolling_agg(Expr.median, window_size, min_samples, center=center)

    def rolling_sum(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
    ) -> Self:
        """Compute rolling sum.

        Returns:
            Self: A new expression that evaluates to the rolling sum.
        """
        return self._rolling_agg(Expr.sum, window_size, min_samples, center=center)

    def rolling_std(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Compute rolling standard deviation.

        Returns:
            Self: A new expression that evaluates to the rolling standard deviation.
        """
        return self._rolling_agg(
            lambda expr: expr.std(ddof), window_size, min_samples, center=center
        )

    def rolling_var(
        self,
        window_size: int,
        min_samples: int | None = None,
        *,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Compute rolling variance.

        Returns:
            Self: A new expression that evaluates to the rolling variance.
        """
        return self._rolling_agg(
            lambda expr: expr.var(ddof), window_size, min_samples, center=center
        )

    def _build_op[T: exp.Binary](
        self, op: type[T], left: exp.Expr, right: exp.Expr
    ) -> Self:

        def _cols_op(expr: exp.Expr) -> exp.Expr:
            match expr:
                case exp.Star():
                    return exp.Columns(this=expr.copy())
                case _:
                    return expr

        expr = exp.Paren(this=op(this=_cols_op(left), expression=_cols_op(right)))
        return self._cls(expr)

    def _binop[T: exp.Binary](self, op: type[T], other: IntoExpr) -> Self:
        return self._build_op(op, self.inner, into_expr(other))

    def _rbinop[T: exp.Binary](self, op: type[T], other: IntoExpr) -> Self:
        return self._build_op(op, into_expr(other), self.inner).alias(Marker.LITERAL)

    def __add__(self, other: IntoExpr) -> Self:
        return self._binop(exp.Add, other)

    def add(self, other: IntoExpr) -> Self:
        return self.__add__(other)

    def __and__(self, other: IntoExpr) -> Self:
        return self._binop(exp.And, other)

    def and_(self, other: IntoExpr) -> Self:
        return self.__and__(other)

    def __truediv__(self, other: IntoExpr) -> Self:
        return self._binop(exp.Div, other)

    def truediv(self, other: IntoExpr) -> Self:
        return self.__truediv__(other)

    @override
    def __eq__(self, other: IntoExpr) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._binop(exp.EQ, other)

    def eq(self, other: IntoExpr) -> Self:
        return self.__eq__(other)

    def _floordiv_op(self, left: exp.Expr, right: exp.Expr) -> Self:
        return self._build_op(exp.Div, left, right).floor()

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self._floordiv_op(self.inner, into_expr(other))

    def floordiv(self, other: IntoExpr) -> Self:
        return self.__floordiv__(other)

    def __ge__(self, other: IntoExpr) -> Self:
        return self._binop(exp.GTE, other)

    def ge(self, other: IntoExpr) -> Self:
        return self.__ge__(other)

    def __gt__(self, other: IntoExpr) -> Self:
        return self._binop(exp.GT, other)

    def gt(self, other: IntoExpr) -> Self:
        return self.__gt__(other)

    def __invert__(self) -> Self:
        return self.not_()

    def not_(self) -> Self:
        return self._cls(exp.Not(this=self.inner))

    def __le__(self, other: IntoExpr) -> Self:
        return self._binop(exp.LTE, other)

    def le(self, other: IntoExpr) -> Self:
        return self.__le__(other)

    def __lt__(self, other: IntoExpr) -> Self:
        return self._binop(exp.LT, other)

    def lt(self, other: IntoExpr) -> Self:
        return self.__lt__(other)

    def __mod__(self, other: IntoExprColumn | Decimal | float) -> Self:
        return self.mod(other)

    def __mul__(self, other: IntoExpr) -> Self:
        return self._binop(exp.Mul, other)

    def mul(self, other: IntoExpr) -> Self:
        return self.__mul__(other)

    @override
    def __ne__(self, other: IntoExpr) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._binop(exp.NEQ, other)

    def ne(self, other: IntoExpr) -> Self:
        return self.__ne__(other)

    def __neg__(self) -> Self:
        return self._cls(exp.Neg(this=self.inner))

    def neg(self) -> Self:
        return self.__neg__()

    def __or__(self, other: IntoExpr) -> Self:
        return self._binop(exp.Or, other)

    def or_(self, other: IntoExpr) -> Self:
        return self.__or__(other)

    def __pow__(self, other: IntoExprColumn | float) -> Self:
        return self.pow(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.Add, other)

    def radd(self, other: IntoExpr) -> Self:
        return self.__radd__(other)

    def __rand__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.And, other)

    def rand(self, other: IntoExpr) -> Self:
        return self.__rand__(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self._floordiv_op(into_expr(other), self.inner).alias(Marker.LITERAL)

    def rfloordiv(self, other: IntoExpr) -> Self:
        return self.__rfloordiv__(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self.new(other).fmod(self.inner).alias(Marker.LITERAL)

    def rmod(self, other: IntoExpr) -> Self:
        return self.__rmod__(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.Mul, other)

    def rmul(self, other: IntoExpr) -> Self:
        return self.__rmul__(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.Or, other)

    def ror(self, other: IntoExpr) -> Self:
        return self.__ror__(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.Pow, other)

    def rpow(self, other: IntoExpr) -> Self:
        return self.__rpow__(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.Sub, other)

    def rsub(self, other: IntoExpr) -> Self:
        return self.__rsub__(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self._rbinop(exp.Div, other)

    def rtruediv(self, other: IntoExpr) -> Self:
        return self.__rtruediv__(other)

    def __sub__(self, other: IntoExpr) -> Self:
        return self._binop(exp.Sub, other)

    @override
    def __hash__(self) -> int:
        return hash(self.inner.output_name)

    def sub(self, other: IntoExpr) -> Self:
        return self.__sub__(other)

    def alias(self, name: str) -> Self:
        expr = exp.Alias(this=self.inner.unalias(), alias=exp.to_identifier(name))
        return self.__class__(expr, self.meta.unalias())

    def asc(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner, desc=False))

    def between(self, lower: IntoExpr, upper: IntoExpr) -> Self:
        return self._cls(
            exp.Between(this=self.inner, low=into_expr(lower), high=into_expr(upper))
        )

    def cast(self, dtype: IntoDataType) -> Self:
        match dtype:
            case DataType():
                dtype_expr = dtype.raw
            case exp.DataType():
                dtype_expr = dtype

        return self._cls(exp.Cast(this=self.inner, to=dtype_expr))

    def collate(self, collation: str) -> Self:
        expr = exp.Collate(this=self.inner, expression=exp.to_identifier(collation))
        return self._cls(expr)

    def desc(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner, desc=True))

    def is_in(self, args: TryIter[IntoExpr], *more_args: IntoExpr) -> Self:
        exprs = into_expr_list(try_iter(args).chain(more_args))
        return self._cls(exp.In(this=self.inner, expressions=exprs))

    def is_not_in(self, args: TryIter[IntoExpr], *more_args: IntoExpr) -> Self:
        return self._cls(exp.Not(this=self.is_in(args, *more_args).inner))

    def is_not_null(self) -> Self:
        return self._cls(exp.Not(this=self.is_null().inner))

    def is_null(self) -> Self:
        return self._cls(exp.Is(this=self.inner, expression=exp.Null()))

    def nulls_first(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner, nulls_first=True))

    def nulls_last(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner, nulls_first=False))

    def show(self) -> None:
        print(self.inner.sql(dialect="duckdb", identify=True))

    def _reversed(self, *, reverse: bool = False) -> Self:
        match reverse:
            case True:
                return self.window(frame_start=0)
            case False:
                return self.window(frame_end=0)

    @property
    def arr(self) -> nm.ExprArrayNameSpace:
        """Access array functions."""
        from .namespaces import ExprArrayNameSpace

        return ExprArrayNameSpace(self)

    @property
    def str(self) -> nm.ExprStringNameSpace:
        """Access string functions."""
        from .namespaces import ExprStringNameSpace

        return ExprStringNameSpace(self)

    @property
    def list(self) -> nm.ExprListNameSpace:
        """Access list functions."""
        from .namespaces import ExprListNameSpace

        return ExprListNameSpace(self)

    @property
    def struct(self) -> nm.ExprStructNameSpace:
        """Access struct functions."""
        from .namespaces import ExprStructNameSpace

        return ExprStructNameSpace(self)

    @property
    def dt(self) -> nm.ExprDateTimeNameSpace:
        """Access datetime functions."""
        from .namespaces import ExprDateTimeNameSpace

        return ExprDateTimeNameSpace(self)

    @property
    def json(self) -> nm.ExprJsonNameSpace:
        """Access JSON functions."""
        from .namespaces import ExprJsonNameSpace

        return ExprJsonNameSpace(self)

    @property
    def re(self) -> nm.ExprRegexNameSpace:
        """Access regex functions."""
        from .namespaces import ExprRegexNameSpace

        return ExprRegexNameSpace(self)

    @property
    def map(self) -> nm.ExprMapNameSpace:
        """Access map functions."""
        from .namespaces import ExprMapNameSpace

        return ExprMapNameSpace(self)

    @property
    def enum(self) -> nm.ExprEnumNameSpace:
        """Access enum functions."""
        from .namespaces import ExprEnumNameSpace

        return ExprEnumNameSpace(self)

    @property
    def geo(self) -> nm.ExprGeoSpatialNameSpace:
        """Access geospatial functions."""
        from .namespaces import ExprGeoSpatialNameSpace

        return ExprGeoSpatialNameSpace(self)

    @property
    def name(self) -> nm.ExprNameNameSpace:
        """Access name functions."""
        from .namespaces import ExprNameNameSpace

        return ExprNameNameSpace(self)

    def fill_null(
        self,
        value: IntoExpr = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        def _get_strat() -> Result[Expr | Self, ValueError]:  # noqa: PLR0911
            from ._funcs import coalesce

            match (Option(value), Option(strategy), Option(limit)):
                case (Some(_), Some(_), _):
                    msg = "cannot specify both `value` and `strategy`"
                    return Err(ValueError(msg))
                case (_, _, Some(lim)) if lim < 0:
                    msg = "Can't process negative `limit` value for fill_null"
                    return Err(ValueError(msg))
                case (
                    _,
                    Some("forward") | Some("backward") as strat,
                    Some(lim),
                ):
                    iterator = Iter(range(1, lim + 1))
                    match strat.value:
                        case "forward":
                            exprs: Iter[Expr] = iterator.map(self.shift)
                        case _:
                            exprs = iterator.map(lambda offset: self.shift(-offset))
                    return Ok(exprs.insert(self).reduce(coalesce))
                case (_, _, Some(_)):
                    msg = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
                    return Err(ValueError(msg))
                case (Some(val), Null(), Null()):
                    return Ok(self.coalesce(val))
                case (_, Some(strat), Null()):
                    return Ok(self.pipe(_FILL_STRATEGY[strat]))  # pyright: ignore[reportArgumentType]
                case _:
                    msg = "must specify either a fill `value` or `strategy`"
                    return Err(ValueError(msg))

        return _get_strat().map(lambda e: self._cls(e.inner)).unwrap()

    def cum_count(self, *, reverse: bool = False) -> Self:
        """Cumulative non-null count.

        Returns:
            Self: A cumulative count expression.
        """
        return self.count()._reversed(reverse=reverse)

    def cum_sum(self, *, reverse: bool = False) -> Self:
        """Cumulative sum.

        Returns:
            Self: A cumulative sum expression.
        """
        return self.sum()._reversed(reverse=reverse)

    def cum_prod(self, *, reverse: bool = False) -> Self:
        """Cumulative product.

        Returns:
            Self: A cumulative product expression.
        """
        return self.product()._reversed(reverse=reverse)

    def cum_min(self, *, reverse: bool = False) -> Self:
        """Cumulative minimum.

        Returns:
            Self: A cumulative minimum expression.
        """
        return self.min()._reversed(reverse=reverse)

    def cum_max(self, *, reverse: bool = False) -> Self:
        """Cumulative maximum.

        Returns:
            Self: A cumulative maximum expression.
        """
        return self.max()._reversed(reverse=reverse)

    def var(self, ddof: int = 1) -> Self:
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def std(self, ddof: int = 1) -> Self:
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def kurtosis_fisher(self, *, bias: bool = True) -> Self:
        match bias:
            case True:
                return self.kurtosis_pop()
            case False:
                return self.kurtosis_samp()

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
        base = self.kurtosis_fisher(bias=bias)
        match fisher:
            case True:
                return base
            case False:
                return base.add(3)

    def skew(self, *, bias: bool) -> Self:
        adjusted = self.skewness()
        match bias:
            case False:
                return adjusted
            case True:
                n = self.count()
                factor = n.sub(2).truediv(n.mul(n.sub(1)).sqrt())
                return adjusted.mul(factor)

    def shift(self, n: int = 1) -> Self:
        match n:
            case 0:
                return self
            case n_val if n_val > 0:
                return self.lag(n_val, None).window()
            case _:
                return self.lead(-n, None).window()

    def round(self, decimals: int, mode: RoundMode) -> Self:
        match mode:
            case "half_to_even":
                return self.round_even(decimals)
            case "half_away_from_zero":
                return self.round_from_zero(decimals)

    def quantile(self, quantile: float, *, interpolation: bool = True) -> Self:
        match interpolation:
            case True:
                return self.quantile_cont(quantile)
            case False:
                return self.quantile_disc(quantile)

    def is_between(
        self, lower_bound: IntoExpr, upper_bound: IntoExpr, closed: ClosedInterval
    ) -> Self:
        match closed:
            case "both":
                return self.ge(lower_bound).and_(self.le(upper_bound))
            case "left":
                return self.ge(lower_bound).and_(self.lt(upper_bound))
            case "right":
                return self.gt(lower_bound).and_(self.le(upper_bound))
            case "none":
                return self.gt(lower_bound).and_(self.lt(upper_bound))

    def clip(self, lower_bound: IntoExpr = None, upper_bound: IntoExpr = None) -> Self:
        match (lower_bound, upper_bound):
            case (None, None):
                return self
            case (None, upper):
                return self.least(upper)
            case (lower, None):
                return self.greatest(lower)
            case (lower, upper):
                return self.greatest(lower).least(upper)

    @override
    def len(self) -> Self:
        """Get the number of rows in context (including nulls).

        Returns:
            Self
        """
        return self.is_null().count()

    def null_count(self) -> Self:
        """Count null values.

        Returns:
            Self: A new expression that evaluates to the number of null values.
        """
        return self.is_null().count().sub(self.count())

    def diff(self) -> Self:
        return self.sub(self.shift())

    def pct_change(self, n: int = 1) -> Self:
        return self.truediv(self.shift(n)).sub(1)

    def n_unique(self) -> Self:
        """Count distinct values.

        Returns:
            Self: A expression representing the count of distinct values.
        """
        return self._cls(self.implode().list.distinct().list.length().inner)

    def unique(self) -> Self:
        return self._cls(exp.Distinct(expressions=[self.inner]))

    def has_nulls(self) -> Self:
        """Return whether the expression contains nulls.

        Returns:
            Self: A boolean expression indicating whether the expression contains nulls.
        """
        return self.is_null().any()

    def repeat_by(self, by: IntoExprColumn | int) -> Self:
        """Repeat values by count, returning a list.

        Returns:
            Self: A list expression with repeated values.
        """
        return self._cls(
            self
            .new(by, as_col=True)
            .list.range()
            .list.eval(self)
            .alias(self.inner.output_name)
            .inner
        )

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values.

        Returns:
            Self: An expression with values replaced.
        """
        from ._when import when

        return self._cls(
            when(self.eq(old))
            .then(new)
            .otherwise(self)
            .alias(self.inner.output_name)
            .inner
        )

    def is_close(
        self,
        other: IntoExpr,
        abs_tol: float = 1e-8,
        rel_tol: float = 1e-5,
        *,
        nans_equal: bool = False,
    ) -> Self:
        """Check if two floating point values are close.

        Returns:
            Self: A boolean expression indicating whether the values are close.
        """
        from ._funcs import lit
        from ._when import when

        other_expr = self.new(other)
        threshold = lit(abs_tol).add(lit(rel_tol).mul(other_expr.abs()))
        close = self.sub(other_expr).abs().le(threshold)
        match nans_equal:
            case False:
                return close
            case True:
                return self._cls(
                    when(self.is_nan().and_(other_expr.is_nan()))
                    .then(value=True)
                    .otherwise(close)
                    .alias(self.inner.output_name)
                    .inner
                )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence.

        Returns:
            Self: A boolean expression indicating whether the value is the first occurrence.
        """
        return self.row_number().window(self).eq(1)

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence.

        Returns:
            Self: A boolean expression indicating whether the value is the last occurrence.
        """
        row_idx = Marker.TEMP.to_expr()
        return self.row_number().window(self, row_idx, descending=True).eq(1)

    def is_duplicated(self) -> Self:
        """Check if value is duplicated.

        Returns:
            Self: A boolean expression indicating whether the value is duplicated.
        """
        from ._funcs import all

        return self._cls(all().count().window(self).gt(1).inner)

    def is_unique(self) -> Self:
        """Check if value is unique.

        Returns:
            Self: A boolean expression indicating whether the value is unique.
        """
        from ._funcs import all

        return self._cls(all().count().window(self).eq(1).inner)

    def arg_sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Return indices that would sort the expression."""
        row_idx = Marker.TEMP.to_expr()
        return self._cls(
            row_idx
            .nth_value(row_idx.add(1))
            .window(
                order_by=(self, row_idx),
                descending=(descending, False),
                nulls_last=(nulls_last, False),
            )
            .inner
        )

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value.

        Returns:
            Self: An expression with null values filled with the last non-null value.
        """
        return self.last_value().window(frame_end=0, ignore_nulls=True)

    def backward_fill(self, limit: int | None) -> Self:
        """Fill null values with the next non-null value.

        Returns:
            Self: An expression with null values filled with the next non-null value.
        """
        expr = self.any_value().window
        return (
            Option(limit)
            .map(lambda lmt: expr(frame_start=0, frame_end=lmt))
            .unwrap_or_else(lambda: expr(frame_start=0))
        )

    def fill_nan(self, value: float | IntoExprColumn | None) -> Self:
        """Fill NaN values.

        Returns:
            Self: An expression with NaN values filled with the specified value.
        """
        from ._when import when

        return self._cls(
            when(self.is_nan())
            .then(value)
            .otherwise(self)
            .alias(self.inner.output_name)
            .inner
        )

    def dot(self, other: IntoExpr) -> Self:
        """Compute the dot product with another expression.

        Returns:
            Self: An expression representing the dot product.
        """
        return self.mul(other).sum()

    def entropy(self, base: float = math.e, *, normalize: bool = True) -> Self:
        """Compute the entropy.

        Returns:
            Self: An expression representing the entropy.
        """
        from ._funcs import lit

        match normalize:
            case True:
                expr = (
                    self.sum().ln().sub(self.mul(self.ln()).sum().truediv(self.sum()))
                )
            case False:
                expr = self.mul(self.ln().neg()).sum()
        return expr.truediv(lit(base).ln())

    def log(self, x: IntoExprColumn | float | None = None) -> Self:
        """Computes the logarithm of x to base b.

        b may be omitted, in which case the default 10.

        **SQL name**: *log*

        Args:
            x (IntoExprColumn | float | None): `DOUBLE` expression

        Returns:
            Self
        """
        return self._cls(func("LOG", x, self.inner))

    def greatest(self, *args: IntoExpr) -> Self:
        """Returns the largest value.

        For strings lexicographical ordering is used.

        Note that lowercase characters are considered “larger” than uppercase characters and collations are not supported.

        **SQL name**: *greatest*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            Self
        """
        expr = exp.Greatest(this=self.inner, expressions=into_expr_list(args))
        return self._cls(expr)

    def least(self, *args: IntoExpr) -> Self:
        """Returns the smallest value.

        For strings lexicographical ordering is used.

        Note that uppercase characters are considered “smaller” than lowercase characters, and collations are not supported.

        **SQL name**: *least*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            Self
        """
        expr = exp.Least(this=self.inner, expressions=into_expr_list(args))
        return self._cls(expr)

    def window(  # noqa: PLR0913, PLR0917
        self,
        partition_by: TryIter[IntoExprColumn] | None = None,
        order_by: TryIter[IntoExprColumn] | None = None,
        frame_start: FrameBound | None = None,
        frame_end: FrameBound | None = None,
        frame_mode: FrameMode = "ROWS",
        exclude: WindowExclude | None = None,
        filter_cond: IntoExprColumn | None = None,
        fn_order_by: TryIter[IntoExprColumn] | None = None,
        *,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
        ignore_nulls: bool = False,
        distinct: bool = False,
        fn_descending: TryIter[bool] = False,
        fn_nulls_last: TryIter[bool] = False,
    ) -> Self:
        order = get_order(
            Option(order_by), descending=descending, nulls_last=nulls_last
        )
        spec = make_spec(
            frame_mode,
            has_order_by=order_by is not None,
            frame_start=Option(frame_start),
            frame_end=Option(frame_end),
            exclude=Option(exclude),
        )
        return self._cls(
            OverBuilder(self.inner)
            .handle_nulls(ignore_nulls=ignore_nulls)
            .handle_distinct(distinct=distinct)
            .handle_fn_order_by(
                fn_order_by=Option(fn_order_by),
                fn_descending=fn_descending,
                fn_nulls_last=fn_nulls_last,
            )
            .handle_filter(Option(filter_cond))
            .handle_clauses(
                partition_by=get_partition(Option(partition_by)),
                order=order,
                spec=spec,
            )
            .build()
        )

    def over(
        self,
        partition_by: TryIter[IntoExpr],
        *more_exprs: IntoExpr,
        order_by: TryIter[IntoExpr] = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        expr = partial(self.window, descending=descending, nulls_last=nulls_last)
        partition_exprs = (
            try_iter(partition_by)
            .chain(more_exprs)
            .map(lambda x: self.new(x, as_col=True))
        )
        return (
            Option(order_by)
            .map(lambda value: try_iter(value).map(lambda x: self.new(x, as_col=True)))
            .map(lambda order_exprs: expr(partition_exprs, order_exprs))
            .unwrap_or_else(lambda: expr(partition_exprs))
        )

    def set_order(self, *, desc: bool, nulls_last: bool) -> Self:
        """Set the ordering of the expression. Syntactic sugar for use in parameterized functions.

        Args:
        desc (bool): Whether to sort in descending order.
        nulls_last (bool): Whether to put nulls last.

        Returns:
            Self
        """
        match (desc, nulls_last):
            case (True, True):
                return self.desc().nulls_last()
            case (True, False):
                return self.desc()
            case (False, True):
                return self.asc().nulls_last()
            case (False, False):
                return self.asc()

    def dense_rank(self) -> Self:
        """The rank of the current row without gaps; this function counts peer groups.

        Returns:
            Self
        """
        return self._cls(exp.DenseRank())

    def cume_dist(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The cumulative distribution: (number of partition rows preceding or peer with current row) / total partition rows.

        If an `ORDER BY` clause is specified, the distribution is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._cls(
            OverBuilder(exp.CumeDist()).build_fn(
                fn_order_by=Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def percent_rank(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The relative rank of the current row: (rank() - 1) / (total partition rows - 1).

        If an `ORDER BY` clause is specified, the relative rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._cls(
            OverBuilder(exp.PercentRank()).build_fn(
                fn_order_by=Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def rank(self, method: RankMethod = "average", *, descending: bool = False) -> Self:
        """Compute rank values.

        Returns:
            Self: A new expression that evaluates to the rank of values according to the specified method.
        """

        def _peer_count() -> Expr:
            from ._funcs import all

            return all().count().window(self.inner)

        def _base_rank() -> Self:
            return (
                OverBuilder(exp.Rank())
                .build_fn(
                    fn_order_by=NONE,
                    ignore_nulls=False,
                    fn_descending=descending,
                    fn_nulls_last=False,
                )
                .pipe(self._cls)
                .pipe(_over)
            )

        def _over(expr: Self) -> Self:
            return expr.window(order_by=self.inner, descending=descending)

        match method:
            case "average":
                br = _base_rank()
                max_rank = br.add(_peer_count()).sub(1)
                return br.add(max_rank).truediv(2)
            case "min":
                return _base_rank()
            case "max":
                return _base_rank().add(_peer_count()).sub(1)
            case "dense":
                return self.dense_rank().pipe(_over)
            case "ordinal":
                return self.row_number().pipe(_over)

    def row_number(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The number of the current row within the partition, counting from 1.

        If an `ORDER BY` clause is specified, the row number is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._cls(
            OverBuilder(exp.RowNumber()).build_fn(
                fn_order_by=Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def __xor__(
        self, right: IntoExprColumn | bytes | bytearray | memoryview | int
    ) -> Self:
        return self.xor(right)

    def xor(self, right: IntoExprColumn | bytes | bytearray | memoryview | int) -> Self:
        """Bitwise XOR.

        **SQL name**: *xor*

        Args:
            right (IntoExprColumn | bytes | bytearray | memoryview | int): `BIGINT | BIT | HUGEINT | INTEGER | SMALLINT | TINYINT | UBIGINT | UHUGEINT | UINTEGER | USMALLINT | UTINYINT` expression

        Examples:
            ```sql
            xor(17, 5)
            ```

        Returns:
            Self
        """
        return self._cls(exp.BitwiseXor(this=self.inner, expression=into_expr(right)))

    def truncate(self, decimals: int = 0) -> Self:
        """Truncate numeric value to given number of decimal places.

        Returns:
            Self
        """
        return self.trunc(decimals)

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x.

        Returns:
            Self
        """
        return self.add(1).ln()

    def is_not_nan(self) -> Self:
        """Check if value is not NaN.

        Returns:
            Self
        """
        return self.is_nan().not_()

    def is_infinite(self) -> Self:
        """Check if value is infinite.

        Returns:
            Self
        """
        return self.is_inf()

    def approx_n_unique(self) -> Self:
        """Approximate the number of unique values.

        Returns:
            Self: A new expression that evaluates to the approximate number of unique values.
        """
        return self.approx_count_distinct()

    def bitwise_and(self) -> Self:
        return self.bit_and()

    def bitwise_or(self) -> Self:
        return self.bit_or()

    def bitwise_xor(self) -> Self:
        return self.bit_xor()

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash.

        Returns:
            Self
        """
        return self._cls(self.str.hash(seed).inner)

    def coalesce(self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
        """Create a `COALESCE` expression.

        Args:
            exprs (TryIter[IntoExpr]): The expressions to coalesce.
            *more_exprs (IntoExpr): Additional expressions to coalesce.

        Returns:
            Expr: An expression representing the `COALESCE` operation.
        """
        exprs_lst = try_iter(exprs).chain(more_exprs).into(into_expr_list, as_col=True)
        return self._cls(exp.Coalesce(this=self.inner, expressions=exprs_lst))

    def arctan(self) -> Self:
        """Compute the arc tangent.

        Returns:
            Self
        """
        return self.atan()

    def arccos(self) -> Self:
        """Compute the arc cosine.

        Returns:
            Self
        """
        return self.acos()

    def arccosh(self) -> Self:
        """Compute the inverse hyperbolic cosine.

        Returns:
            Self
        """
        return self.acosh()

    def arcsin(self) -> Self:
        """Compute the arc sine.

        Returns:
            Self
        """
        return self.asin()

    def arcsinh(self) -> Self:
        """Compute the inverse hyperbolic sine.

        Returns:
            Self
        """
        return self.asinh()

    def arctanh(self) -> Self:
        """Compute the inverse hyperbolic tangent.

        Returns:
            Self
        """
        return self.atanh()

    def max_by(self, by: IntoExpr, col2: IntoExprColumn | int | None = None) -> Self:
        """Finds the row with the maximum val.

        Calculates the non-NULL arg expression at that row.

        **SQL name**: *max_by*

        See Also:
            arg_max, argmax

        Args:
            by (IntoExpr): `ANY | BIGINT | BLOB | DATE | DOUBLE | HUGEINT | INTEGER | TIMESTAMP | TIMESTAMP WITH TIME ZONE | VARCHAR` expression
            col2 (IntoExprColumn | int | None): `BIGINT` expression

        Examples:
            ```sql
            max_by(A, B)
            ```

        Returns:
            Self
        """
        return self._cls(func("MAX_BY", self.inner, into_expr(by, as_col=True), col2))

    def min_by(self, by: IntoExpr, col2: IntoExprColumn | int | None = None) -> Self:
        """Finds the row with the minimum val.

        Calculates the non-NULL arg expression at that row.

        **SQL name**: *min_by*

        See Also:
            arg_min, argmin

        Args:
            by (IntoExpr): `ANY | BIGINT | BLOB | DATE | DOUBLE | HUGEINT | INTEGER | TIMESTAMP | TIMESTAMP WITH TIME ZONE | VARCHAR` expression
            col2 (IntoExprColumn | int | None): `BIGINT` expression

        Examples:
            ```sql
            min_by(A, B)
            ```

        Returns:
            Self
        """
        return self._cls(func("MIN_BY", self.inner, into_expr(by, as_col=True), col2))
