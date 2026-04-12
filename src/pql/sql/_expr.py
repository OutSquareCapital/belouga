from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from functools import cache
from typing import TYPE_CHECKING, ClassVar, Self, override

import pyochain as pc
from duckdb import Expression
from sqlglot import exp, parse_one

from pql.sql.typing import IntoExprColumn

from ._code_gen import Fns
from ._conversions import args_into_glot, pql_into_glot
from ._core import DuckHandler, func
from ._window import FrameBound, OverBuilder, get_order, get_partition, make_spec

if TYPE_CHECKING:
    from decimal import Decimal

    from . import namespaces as nm
    from .typing import (
        ClosedInterval,
        FillNullStrategy,
        FrameMode,
        IntoExpr,
        IntoExprColumn,
        RoundMode,
        WindowExclude,
    )
    from .utils import TryIter


@cache
def _fill_strategy() -> pc.Dict[FillNullStrategy, Callable[[SqlExpr], SqlExpr]]:
    from ._funcs import coalesce

    return pc.Dict.from_ref({
        "forward": lambda expr: expr.last_value().over(
            frame_end=pc.Some(0), ignore_nulls=True
        ),
        "backward": lambda expr: expr.any_value().over(frame_start=pc.Some(0)),
        "min": lambda expr: coalesce(expr, expr.min().over()),
        "max": lambda expr: coalesce(expr, expr.max().over()),
        "mean": lambda expr: coalesce(expr, expr.mean().over()),
        "zero": lambda expr: coalesce(expr, 0),
        "one": lambda expr: coalesce(expr, 1),
    })


"""Computation strategies for `fill_null` when ."""


class SqlExpr(Fns):  # noqa: PLW1641
    """A wrapper around sqlglot.exp.Expr that provides operator overloading and SQL function methods."""

    __slots__: ClassVar[Iterable[str]] = ()

    @classmethod
    def new(cls, value: IntoExpr, *, as_col: bool = False) -> Self:
        """Convert a value to a `SqlExpr`.

        Args:
            value (IntoExpr): The value to convert.
            as_col (bool): Whether to treat `str` values as column names (default: `False`).

        Returns:
            SqlExpr
        """
        from .._expr import Expr

        match value:
            case DuckHandler():
                return cls(value.inner())
            case Expr():
                return cls(value.inner().inner())
            case str() if as_col:
                return cls(exp.column(value))
            case exp.Expr():
                return cls(value)
            case Expression():
                return cls(parse_one(str(value), dialect="duckdb"))
            case _:
                return cls(exp.convert(value))

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
        return self._build_op(op, self.inner(), pql_into_glot(other))

    def _rbinop[T: exp.Binary](self, op: type[T], other: IntoExpr) -> Self:
        return self._build_op(op, pql_into_glot(other), self.inner())

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
        return self._floordiv_op(self.inner(), pql_into_glot(other))

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
        return self._cls(exp.Not(this=self.inner()))

    def not_(self) -> Self:
        return self.__invert__()

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
        return self._cls(exp.Neg(this=self.inner()))

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
        return self._floordiv_op(pql_into_glot(other), self.inner())

    def rfloordiv(self, other: IntoExpr) -> Self:
        return self.__rfloordiv__(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self.new(other).fmod(self.inner())

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

    def sub(self, other: IntoExpr) -> Self:
        return self.__sub__(other)

    def alias(self, name: str) -> Self:
        return self._cls(exp.Alias(this=self.inner(), alias=exp.to_identifier(name)))

    def asc(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner(), desc=False))

    def between(self, lower: IntoExpr, upper: IntoExpr) -> Self:
        return self._cls(
            exp.Between(
                this=self.inner(), low=pql_into_glot(lower), high=pql_into_glot(upper)
            )
        )

    def cast(self, dtype: exp.DataType) -> Self:
        return self._cls(exp.Cast(this=self.inner(), to=dtype))

    def collate(self, collation: str) -> Self:
        expr = exp.Collate(this=self.inner(), expression=exp.to_identifier(collation))
        return self._cls(expr)

    def desc(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner(), desc=True))

    def is_in(self, *args: IntoExpr) -> Self:
        return self._cls(exp.In(this=self.inner(), expressions=args_into_glot(args)))

    def is_not_in(self, *args: IntoExpr) -> Self:
        return self._cls(exp.Not(this=self.is_in(*args).inner()))

    def is_not_null(self) -> Self:
        return self._cls(exp.Not(this=self.is_null().inner()))

    def is_null(self) -> Self:
        return self._cls(exp.Is(this=self.inner(), expression=exp.Null()))

    def nulls_first(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner(), nulls_first=True))

    def nulls_last(self) -> Self:
        return self._cls(exp.Ordered(this=self.inner(), nulls_first=False))

    def show(self) -> None:
        print(self.inner().sql(dialect="duckdb"))

    def _reversed(self, *, reverse: bool = False) -> Self:
        match reverse:
            case True:
                return self.over(frame_start=pc.Some(0))
            case False:
                return self.over(frame_end=pc.Some(0))

    @property
    def arr(self) -> nm.SqlExprArrayNameSpace:
        """Access array functions."""
        from .namespaces import SqlExprArrayNameSpace

        return SqlExprArrayNameSpace(self)

    @property
    def str(self) -> nm.SqlExprStringNameSpace:
        """Access string functions."""
        from .namespaces import SqlExprStringNameSpace

        return SqlExprStringNameSpace(self)

    @property
    def list(self) -> nm.SqlExprListNameSpace:
        """Access list functions."""
        from .namespaces import SqlExprListNameSpace

        return SqlExprListNameSpace(self)

    @property
    def struct(self) -> nm.SqlExprStructNameSpace:
        """Access struct functions."""
        from .namespaces import SqlExprStructNameSpace

        return SqlExprStructNameSpace(self)

    @property
    def dt(self) -> nm.SqlExprDateTimeNameSpace:
        """Access datetime functions."""
        from .namespaces import SqlExprDateTimeNameSpace

        return SqlExprDateTimeNameSpace(self)

    @property
    def json(self) -> nm.SqlExprJsonNameSpace:
        """Access JSON functions."""
        from .namespaces import SqlExprJsonNameSpace

        return SqlExprJsonNameSpace(self)

    @property
    def re(self) -> nm.SqlExprRegexNameSpace:
        """Access regex functions."""
        from .namespaces import SqlExprRegexNameSpace

        return SqlExprRegexNameSpace(self)

    @property
    def map(self) -> nm.SqlExprMapNameSpace:
        """Access map functions."""
        from .namespaces import SqlExprMapNameSpace

        return SqlExprMapNameSpace(self)

    @property
    def enum(self) -> nm.SqlExprEnumNameSpace:
        """Access enum functions."""
        from .namespaces import SqlExprEnumNameSpace

        return SqlExprEnumNameSpace(self)

    @property
    def geo(self) -> nm.SqlExprGeoSpatialNameSpace:
        """Access geospatial functions."""
        from .namespaces import SqlExprGeoSpatialNameSpace

        return SqlExprGeoSpatialNameSpace(self)

    def fill_nulls(
        self,
        value: pc.Option[IntoExpr],
        strategy: pc.Option[FillNullStrategy],
        limit: pc.Option[int],
    ) -> Self:
        def _get_strat() -> pc.Result[SqlExpr | Self, ValueError]:  # noqa: PLR0911
            from ._funcs import coalesce

            match (value, strategy, limit):
                case (pc.Some(_), pc.Some(_), _):
                    msg = "cannot specify both `value` and `strategy`"
                    return pc.Err(ValueError(msg))
                case (_, _, pc.Some(lim)) if lim < 0:
                    msg = "Can't process negative `limit` value for fill_null"
                    return pc.Err(ValueError(msg))
                case (
                    _,
                    pc.Some("forward") | pc.Some("backward") as strat,
                    pc.Some(lim),
                ):
                    iterator = pc.Iter(range(1, lim + 1))
                    match strat.value:
                        case "forward":
                            exprs: pc.Iter[SqlExpr] = iterator.map(self.shift)
                        case _:
                            exprs = iterator.map(lambda offset: self.shift(-offset))
                    return pc.Ok(exprs.insert(self).reduce(coalesce))
                case (_, _, pc.Some(_)):
                    msg = "can only specify `limit` when strategy is set to 'backward' or 'forward'"
                    return pc.Err(ValueError(msg))
                case (pc.Some(val), pc.NONE, pc.NONE):
                    return pc.Ok(coalesce((self.inner(), val)))
                case (_, pc.Some(strat), pc.NONE):
                    return pc.Ok(self.pipe(_fill_strategy()[strat]))
                case _:
                    msg = "must specify either a fill `value` or `strategy`"
                    return pc.Err(ValueError(msg))

        return _get_strat().map(lambda e: self._cls(e.inner())).unwrap()

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

    def var(self, ddof: int) -> Self:
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def std(self, ddof: int) -> Self:
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def kurtosis(self, *, bias: bool = True) -> Self:
        match bias:
            case True:
                return self.kurtosis_pop()
            case False:
                return self.kurtosis_samp()

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
                return self.lag(n_val, None).over()
            case _:
                return self.lead(-n, None).over()

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

    def n_unique(self) -> Self:
        """Count distinct values.

        Returns:
            Self: A expression representing the count of distinct values.
        """
        return self._cls(self.implode().list.distinct().list.length().inner())

    def unique(self) -> Self:
        return self._cls(exp.Distinct(expressions=[self.inner()]))

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
        expr = self.new(by, as_col=True).list.range().list.eval(self).inner()
        return self._cls(expr)

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values.

        Returns:
            Self: An expression with values replaced.
        """
        from ._when import when

        return self._cls(when(self.eq(old)).then(new).otherwise(self).inner())

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
                    .inner()
                )

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence.

        Returns:
            Self: A boolean expression indicating whether the value is the first occurrence.
        """
        return self.row_number().over(pc.Some(self)).eq(1)

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence.

        Returns:
            Self: A boolean expression indicating whether the value is the last occurrence.
        """
        from .._meta import Marker

        row_idx = Marker.TEMP.to_expr()
        return (
            self
            .row_number()
            .over(pc.Some(self), pc.Some(row_idx), descending=True)
            .eq(1)
        )

    def is_duplicated(self) -> Self:
        """Check if value is duplicated.

        Returns:
            Self: A boolean expression indicating whether the value is duplicated.
        """
        from ._funcs import all

        return self._cls(all().count().over(pc.Some(self)).gt(1).inner())

    def is_unique(self) -> Self:
        """Check if value is unique.

        Returns:
            Self: A boolean expression indicating whether the value is unique.
        """
        from ._funcs import all

        return self._cls(all().count().over(pc.Some(self)).eq(1).inner())

    def arg_sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Return indices that would sort the expression."""
        from .._meta import Marker

        row_idx = Marker.TEMP.to_expr()
        return self._cls(
            row_idx
            .nth_value(row_idx.add(1))
            .over(
                order_by=pc.Some((self, row_idx)),
                descending=(descending, False),
                nulls_last=(nulls_last, False),
            )
            .inner()
        )

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value.

        Returns:
            Self: An expression with null values filled with the last non-null value.
        """
        return self.last_value().over(frame_end=pc.Some(0), ignore_nulls=True)

    def backward_fill(self, limit: int | None) -> Self:
        """Fill null values with the next non-null value.

        Returns:
            Self: An expression with null values filled with the next non-null value.
        """
        expr = self.any_value().over
        return (
            pc
            .Option(limit)
            .map(lambda lmt: expr(frame_start=pc.Some(0), frame_end=pc.Some(lmt)))
            .unwrap_or_else(lambda: expr(frame_start=pc.Some(0)))
        )

    def fill_nan(self, value: float | IntoExprColumn | None) -> Self:
        """Fill NaN values.

        Returns:
            Self: An expression with NaN values filled with the specified value.
        """
        from ._when import when

        return self._cls(when(self.is_nan()).then(value).otherwise(self).inner())

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
        return self._cls(func("LOG", x, self.inner()))

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
        expr = exp.Greatest(this=self.inner(), expressions=args_into_glot(args))
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
        expr = exp.Least(this=self.inner(), expressions=args_into_glot(args))
        return self._cls(expr)

    def over(  # noqa: PLR0913, PLR0917
        self,
        partition_by: pc.Option[TryIter[IntoExprColumn]] = pc.NONE,
        order_by: pc.Option[TryIter[IntoExprColumn]] = pc.NONE,
        frame_start: pc.Option[FrameBound] = pc.NONE,
        frame_end: pc.Option[FrameBound] = pc.NONE,
        frame_mode: FrameMode = "ROWS",
        exclude: pc.Option[WindowExclude] = pc.NONE,
        filter_cond: pc.Option[IntoExprColumn] = pc.NONE,
        fn_order_by: pc.Option[TryIter[IntoExprColumn]] = pc.NONE,
        *,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
        ignore_nulls: bool = False,
        distinct: bool = False,
        fn_descending: TryIter[bool] = False,
        fn_nulls_last: TryIter[bool] = False,
    ) -> Self:
        order = get_order(order_by, descending=descending, nulls_last=nulls_last)
        spec = make_spec(
            frame_mode,
            has_order_by=order_by.is_some(),
            frame_start=frame_start,
            frame_end=frame_end,
            exclude=exclude,
        )
        return self.__class__(
            OverBuilder(self.inner())
            .handle_nulls(ignore_nulls=ignore_nulls)
            .handle_distinct(distinct=distinct)
            .handle_fn_order_by(
                fn_order_by=fn_order_by,
                fn_descending=fn_descending,
                fn_nulls_last=fn_nulls_last,
            )
            .handle_filter(filter_cond)
            .handle_clauses(
                partition_by=get_partition(partition_by), order=order, spec=spec
            )
            .build()
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
                fn_order_by=pc.Option(order_by),
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
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

    def rank(
        self,
        *,
        order_by: TryIter[IntoExprColumn] = None,
        ignore_nulls: bool = False,
        descending: TryIter[bool] = False,
        nulls_last: TryIter[bool] = False,
    ) -> Self:
        """The rank of the current row with gaps; same as row_number of its first peer.

        If an `ORDER BY` clause is specified, the rank is computed within the frame using the provided ordering instead of the frame ordering.

        Returns:
            Self
        """
        return self._cls(
            OverBuilder(exp.Rank()).build_fn(
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

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
                fn_order_by=pc.Option(order_by),
                ignore_nulls=ignore_nulls,
                fn_descending=descending,
                fn_nulls_last=nulls_last,
            )
        )

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
        return self._cls(
            exp.BitwiseXor(this=self.inner(), expression=pql_into_glot(right))
        )
