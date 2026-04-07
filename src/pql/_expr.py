from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Self, override

import pyochain as pc

from . import sql
from ._meta import ExprMeta, Marker
from .sql import SqlExpr
from .sql.utils import TryIter, try_iter

if TYPE_CHECKING:
    from ._datatypes import DataType
    from ._namespaces import (
        ExprArrayNameSpace,
        ExprDateTimeNameSpace,
        ExprListNameSpace,
        ExprNameNameSpace,
        ExprStringNameSpace,
        ExprStructNameSpace,
    )
    from ._typing import RankMethod
    from .sql.typing import (
        ClosedInterval,
        FillNullStrategy,
        IntoExpr,
        IntoExprColumn,
        RoundMode,
    )

_NONE = sql.lit(None)


@dataclass(slots=True)
class Expr(sql.CoreHandler[SqlExpr]):
    _inner: SqlExpr
    meta: ExprMeta

    @override
    def _cls(self, value: SqlExpr) -> Self:
        return self.__class__(value, self.meta)

    def _as_literal_name(self, expr: SqlExpr) -> Self:
        return self.__class__(expr.alias(Marker.LIT), self.meta.clear_alias())

    def _clear_alias_name(self) -> Expr:
        return self.__class__(self.inner(), self.meta.clear_alias())

    def _with_alias_mapper(self, mapper: Callable[[str], str]) -> Expr:
        return self.__class__(self.inner(), self.meta.with_alias_mapper(mapper))

    def _set_alias(self, alias: str) -> Self:
        return self.__class__(self.inner().alias(alias), self.meta.clear_alias())

    def _rolling_agg(
        self,
        agg: Callable[[SqlExpr], SqlExpr],
        window_size: int,
        min_samples: int | None,
        *,
        center: bool,
    ) -> Self:
        spec = sql.BoundsValues.rolling(window_size, center=center)

        def _clause(e: SqlExpr) -> SqlExpr:
            return SqlExpr(sql.rolling_agg(e.inner(), Marker.TEMP, spec))

        return (
            sql
            .when(self.inner().count().pipe(_clause).ge(min_samples or window_size))
            .then(self.inner().pipe(agg).pipe(_clause))
            .otherwise(_NONE)
            .pipe(self._cls)
        )

    @property
    def str(self) -> ExprStringNameSpace:
        """Access string operations.

        Returns:
            Self
        """
        from ._namespaces import ExprStringNameSpace

        return ExprStringNameSpace(self)

    @property
    def list(self) -> ExprListNameSpace:
        """Access list operations.

        Returns:
            Self
        """
        from ._namespaces import ExprListNameSpace

        return ExprListNameSpace(self)

    @property
    def arr(self) -> ExprArrayNameSpace:
        """Access array operations.

        Returns:
            Self
        """
        from ._namespaces import ExprArrayNameSpace

        return ExprArrayNameSpace(self)

    @property
    def struct(self) -> ExprStructNameSpace:
        """Access struct operations.

        Returns:
            Self
        """
        from ._namespaces import ExprStructNameSpace

        return ExprStructNameSpace(self)

    @property
    def name(self) -> ExprNameNameSpace:
        """Access name operations.

        Returns:
            Self
        """
        from ._namespaces import ExprNameNameSpace

        return ExprNameNameSpace(self)

    @property
    def dt(self) -> ExprDateTimeNameSpace:
        """Access datetime operations.

        Returns:
            Self
        """
        from ._namespaces import ExprDateTimeNameSpace

        return ExprDateTimeNameSpace(self)

    def __add__(self, other: IntoExpr) -> Self:
        return self.add(other)

    def __radd__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().radd(other))

    def __sub__(self, other: IntoExpr) -> Self:
        return self.sub(other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rsub(other))

    def __mul__(self, other: IntoExpr) -> Self:
        return self.mul(other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rmul(other))

    def __truediv__(self, other: IntoExpr) -> Self:
        return self.truediv(other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rtruediv(other))

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self.floordiv(other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rfloordiv(other))

    def __mod__(self, other: IntoExprColumn | Decimal | float) -> Self:
        return self.mod(other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rmod(other))

    def __pow__(self, other: IntoExprColumn | float) -> Self:
        return self.pow(other)

    def __rpow__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rpow(other))

    def __neg__(self) -> Self:
        return self.neg()

    @override
    def __eq__(self, other: IntoExpr) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.eq(other)

    @override
    def __ne__(self, other: IntoExpr) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.ne(other)

    def __lt__(self, other: IntoExpr) -> Self:
        return self.lt(other)

    def __le__(self, other: IntoExpr) -> Self:
        return self.le(other)

    def __gt__(self, other: IntoExpr) -> Self:
        return self.gt(other)

    def __ge__(self, other: IntoExpr) -> Self:
        return self.ge(other)

    def __and__(self, other: IntoExpr) -> Self:
        return self.and_(other)

    def __rand__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().rand(other))

    def __or__(self, other: IntoExpr) -> Self:
        return self.or_(other)

    def __xor__(self, other: IntoExpr) -> Self:
        return self.xor(other)

    def __ror__(self, other: IntoExpr) -> Self:
        return self._as_literal_name(self.inner().ror(other))

    def __invert__(self) -> Self:
        return self.not_()

    @override
    def __hash__(self) -> int:
        return hash(self.inner().get_name())

    def add(self, other: IntoExpr) -> Self:
        """Add another expression or value.

        Returns:
            Self
        """
        return self._cls(self.inner().add(other))

    def sub(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().sub(other))

    def mul(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().mul(other))

    def truediv(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().truediv(other))

    def floordiv(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().floordiv(other))

    def mod(self, other: IntoExprColumn | Decimal | float) -> Self:
        return self._cls(self.inner().mod(other))

    def pow(self, other: IntoExprColumn | float) -> Self:
        return self._cls(self.inner().pow(other))

    def neg(self) -> Self:
        return self._cls(self.inner().neg())

    def abs(self) -> Self:
        return self._cls(self.inner().abs())

    def eq(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().eq(other))

    def ne(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().ne(other))

    def lt(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().lt(other))

    def le(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().le(other))

    def gt(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().gt(other))

    def ge(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().ge(other))

    def and_(self, others: IntoExpr) -> Self:
        return self._cls(self.inner().and_(others))

    def or_(self, others: IntoExpr) -> Self:
        return self._cls(self.inner().or_(others))

    def not_(self) -> Self:
        return self._cls(self.inner().not_())

    def bitwise_and(self) -> Self:
        return self._cls(self.inner().bit_and())

    def bitwise_or(self) -> Self:
        return self._cls(self.inner().bit_or())

    def bitwise_xor(self) -> Self:
        return self._cls(self.inner().bit_xor())

    def xor(self, other: IntoExpr) -> Self:
        return self._cls(self.inner().xor(SqlExpr.new(other)))

    def alias(self, name: str) -> Self:
        """Rename the expression.

        Returns:
            Self: A new expression with the given alias.
        """
        return self._set_alias(name)

    def is_null(self) -> Self:
        """Check if the expression is NULL.

        Returns:
            Self: A new expression that evaluates to true if the original expression is NULL, false otherwise.
        """
        return self._cls(self.inner().is_null())

    def is_not_null(self) -> Self:
        """Check if the expression is not NULL.

        Returns:
            Self: A new expression that evaluates to true if the original expression is not NULL, false otherwise.
        """
        return self._cls(self.inner().is_not_null())

    def cast(self, dtype: DataType) -> Self:
        """Cast to a different data type.

        Returns:
            Self: A new expression cast to the given data type.
        """
        return self._cls(self.inner().cast(dtype.raw))

    def is_in(self, other: TryIter[IntoExpr]) -> Self:
        """Check if value is in an iterable of values.

        Returns:
            Self: A new expression that evaluates to true if the original expression is in the given iterable
        """
        return self._cls(self.inner().is_in(*try_iter(other)))

    def shift(self, n: int = 1) -> Self:
        return self._cls(self.inner().shift(n))

    def diff(self) -> Self:
        return self.sub(self.shift())

    def pct_change(self, n: int = 1) -> Self:
        return self.truediv(self.shift(n)).sub(1)

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        return self._cls(self.inner().is_between(lower_bound, upper_bound, closed))

    def clip(self, lower_bound: IntoExpr = None, upper_bound: IntoExpr = None) -> Self:
        return self._cls(self.inner().clip(lower_bound, upper_bound))

    def count(self) -> Self:
        """Count the number of values.

        Returns:
            Self: A new expression that evaluates to the count of values.
        """
        return self._cls(self.inner().count())

    def len(self) -> Self:
        """Get the number of rows in context (including nulls).

        Returns:
            Self: A new expression that evaluates to the number of rows in context (including nulls).
        """
        return self._cls(self.inner().is_null().count())

    def sum(self) -> Self:
        """Compute the sum.

        Returns:
            Self: A new expression that evaluates to the sum of values.
        """
        return self._cls(self.inner().sum())

    def mean(self) -> Self:
        """Compute the mean.

        Returns:
            Self: A new expression that evaluates to the mean of values.
        """
        return self._cls(self.inner().mean())

    def median(self) -> Self:
        """Compute the median.

        Returns:
            Self: A new expression that evaluates to the median of values.
        """
        return self._cls(self.inner().median())

    def min(self) -> Self:
        """Compute the minimum.

        Returns:
            Self: A new expression that evaluates to the minimum of values.
        """
        return self._cls(self.inner().min())

    def max(self) -> Self:
        """Compute the maximum.

        Returns:
            Self: A new expression that evaluates to the maximum of values.
        """
        return self._cls(self.inner().max())

    def first(self, *, ignore_nulls: bool = False) -> Self:
        """Get first value.

        Returns:
            Self: A new expression that evaluates to the first value.
        """
        match ignore_nulls:
            case True:
                return self._cls(self.inner().any_value())
            case False:
                return self._cls(self.inner().first())

    def last(self) -> Self:
        """Get last value.

        Returns:
            Self: A new expression that evaluates to the last value.
        """
        return self._cls(self.inner().last())

    def mode(self) -> Self:
        """Compute mode.

        Returns:
            Self: A new expression that evaluates to the mode of values.
        """
        return self._cls(self.inner().mode())

    def approx_n_unique(self) -> Self:
        """Approximate the number of unique values.

        Returns:
            Self: A new expression that evaluates to the approximate number of unique values.
        """
        return self._cls(self.inner().approx_count_distinct())

    def product(self) -> Self:
        """Compute the product.

        Returns:
            Self: A new expression that evaluates to the product of values.
        """
        return self._cls(self.inner().product())

    def dot(self, other: IntoExpr) -> Self:
        """Compute the dot product with another expression.

        Returns:
            Self: A new expression that evaluates to the dot product with another expression.
        """
        return self._cls(self.inner().dot(other))

    def max_by(self, by: IntoExpr) -> Self:
        """Return the value corresponding to the maximum of another expression.

        Returns:
            Self: A new expression that evaluates to the value corresponding to the maximum of another expression.
        """
        return self._cls(self.inner().max_by(by))

    def min_by(self, by: IntoExpr) -> Self:
        """Return the value corresponding to the minimum of another expression.

        Returns:
            Self: A new expression that evaluates to the value corresponding to the minimum of another expression.
        """
        return self._cls(self.inner().min_by(by))

    def implode(self) -> Self:
        """Aggregate values into a list.

        Returns:
            Self: A new expression that evaluates to a list of values.
        """
        return self._cls(self.inner().implode())

    def unique(self) -> Self:
        """Get unique values.

        Returns:
            Self: A new expression that evaluates to the unique values.
        """
        return self._cls(self.inner().unique())

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
            Self: A new expression that evaluates to whether two floating point values are close.
        """
        return self._cls(
            self.inner().is_close(other, abs_tol, rel_tol, nans_equal=nans_equal)
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
        return self._rolling_agg(SqlExpr.max, window_size, min_samples, center=center)

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
        return self._rolling_agg(SqlExpr.min, window_size, min_samples, center=center)

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
        return self._rolling_agg(SqlExpr.mean, window_size, min_samples, center=center)

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
        return self._rolling_agg(
            SqlExpr.median, window_size, min_samples, center=center
        )

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
        return self._rolling_agg(SqlExpr.sum, window_size, min_samples, center=center)

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

    def std(self, ddof: int = 1) -> Self:
        """Compute the standard deviation.

        Returns:
            Self: A new expression that evaluates to the standard deviation.
        """
        return self._cls(self.inner().std(ddof))

    def var(self, ddof: int = 1) -> Self:
        """Compute the variance.

        Returns:
            Self: A new expression that evaluates to the variance.
        """
        return self._cls(self.inner().var(ddof))

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
        """Compute the kurtosis.

        Returns:
            Self: A new expression that evaluates to the kurtosis.
        """
        base = self.inner().kurtosis(bias=bias)
        match fisher:
            case True:
                return self._cls(base)
            case False:
                return self._cls(base.add(3))

    def skew(self, *, bias: bool = True) -> Self:
        """Compute the skewness.

        Returns:
            Self: A new expression that evaluates to the skewness.
        """
        return self._cls(self.inner().skew(bias=bias))

    def entropy(self, base: float = math.e, *, normalize: bool = True) -> Self:
        """Compute the entropy.

        Returns:
            Self: A new expression that evaluates to the entropy.
        """
        return self._cls(self.inner().entropy(base=base, normalize=normalize))

    def quantile(self, quantile: float, *, interpolation: bool = True) -> Self:
        """Compute the quantile.

        Returns:
            Self: A new expression that evaluates to the quantile.
        """
        return self._cls(self.inner().quantile(quantile, interpolation=interpolation))

    def all(self) -> Self:
        """Return whether all values are true.

        Returns:
            Self
        """
        return self._cls(self.inner().all())

    def any(self) -> Self:
        """Return whether any value is true.

        Returns:
            Self
        """
        return self._cls(self.inner().any())

    def n_unique(self) -> Self:
        """Count distinct values.

        Returns:
            Self: A new expression that evaluates to the number of distinct values.
        """
        return self._cls(self.inner().n_unique())

    def null_count(self) -> Self:
        """Count null values.

        Returns:
            Self: A new expression that evaluates to the number of null values.
        """
        return self.len().sub(self.count())

    def has_nulls(self) -> Self:
        """Return whether the expression contains nulls.

        Returns:
            Self: A new expression that evaluates to whether the expression contains nulls.
        """
        return self._cls(self.inner().has_nulls())

    def rank(self, method: RankMethod = "average", *, descending: bool = False) -> Self:
        """Compute rank values.

        Returns:
            Self: A new expression that evaluates to the rank of values according to the specified method.
        """
        base_rank = (
            self
            .inner()
            .rank()
            .over(order_by=pc.Some(self.inner()), descending=descending)
        )
        peer_count = sql.all().count().over(pc.Some(self.inner()))
        match method:
            case "average":
                max_rank = base_rank.add(peer_count).sub(1)
                expr = base_rank.add(max_rank).truediv(2)
            case "min":
                expr = base_rank
            case "max":
                expr = base_rank.add(peer_count).sub(1)
            case "dense":
                expr = (
                    self
                    .inner()
                    .dense_rank()
                    .over(order_by=pc.Some(self.inner()), descending=descending)
                )
            case "ordinal":
                expr = (
                    self
                    .inner()
                    .row_number()
                    .over(order_by=pc.Some(self.inner()), descending=descending)
                )
        return self._cls(expr)

    def cum_count(self, *, reverse: bool = False) -> Self:
        """Cumulative non-null count.

        Returns:
            Self
        """
        return self._cls(self.inner().cum_count(reverse=reverse))

    def cum_sum(self, *, reverse: bool = False) -> Self:
        """Cumulative sum.

        Returns:
            Self
        """
        return self._cls(self.inner().cum_sum(reverse=reverse))

    def cum_prod(self, *, reverse: bool = False) -> Self:
        """Cumulative product.

        Returns:
            Self
        """
        return self._cls(self.inner().cum_prod(reverse=reverse))

    def cum_min(self, *, reverse: bool = False) -> Self:
        """Cumulative minimum.

        Returns:
            Self
        """
        return self._cls(self.inner().cum_min(reverse=reverse))

    def cum_max(self, *, reverse: bool = False) -> Self:
        """Cumulative maximum.

        Returns:
            Self
        """
        return self._cls(self.inner().cum_max(reverse=reverse))

    def over(
        self,
        partition_by: TryIter[IntoExpr],
        *more_exprs: IntoExpr,
        order_by: TryIter[IntoExpr] = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        expr = partial(self.inner().over, descending=descending, nulls_last=nulls_last)
        partition_exprs: pc.Option[TryIter[IntoExprColumn]] = pc.Some(
            try_iter(partition_by)
            .chain(more_exprs)
            .map(lambda x: SqlExpr.new(x, as_col=True))
        )
        return (
            pc
            .Option(order_by)
            .map(
                lambda value: try_iter(value).map(lambda x: SqlExpr.new(x, as_col=True))
            )
            .map(lambda order_exprs: expr(partition_exprs, pc.Some(order_exprs)))
            .unwrap_or_else(lambda: expr(partition_exprs))
            .pipe(self._cls)
        )

    def floor(self) -> Self:
        """Round down to the nearest integer.

        Returns:
            Self
        """
        return self._cls(self.inner().floor())

    def ceil(self) -> Self:
        """Round up to the nearest integer.

        Returns:
            Self
        """
        return self._cls(self.inner().ceil())

    def round(self, decimals: int = 0, *, mode: RoundMode = "half_to_even") -> Self:
        """Round to given number of decimal places.

        Returns:
            Self
        """
        return self._cls(self.inner().round(decimals, mode=mode))

    def truncate(self, decimals: int = 0) -> Self:
        """Truncate numeric value to given number of decimal places.

        Returns:
            Self
        """
        return self._cls(self.inner().trunc(decimals))

    def sqrt(self) -> Self:
        """Compute the square root.

        Returns:
            Self
        """
        return self._cls(self.inner().sqrt())

    def cbrt(self) -> Self:
        """Compute the cube root.

        Returns:
            Self
        """
        return self._cls(self.inner().cbrt())

    def log(self, base: float = math.e) -> Self:
        """Compute the logarithm.

        Returns:
            Self
        """
        return self._cls(self.inner().log(base))

    def log10(self) -> Self:
        """Compute the base 10 logarithm.

        Returns:
            Self
        """
        return self._cls(self.inner().log10())

    def log1p(self) -> Self:
        """Compute the natural logarithm of 1+x.

        Returns:
            Self
        """
        return self._cls(self.inner().add(1).ln())

    def exp(self) -> Self:
        """Compute the exponential.

        Returns:
            Self
        """
        return self._cls(self.inner().exp())

    def sin(self) -> Self:
        """Compute the sine.

        Returns:
            Self
        """
        return self._cls(self.inner().sin())

    def cos(self) -> Self:
        """Compute the cosine.

        Returns:
            Self
        """
        return self._cls(self.inner().cos())

    def tan(self) -> Self:
        """Compute the tangent.

        Returns:
            Self
        """
        return self._cls(self.inner().tan())

    def arctan(self) -> Self:
        """Compute the arc tangent.

        Returns:
            Self
        """
        return self._cls(self.inner().atan())

    def arccos(self) -> Self:
        """Compute the arc cosine.

        Returns:
            Self
        """
        return self._cls(self.inner().acos())

    def arccosh(self) -> Self:
        """Compute the inverse hyperbolic cosine.

        Returns:
            Self
        """
        return self._cls(self.inner().acosh())

    def arcsin(self) -> Self:
        """Compute the arc sine.

        Returns:
            Self
        """
        return self._cls(self.inner().asin())

    def arcsinh(self) -> Self:
        """Compute the inverse hyperbolic sine.

        Returns:
            Self
        """
        return self._cls(self.inner().asinh())

    def arctanh(self) -> Self:
        """Compute the inverse hyperbolic tangent.

        Returns:
            Self
        """
        return self._cls(self.inner().atanh())

    def cot(self) -> Self:
        """Compute the cotangent.

        Returns:
            Self
        """
        return self._cls(self.inner().cot())

    def sinh(self) -> Self:
        """Compute the hyperbolic sine.

        Returns:
            Self
        """
        return self._cls(self.inner().sinh())

    def cosh(self) -> Self:
        """Compute the hyperbolic cosine.

        Returns:
            Self
        """
        return self._cls(self.inner().cosh())

    def tanh(self) -> Self:
        """Compute the hyperbolic tangent.

        Returns:
            Self
        """
        return self._cls(self.inner().tanh())

    def degrees(self) -> Self:
        """Convert radians to degrees.

        Returns:
            Self
        """
        return self._cls(self.inner().degrees())

    def radians(self) -> Self:
        """Convert degrees to radians.

        Returns:
            Self
        """
        return self._cls(self.inner().radians())

    def sign(self) -> Self:
        """Get the sign of the value.

        Returns:
            Self
        """
        return self._cls(self.inner().sign())

    def forward_fill(self) -> Self:
        """Fill null values with the last non-null value.

        Returns:
            Self
        """
        return self._cls(self.inner().forward_fill())

    def backward_fill(self, limit: int | None = None) -> Self:
        """Fill null values with the next non-null value.

        Returns:
            Self
        """
        return self._cls(self.inner().backward_fill(limit))

    def is_nan(self) -> Self:
        """Check if value is NaN.

        Returns:
            Self
        """
        return self._cls(self.inner().is_nan())

    def is_not_nan(self) -> Self:
        """Check if value is not NaN.

        Returns:
            Self
        """
        return self._cls(self.inner().is_nan().not_())

    def is_finite(self) -> Self:
        """Check if value is finite.

        Returns:
            Self
        """
        return self._cls(self.inner().is_finite())

    def is_infinite(self) -> Self:
        """Check if value is infinite.

        Returns:
            Self
        """
        return self._cls(self.inner().is_inf())

    def fill_nan(self, value: float | IntoExprColumn | None) -> Self:
        """Fill NaN values.

        Returns:
            Self
        """
        return self._cls(self.inner().fill_nan(value))

    def fill_null(
        self,
        value: IntoExpr = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        return (
            self
            .inner()
            .fill_nulls(pc.Option(value), pc.Option(strategy), pc.Option(limit))
            .pipe(self._cls)
        )

    def hash(self, seed: int = 0) -> Self:
        """Compute a hash.

        Returns:
            Self
        """
        return self._cls(self.inner().str.hash(seed))

    def replace(self, old: IntoExpr, new: IntoExpr) -> Self:
        """Replace values.

        Returns:
            Self
        """
        return self._cls(self.inner().replace(old, new))

    def repeat_by(self, by: IntoExprColumn | int) -> Self:
        """Repeat values by count, returning a list.

        Returns:
            Self
        """
        return self._cls(self.inner().repeat_by(by))

    def is_duplicated(self) -> Self:
        """Check if value is duplicated.

        Returns:
            Self
        """
        return self._cls(self.inner().is_duplicated())

    def is_unique(self) -> Self:
        """Check if value is unique.

        Returns:
            Self
        """
        return self._cls(self.inner().is_unique())

    def is_first_distinct(self) -> Self:
        """Check if value is first occurrence.

        Returns:
            Self
        """
        return self._cls(self.inner().is_first_distinct())

    def is_last_distinct(self) -> Self:
        """Check if value is last occurrence.

        Returns:
            Self
        """
        return self._cls(self.inner().is_last_distinct())

    def arg_sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Get the indices that would sort this expression.

        Returns:
            Self
        """
        return self._cls(
            self.inner().arg_sort(descending=descending, nulls_last=nulls_last)
        )
