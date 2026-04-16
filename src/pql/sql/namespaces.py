"""SQL function namespaces for SQL expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

from sqlglot import exp

from ._code_gen import (
    ArrayFns,
    DateTimeFns,
    EnumFns,
    GeoSpatialFns,
    JsonFns,
    ListFns,
    MapFns,
    RegexFns,
    StringFns,
    StructFns,
)
from ._core import DuckHandler, func
from ._expr import SqlExpr
from ._funcs import coalesce, element, lit
from ._when import when

if TYPE_CHECKING:
    from .typing import IntoExpr, IntoExprColumn


@final
class Lit:
    """Literals constants expressions."""

    TITLECASE = lit(r"[a-z]*[^a-z]*")
    NONE = lit(None)
    G_PARAM = lit("g")
    EMPTY_STR = lit("")
    ESCAPE_REGEX = lit(r"([.^$*+?{}\[\]\\|()])")
    ESCAPE_REPLACE = lit(r"\\\1")
    ESCAPE = lit(" ")
    STR_AGG = lit("string_agg")
    DAY = lit("day")
    MONTH = lit("month")
    ZERO = lit("0")


@dataclass(slots=True)
class SqlExprStringNameSpace(StringFns[SqlExpr]):
    """String function namespace for SQL expressions."""

    def concat(self, *args: IntoExpr) -> SqlExpr:
        """Concatenates multiple strings or lists.

        `NULL` inputs are skipped.

        See also operator `||`.

        **SQL name**: *concat*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            SqlExpr
        """
        return self._cls(func("CONCAT", self.inner(), *args))

    def to_titlecase(self) -> SqlExpr:
        """Convert to title case.

        Returns:
            SqlExpr: A new expression that evaluates to the title-cased string.
        """
        return (
            self
            .lower()
            .re.extract_all(Lit.TITLECASE)
            .list.eval(
                element()
                .list.extract(1)
                .str.upper()
                .str.concat(element().str.substring(2))
            )
            .list.aggregate(Lit.STR_AGG, Lit.EMPTY_STR)
        )

    def escape_regex(self) -> SqlExpr:
        """Escape regex metacharacters without escaping plain spaces.

        Returns:
            SqlExpr: A new expression that evaluates to the escaped string.
        """
        return self.inner().re.replace(
            Lit.ESCAPE_REGEX, Lit.ESCAPE_REPLACE, Lit.G_PARAM
        )

    def find(self, pattern: IntoExprColumn, *, literal: bool = False) -> SqlExpr:
        """Return the first match offset as a zero-based index."""
        match literal:
            case True:
                return self.strpos(pattern).pipe(
                    lambda pos: when(pos.eq(0)).then(Lit.NONE).otherwise(pos.sub(1))
                )
            case False:
                return (
                    self
                    .inner()
                    .re.extract(pattern, 0)
                    .pipe(
                        lambda matched: (
                            when(matched.eq(Lit.EMPTY_STR))
                            .then(Lit.NONE)
                            .otherwise(self.strpos(matched).sub(1))
                        )
                    )
                )

    def join(
        self, delimiter: IntoExprColumn = Lit.EMPTY_STR, *, ignore_nulls: bool = True
    ) -> SqlExpr:
        """Vertically concatenate string values into a single string.

        Args:
            delimiter (IntoExprColumn, optional): The separator to use for joining. Defaults to an empty string.
            ignore_nulls (bool, optional): Whether to ignore null values. Defaults to True.

        Returns:
            SqlExpr: A new expression that evaluates to the joined string.
        """
        aggregated = self.agg(self.inner().new(delimiter))
        match ignore_nulls:
            case True:
                return aggregated
            case False:
                return (
                    when(self.inner().is_null().any())
                    .then(Lit.NONE)
                    .otherwise(aggregated)
                )

    def count_matches(
        self, pattern: IntoExprColumn, *, literal: bool = False
    ) -> SqlExpr:
        """Count pattern matches.

        Returns:
            SqlExpr: A new expression that evaluates to the number of matches.
        """
        pattern_expr = self.inner().new(pattern)
        match literal:
            case False:
                return self.inner().re.extract_all(pattern_expr).list.len()
            case True:
                return (
                    self
                    .length()
                    .sub(self.replace(pattern_expr, Lit.EMPTY_STR).str.length())
                    .truediv(pattern_expr.str.length())
                )

    def strip_prefix(self, prefix: IntoExpr) -> SqlExpr:
        """Strip prefix from string.

        Args:
            prefix (IntoExpr): The prefix to strip.

        Returns:
            SqlExpr: A new expression that evaluates to the string with the prefix removed.
        """
        match prefix:
            case str() as prefix_str:
                return self.inner().re.replace(
                    lit(f"^{re.escape(prefix_str)}"), Lit.EMPTY_STR
                )
            case _:
                return (
                    self
                    .inner()
                    .new(prefix)
                    .pipe(
                        lambda prefix: when(
                            self.inner().str.starts_with(prefix),
                        ).then(self.substring(prefix.str.length().add(1)))
                    )
                    .otherwise(self.inner())
                )

    def strip_suffix(self, suffix: IntoExpr) -> SqlExpr:
        """Strip suffix from string.

        Args:
            suffix (IntoExpr): The suffix to strip.

        Returns:
            SqlExpr: A new expression that evaluates to the string with the suffix removed.
        """
        match suffix:
            case str() as suffix_str:
                return self.inner().re.replace(
                    lit(f"{re.escape(suffix_str)}$"), Lit.EMPTY_STR
                )
            case _:
                return (
                    self
                    .inner()
                    .new(suffix)
                    .pipe(
                        lambda expr: (
                            when(self.inner().str.ends_with(expr))
                            .then(
                                self.substring(1, self.length().sub(expr.str.length()))
                            )
                            .otherwise(self.inner())
                        )
                    )
                )

    def zfill(self, length: IntoExprColumn | int) -> SqlExpr:
        """Pad string values on the left with zeros without truncating.

        Returns:
            SqlExpr: A new expression that evaluates to the zero-filled string.
        """
        expr = self.inner()
        width = expr.new(length, as_col=True)
        signed = expr.str.starts_with(lit("-")).or_(expr.str.starts_with(lit("+")))
        return (
            when(self.length().ge(width))
            .then(expr)
            .when(signed)
            .then(
                self.left(1).str.concat(
                    self.substring(2).str.lpad(width.sub(1), Lit.ZERO)
                )
            )
            .otherwise(self.lpad(width, Lit.ZERO))
        )

    def replace_all(
        self, pattern: IntoExprColumn, value: IntoExprColumn, *, literal: bool = False
    ) -> SqlExpr:
        """Replace all occurrences.

        Args:
            pattern (IntoExprColumn): The pattern to replace.
            value (IntoExprColumn): The value to replace the pattern with.
            literal (bool, optional): Whether to treat the pattern as a literal string. Defaults to False.

        Returns:
            SqlExpr: A new expression that evaluates to the string with the replacements applied.
        """
        match literal:
            case True:
                return self.replace(pattern, value)
            case False:
                return self.inner().re.replace(pattern, value, Lit.G_PARAM)


@dataclass(slots=True)
class SqlExprStructNameSpace(StructFns[SqlExpr]):
    """Struct function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprDateTimeNameSpace(DateTimeFns[SqlExpr]):
    """Datetime function namespace for SQL expressions."""

    def trunc(self, precision: IntoExprColumn) -> SqlExpr:
        """Truncate to specified precision.

        **SQL name**: *date_trunc*

        Args:
            precision (IntoExprColumn): `VARCHAR` expression

        Examples:
            date_trunc('hour', TIMESTAMPTZ '1992-09-20 20:38:40')

        Returns:
            T
        """
        return self._cls(func("DATE_TRUNC", precision, self.inner()))

    def month_start(self) -> SqlExpr:
        """Get the first day of the month.

        Returns:
            SqlExpr: A new expression that evaluates to the first day of the month.
        """
        return self.trunc(Lit.MONTH).add(self.inner().sub(self.trunc(Lit.DAY)))

    def month_end(self) -> SqlExpr:
        """Get the last day of the month.

        Returns:
            SqlExpr: A new expression that evaluates to the last day of the month.
        """
        return self.last_day().add(self.inner().sub(self.trunc(Lit.DAY)))

    def to_datetime(self, format: IntoExprColumn | None = None) -> SqlExpr:  # noqa: A002
        """Parse string values as datetime.

        Args:
            format (IntoExprColumn | None): The format to use for parsing. Defaults to None.

        Returns:
            SqlExpr: A new expression that evaluates to the parsed datetime.
        """
        dtype = exp.DType.TIMESTAMP.into_expr()
        match format:
            case None:
                return self.inner().cast(dtype)
            case _:
                return self.inner().str.strptime(format).cast(dtype)

    def to_time(self, format: IntoExprColumn | None = None) -> SqlExpr:  # noqa: A002
        """Parse string values as time.

        Args:
            format (IntoExprColumn | None): The format to use for parsing. Defaults to None.

        Returns:
            SqlExpr: A new expression that evaluates to the parsed time.
        """
        dtype = exp.DType.TIME.into_expr()
        expr = self.inner()
        match format:
            case None:
                return expr.cast(dtype)
            case _:
                return expr.str.strptime(expr.new(format)).cast(dtype)

    def offset_by(self, by: IntoExpr) -> SqlExpr:
        """Offset datetime by an interval.

        An interval can be specified as a string literal (e.g. '1 day', '2 hours', etc.) or as an expression that evaluates to an interval.

        Args:
            by (IntoExpr): The interval to offset by.

        Returns:
            SqlExpr: A new expression that evaluates to the offset datetime.
        """
        match by:
            case DuckHandler():
                return self.add(exp.to_interval(by.inner()))
            case exp.Expr() | str():
                return self.add(exp.to_interval(by))
            case _:
                return self.add(exp.to_interval(str(by)))


@dataclass(slots=True)
class SqlExprListNameSpace(ListFns[SqlExpr]):
    """List function namespace for SQL expressions."""

    def explode(self) -> SqlExpr:
        """Explode lists into multiple rows.

        Returns:
            SqlExpr: A new expression that evaluates to the exploded rows.
        """
        from ._funcs import unnest

        return unnest(self.inner())

    def eval(self, expr: SqlExpr) -> SqlExpr:
        """Run an expression against each array element.

        Args:
            expr (SqlExpr): The expression to run against each element.

        Returns:
            SqlExpr: A new expression that evaluates to the result of the expression for each element.
        """
        from ._funcs import fn_once

        return self.transform(fn_once(expr))

    def std(self, ddof: int = 1) -> SqlExpr:
        """Compute the standard deviation of the lists in the column.

        Args:
            ddof (int, optional): Delta Degrees of Freedom. Defaults to 1.

        Returns:
            SqlExpr: A new expression that evaluates to the standard deviation of the lists.
        """
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def var(self, ddof: int = 1) -> SqlExpr:
        """Compute the variance of the lists in the column.

        Args:
            ddof (int, optional): Delta Degrees of Freedom. Defaults to 1.

        Returns:
            SqlExpr: A new expression that evaluates to the variance of the lists.
        """
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def filter(self, lambda_arg: IntoExprColumn) -> SqlExpr:
        """Constructs a list from those elements of the input `list` for which the `lambda` function returns `true`.

        DuckDB must be able to cast the `lambda` function's return type to `BOOL`.

        The return type of `list_filter` is the same as the input list's.

        **SQL name**: *filter*

        Args:
            lambda_arg (IntoExprColumn): `LAMBDA` expression

        Examples:
            filter([3, 4, 5], lambda x : x > 4)

        Returns:
            T
        """
        from ._funcs import fn_once

        return self._cls(func("LIST_FILTER", self.inner(), fn_once(lambda_arg)))

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> SqlExpr:
        """Join string values in each list with a separator.

        Args:
            separator (IntoExprColumn): The separator to use for joining.
            ignore_nulls (bool, optional): Whether to ignore null values. Defaults to True.

        Returns:
            SqlExpr: A new expression that evaluates to the joined string.
        """
        joined = self.aggregate(Lit.STR_AGG, separator)
        match ignore_nulls:
            case True:
                return coalesce(joined, Lit.EMPTY_STR)
            case False:
                return (
                    when(self.filter(element().is_null()).list.length().gt(0))
                    .then(Lit.NONE)
                    .otherwise(coalesce(joined, Lit.EMPTY_STR))
                )

    def n_unique(self) -> SqlExpr:
        """Return the number of unique values in each array."""
        return self.distinct().list.length()

    def count_matches(self, elem: IntoExpr) -> SqlExpr:
        """Count matches in each array.

        Args:
            elem (IntoExpr): The element to count matches for.

        Returns:
            SqlExpr: A new expression that evaluates to the number of matches in each array.
        """
        return self.filter(element().eq(elem)).arr.length()


@dataclass(slots=True)
class SqlExprArrayNameSpace(ArrayFns[SqlExpr]):
    """Array function namespace for SQL expressions."""

    def explode(self) -> SqlExpr:
        """Explode array into multiple rows.

        Returns:
            SqlExpr: A new expression that evaluates to the exploded rows.
        """
        from ._funcs import unnest

        return unnest(self.inner())

    def eval(self, expr: SqlExpr) -> SqlExpr:
        """Run an expression against each array element.

        Args:
            expr (SqlExpr): The expression to run against each element.

        Returns:
            SqlExpr: A new expression that evaluates to the result of the expression for each element.
        """
        from ._funcs import fn_once

        return self.transform(fn_once(expr))

    def filter(self, lambda_arg: IntoExprColumn) -> SqlExpr:
        """Constructs a list from those elements of the input `list` for which the `lambda` function returns `true`.

        DuckDB must be able to cast the `lambda` function's return type to `BOOL`.

        The return type of `list_filter` is the same as the input list's.

        **SQL name**: *filter*

        Args:
            lambda_arg (IntoExprColumn): `LAMBDA` expression

        Examples:
            filter([3, 4, 5], lambda x : x > 4)

        Returns:
            T
        """
        from ._funcs import fn_once

        return self._cls(func("ARRAY_FILTER", self.inner(), fn_once(lambda_arg)))

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> SqlExpr:
        """Join string values in each array with a separator.

        Args:
            separator (IntoExprColumn): The separator to use for joining.
            ignore_nulls (bool, optional): Whether to ignore null values. Defaults to True.

        Returns:
            SqlExpr: A new expression that evaluates to the joined string.
        """
        joined = self.aggregate(Lit.STR_AGG, separator)
        match ignore_nulls:
            case True:
                return coalesce(joined, Lit.EMPTY_STR)
            case False:
                return (
                    when(self.filter(element().is_null()).arr.length().gt(0))
                    .then(Lit.NONE)
                    .otherwise(coalesce(joined, Lit.EMPTY_STR))
                )

    def n_unique(self) -> SqlExpr:
        """Return the number of unique values in each array."""
        return self.distinct().arr.length()

    def count_matches(self, elem: IntoExpr) -> SqlExpr:
        """Count matches in each array.

        Args:
            elem (IntoExpr): The element to count matches for.

        Returns:
            SqlExpr: A new expression that evaluates to the number of matches in each array.
        """
        return self.filter(element().eq(elem)).arr.length()


@dataclass(slots=True)
class SqlExprJsonNameSpace(JsonFns[SqlExpr]):
    """JSON function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprRegexNameSpace(RegexFns[SqlExpr]):
    """Regex function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprMapNameSpace(MapFns[SqlExpr]):
    """Map function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprEnumNameSpace(EnumFns[SqlExpr]):
    """Enum function namespace for SQL expressions."""


@dataclass(slots=True)
class SqlExprGeoSpatialNameSpace(GeoSpatialFns[SqlExpr]):
    """Geospatial function namespace for SQL expressions."""
