"""SQL function namespaces for SQL expressions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, final

from pyochain import Seq
from sqlglot import exp

from . import datatypes as dt
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
from ._core import DuckHandler, NameSpaceHandler, func, into_expr
from ._expr import Expr
from ._funcs import element, lit
from ._meta import ExprPlan
from ._when import when

if TYPE_CHECKING:
    from ._meta import Aliaser
    from .typing import (
        EpochTimeUnit,
        IntoExpr,
        IntoExprColumn,
        TimeUnit,
        TransferEncoding,
    )
    from .utils import TryIter


class Sec(IntEnum):
    """Time unit constants and conversions."""

    TO_NANO = 1_000_000_000
    TO_MICRO = 1_000_000
    TO_MILLI = 1_000
    BY_MINUTE = 60
    BY_HOUR = 3_600
    BY_DAY = 86_400

    @classmethod
    def micro_by_day(cls) -> int:
        """Number of microseconds in a day.

        Returns:
            int: The number of microseconds in a day.
        """
        return cls.BY_DAY * cls.TO_MICRO


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
class ExprStringNameSpace(StringFns[Expr]):
    """String function namespace for SQL expressions."""

    def to_titlecase(self) -> Expr:
        """Convert to title case.

        Returns:
            Expr: A new expression that evaluates to the title-cased string.
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

    def escape_regex(self) -> Expr:
        """Escape regex metacharacters without escaping plain spaces.

        Returns:
            Expr: A new expression that evaluates to the escaped string.
        """
        return self.inner.re.replace(Lit.ESCAPE_REGEX, Lit.ESCAPE_REPLACE, Lit.G_PARAM)

    def find(self, pattern: IntoExprColumn, *, literal: bool = False) -> Expr:
        """Return the first match offset as a zero-based index."""
        match literal:
            case True:
                return self.strpos(pattern).pipe(
                    lambda pos: when(pos.eq(0)).then(Lit.NONE).otherwise(pos.sub(1))
                )
            case False:
                return self.inner.re.extract(pattern, 0).pipe(
                    lambda matched: (
                        when(matched.eq(Lit.EMPTY_STR))
                        .then(Lit.NONE)
                        .otherwise(self.strpos(matched).sub(1))
                    )
                )

    def join(
        self, delimiter: IntoExprColumn = Lit.EMPTY_STR, *, ignore_nulls: bool = True
    ) -> Expr:
        """Vertically concatenate string values into a single string.

        Args:
            delimiter (IntoExprColumn, optional): The separator to use for joining. Defaults to an empty string.
            ignore_nulls (bool, optional): Whether to ignore null values. Defaults to True.

        Returns:
            Expr: A new expression that evaluates to the joined string.
        """
        aggregated = self.agg(self.inner.new(delimiter))
        match ignore_nulls:
            case True:
                return aggregated
            case False:
                return (
                    when(self.inner.is_null().any())
                    .then(Lit.NONE)
                    .otherwise(aggregated)
                )

    def count_matches(self, pattern: IntoExprColumn, *, literal: bool = False) -> Expr:
        """Count pattern matches.

        Returns:
            Expr: A new expression that evaluates to the number of matches.
        """
        expr = self.inner
        pattern_expr = expr.new(pattern)
        match literal:
            case False:
                return expr.re.extract_all(pattern_expr).list.len()
            case True:
                return (
                    self
                    .length()
                    .sub(self.replace(pattern_expr, Lit.EMPTY_STR).str.length())
                    .truediv(pattern_expr.str.length())
                )

    def strip_prefix(self, prefix: IntoExpr) -> Expr:
        """Strip prefix from string.

        Args:
            prefix (IntoExpr): The prefix to strip.

        Returns:
            Expr: A new expression that evaluates to the string with the prefix removed.
        """
        expr = self.inner
        match prefix:
            case str() as prefix_str:
                return expr.re.replace(lit(f"^{re.escape(prefix_str)}"), Lit.EMPTY_STR)
            case _:
                return (
                    expr
                    .new(prefix)
                    .pipe(
                        lambda prefix: when(
                            expr.str.starts_with(prefix),
                        ).then(self.substring(prefix.str.length().add(1)))
                    )
                    .otherwise(expr)
                )

    def strip_suffix(self, suffix: IntoExpr) -> Expr:
        """Strip suffix from string.

        Args:
            suffix (IntoExpr): The suffix to strip.

        Returns:
            Expr: A new expression that evaluates to the string with the suffix removed.
        """
        expr = self.inner
        match suffix:
            case str() as suffix_str:
                return expr.re.replace(lit(f"{re.escape(suffix_str)}$"), Lit.EMPTY_STR)
            case _:
                return expr.new(suffix).pipe(
                    lambda sfx: (
                        when(expr.str.ends_with(sfx))
                        .then(self.substring(1, self.length().sub(sfx.str.length())))
                        .otherwise(expr)
                    )
                )

    def zfill(self, length: IntoExprColumn | int) -> Expr:
        """Pad string values on the left with zeros without truncating.

        Returns:
            Expr: A new expression that evaluates to the zero-filled string.
        """
        expr = self.inner
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

    def concat(self, *args: IntoExpr) -> Expr:
        """Concatenates multiple strings or lists.

        `NULL` inputs are skipped.

        See also operator `||`.

        **SQL name**: *concat*

        Args:
            *args (IntoExpr): `ANY` expression

        Returns:
            Expr
        """
        return self._cls(func("CONCAT", self.inner, *args))

    def replace_all(
        self, pattern: IntoExprColumn, value: IntoExprColumn, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences.

        Args:
            pattern (IntoExprColumn): The pattern to replace.
            value (IntoExprColumn): The value to replace the pattern with.
            literal (bool, optional): Whether to treat the pattern as a literal string. Defaults to False.

        Returns:
            Expr: A new expression that evaluates to the string with the replacements applied.
        """
        match literal:
            case True:
                return self.replace(pattern, value)
            case False:
                return self.inner.re.replace(pattern, value, Lit.G_PARAM)

    def normalize(self) -> Expr:
        """Normalize strings using NFC normalization.

        Returns:
            Expr
        """
        return self.nfc_normalize()

    def to_decimal(self, *, scale: int) -> Expr:
        """Parse string values as decimal with the requested scale.

        Returns:
            Expr
        """
        return self.inner.cast(dt.Decimal(scale=scale))

    def to_datetime(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as datetime.

        Returns:
            Expr
        """
        return self.inner.dt.to_datetime(format)

    def to_time(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as time.

        Returns:
            Expr
        """
        return self.inner.dt.to_time(format)

    def to_date(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as date.

        Returns:
            Expr
        """
        match format:
            case None:
                return self.inner.cast(dt.Date())
            case _:
                return self.strptime(format).cast(dt.Date())

    def pad_start(self, length: int, fill_char: IntoExprColumn = Lit.ESCAPE) -> Expr:
        """Pad string values on the left.

        Returns:
            Expr: A new expression that evaluates to the left-padded string.
        """
        return self.lpad(length, fill_char)

    def pad_end(self, length: int, fill_char: IntoExprColumn = Lit.ESCAPE) -> Expr:
        """Pad string values on the right.

        Returns:
            Expr
        """
        return self.rpad(length, fill_char)

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring.

        Returns:
            Expr
        """
        return self.substring(offset + 1, length)

    def len_bytes(self) -> Expr:
        """Get the length in bytes.

        Returns:
            Expr
        """
        return self.inner.encode().octet_length()

    def encode(self, encoding: TransferEncoding = "base64") -> Expr:
        """Encode UTF-8 strings as binary values.

        Returns:
            Expr
        """
        expr = self.inner.encode().str
        match encoding:
            case "base64":
                return expr.to_base64()
            case "hex":
                return expr.to_hex().str.lower()

    def extract_all(self, pattern: IntoExprColumn) -> Expr:
        """Extract all regex matches.

        Returns:
            Expr
        """
        return self.inner.re.extract_all(pattern)

    def extract(self, pattern: IntoExprColumn, group_index: int = 1) -> Expr:
        """Extract a regex capture group.

        Returns:
            Expr
        """
        return self.inner.re.extract(pattern, group_index)

    def json_path_match(self, json_path: IntoExprColumn) -> Expr:
        """Extract first JSONPath match from string JSON values.

        Returns:
            Expr
        """
        return self.inner.json.extract_string(json_path)

    def to_uppercase(self) -> Expr:
        """Convert to uppercase.

        Returns:
            Expr
        """
        return self.upper()

    def to_lowercase(self) -> Expr:
        """Convert to lowercase.

        Returns:
            Expr
        """
        return self.lower()

    def len_chars(self) -> Expr:
        """Get the length in characters.

        Returns:
            Expr
        """
        return self.length()

    def strip_chars(self, characters: IntoExprColumn | None = None) -> Expr:
        """Strip leading and trailing characters.

        Returns:
            Expr
        """
        return self.trim(characters)

    def strip_chars_start(self, characters: IntoExprColumn | None = None) -> Expr:
        """Strip leading characters.

        Returns:
            Expr
        """
        return self.ltrim(characters)

    def strip_chars_end(self, characters: IntoExprColumn | None = None) -> Expr:
        """Strip trailing characters.

        Returns:
            Expr
        """
        return self.rtrim(characters)

    def head(self, n: int) -> Expr:
        """Get first n characters.

        Returns:
            Expr
        """
        return self.left(n)

    def tail(self, n: int) -> Expr:
        """Get last n characters.

        Returns:
            Expr
        """
        return self.right(n)


@dataclass(slots=True)
class ExprStructNameSpace(StructFns[Expr]):
    """Struct function namespace for SQL expressions."""

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name.

        Returns:
            Expr
        """
        return self.inner.struct.extract(lit(name)).alias(name)

    def json_encode(self) -> Expr:
        """Encode struct values as JSON strings.

        Returns:
            Expr
        """
        return self.inner.to_json()

    def with_fields(
        self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr, **named_exprs: IntoExpr
    ) -> Expr:
        """Return a new struct with updated or additional fields.

        Returns:
            Expr
        """
        return (
            Seq[str]
            .new()
            .into(ExprPlan, exprs, more_exprs, named_exprs)
            .with_fields_ctx(self.inner)
        )


@dataclass(slots=True)
class ExprDateTimeNameSpace(DateTimeFns[Expr]):
    """Datetime function namespace for SQL expressions."""

    def month_start(self) -> Expr:
        """Get the first day of the month.

        Returns:
            Expr: A new expression that evaluates to the first day of the month.
        """
        return self.trunc(Lit.MONTH).add(self.inner.sub(self.trunc(Lit.DAY)))

    def month_end(self) -> Expr:
        """Get the last day of the month.

        Returns:
            Expr: A new expression that evaluates to the last day of the month.
        """
        return self.last_day().add(self.inner.sub(self.trunc(Lit.DAY)))

    def to_datetime(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as datetime.

        Args:
            format (IntoExprColumn | None): The format to use for parsing. Defaults to None.

        Returns:
            Expr: A new expression that evaluates to the parsed datetime.
        """
        dtype = exp.DType.TIMESTAMP.into_expr()
        match format:
            case None:
                return self.inner.cast(dtype)
            case _:
                return self.inner.str.strptime(format).cast(dtype)

    def to_time(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as time.

        Args:
            format (IntoExprColumn | None): The format to use for parsing. Defaults to None.

        Returns:
            Expr: A new expression that evaluates to the parsed time.
        """
        dtype = exp.DType.TIME.into_expr()
        expr = self.inner
        match format:
            case None:
                return expr.cast(dtype)
            case _:
                return expr.str.strptime(expr.new(format)).cast(dtype)

    def offset_by(self, by: IntoExpr) -> Expr:
        """Offset datetime by an interval.

        An interval can be specified as a string literal (e.g. '1 day', '2 hours', etc.) or as an expression that evaluates to an interval.

        Args:
            by (IntoExpr): The interval to offset by.

        Returns:
            Expr: A new expression that evaluates to the offset datetime.
        """
        match by:
            case DuckHandler():
                return self.add(exp.to_interval(by.inner))
            case exp.Expr() | str():
                return self.add(exp.to_interval(by))
            case _:
                return self.add(exp.to_interval(str(by)))

    def truncate(self, every: str) -> Expr:
        """Truncate datetime to the nearest multiple of a specified time unit.

        Alias for `trunc`.

        Args:
            every (str): The time unit to truncate to (e.g. 'hour', 'day', 'minute', etc.).

        Returns:
            Expr: A new expression that evaluates to the truncated datetime.
        """
        return self.trunc(lit(every))

    def round(self, every: str) -> Expr:
        """Round datetime to the nearest multiple of a specified time unit.

        Alias for `trunc`.

        Args:
            every (str): The time unit to round to (e.g. 'hour', 'day', 'minute', etc.).

        Returns:
            Expr: A new expression that evaluates to the rounded datetime.
        """
        return self.trunc(lit(every))

    def trunc(self, precision: IntoExprColumn) -> Expr:
        """Truncate to specified precision.

        **SQL name**: *date_trunc*

        Args:
            precision (IntoExprColumn): `VARCHAR` expression

        Examples:
            date_trunc('hour', TIMESTAMPTZ '1992-09-20 20:38:40')

        Returns:
            T
        """
        return self._cls(func("DATE_TRUNC", precision, self.inner))

    def date(self) -> Expr:
        """Returns a new expression that evaluates to the date component of the datetime.

        Returns:
            Expr
        """
        return self.inner.cast(dt.Date())

    def time(self) -> Expr:
        """Returns a new expression that evaluates to the time component of the datetime.

        Returns:
            Expr
        """
        return self.inner.cast(dt.Time())

    def to_string(self, format: IntoExprColumn) -> Expr:  # noqa: A002
        """Format datetime as string.

        Args:
            format (IntoExprColumn): The format to use for formatting.

        Returns:
            Expr: A new expression that evaluates to the formatted string.
        """
        return self.inner.str.strftime(format)

    def ordinal_day(self) -> Expr:
        """Extract the dayofyear component from a date or timestamp.

        **SQL name**: *dayofyear*

        Examples:
            dayofyear(timestamp '2021-08-03 11:59:44.123456')

        Returns:
            Expr
        """
        return self.dayofyear()

    def epoch_by(self, time_unit: EpochTimeUnit = "us") -> Expr:
        """Get the time passed since the Unix EPOCH in the give time unit.

        Args:
            time_unit (EpochTimeUnit): The time unit to use for the epoch time. Defaults to "us".

        Returns:
            Expr
        """
        match time_unit:
            case "d":
                return self.epoch_us().truediv(Sec.micro_by_day()).floor()
            case "s":
                return self.epoch_us().truediv(Sec.TO_MICRO).floor()
            case _:
                return self.timestamp(time_unit)

    def timestamp(self, time_unit: TimeUnit = "us") -> Expr:
        """Return the number of time units since the Unix epoch.

        Args:
            time_unit (TimeUnit): The time unit to use for the epoch time. Defaults to "us".

        Returns:
            Expr: A new expression that evaluates to the epoch time in the specified time unit.
        """
        match time_unit:
            case "ms":
                return self.epoch_ms()
            case "us":
                return self.epoch_us()
            case "ns":
                return self.epoch_ns()

    def iso_year(self) -> Expr:
        """Extract the isoyear component from a date or timestamp.

        **SQL name**: *isoyear*

        Examples:
            isoyear(timestamp '2021-08-03 11:59:44.123456')

        Returns:
            Expr
        """
        return self.isoyear()


@final
class ExprListNameSpace(ListFns[Expr]):
    """List function namespace for SQL expressions."""

    __slots__ = ()

    def explode(self) -> Expr:
        """Explode lists into multiple rows.

        Returns:
            Expr: A new expression that evaluates to the exploded rows.
        """
        from ._funcs import unnest

        return unnest(self.inner)

    def eval(self, expr: Expr) -> Expr:
        """Run an expression against each array element.

        Args:
            expr (Expr): The expression to run against each element.

        Returns:
            Expr: A new expression that evaluates to the result of the expression for each element.
        """
        from ._funcs import fn_once

        return self.transform(fn_once(expr))

    def std(self, ddof: int = 1) -> Expr:
        """Compute the standard deviation of the lists in the column.

        Args:
            ddof (int, optional): Delta Degrees of Freedom. Defaults to 1.

        Returns:
            Expr: A new expression that evaluates to the standard deviation of the lists.
        """
        match ddof:
            case 0:
                return self.stddev_pop()
            case _:
                return self.stddev_samp()

    def var(self, ddof: int = 1) -> Expr:
        """Compute the variance of the lists in the column.

        Args:
            ddof (int, optional): Delta Degrees of Freedom. Defaults to 1.

        Returns:
            Expr: A new expression that evaluates to the variance of the lists.
        """
        match ddof:
            case 0:
                return self.var_pop()
            case _:
                return self.var_samp()

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> Expr:
        """Join string values in each list with a separator.

        Args:
            separator (IntoExprColumn): The separator to use for joining.
            ignore_nulls (bool, optional): Whether to ignore null values. Defaults to True.

        Returns:
            Expr: A new expression that evaluates to the joined string.
        """
        joined = self.aggregate(Lit.STR_AGG, separator).coalesce(Lit.EMPTY_STR)
        match ignore_nulls:
            case True:
                return joined
            case False:
                return (
                    when(self.filter(element().is_null()).list.length().gt(0))
                    .then(Lit.NONE)
                    .otherwise(joined)
                )

    def count_matches(self, elem: IntoExpr) -> Expr:
        """Count matches in each array.

        Args:
            elem (IntoExpr): The element to count matches for.

        Returns:
            Expr: A new expression that evaluates to the number of matches in each array.
        """
        return self.filter(element().eq(into_expr(elem, as_col=False))).list.length()

    def drop_nulls(self) -> Expr:
        """Drop null values in each list.

        Returns:
            Expr
        """
        return self.filter(element().is_not_null())

    def filter(self, lambda_arg: IntoExprColumn) -> Expr:
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

        return self._cls(func("LIST_FILTER", self.inner, fn_once(lambda_arg)))

    def get(self, index: int) -> Expr:
        """Return the value by index in each list.

        Returns:
            Expr
        """
        return self.extract(index + 1 if index >= 0 else index)

    def all(self) -> Expr:
        """Return whether all values in the list are true.

        Returns:
            Expr
        """
        return self.bool_and()

    def any(self) -> Expr:
        """Return whether any value in the listis true.

        Returns:
            Expr
        """
        return self.bool_or()

    def mean(self) -> Expr:
        """Compute the mean value of the arrays in the column.

        Returns:
            Expr
        """
        return self.avg()

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sorts the elements of the list.

        **SQL name**: *list_sort*

        See Also:
            array_sort

        Args:
            descending (bool): Whether to sort in descending order.
            nulls_last (bool): Whether to place nulls last.

        Examples:
            ```sql
            list_sort([3, 6, 1, 2])
            ```

        Returns:
            T
        """
        return self._cls(_sort_expr(self.inner, desc=descending, nulls_last=nulls_last))

    def unique(self) -> Expr:
        """Removes all duplicates and NULL values from a list.

        Does not preserve the original order.

        **SQL name**: *list_distinct*

        See Also:
            array_distinct

        Examples:
            list_distinct([1, 1, NULL, -3, 1, 5])

        Returns:
            Expr
        """
        return self.distinct()


@final
class ExprArrayNameSpace(ArrayFns[Expr]):
    """Array function namespace for SQL expressions."""

    __slots__ = ()

    def explode(self) -> Expr:
        """Explode array into multiple rows.

        Returns:
            Expr: A new expression that evaluates to the exploded rows.
        """
        from ._funcs import unnest

        return unnest(self.inner)

    def eval(self, expr: Expr) -> Expr:
        """Run an expression against each array element.

        Args:
            expr (Expr): The expression to run against each element.

        Returns:
            Expr: A new expression that evaluates to the result of the expression for each element.
        """
        from ._funcs import fn_once

        return self.transform(fn_once(expr))

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> Expr:
        """Join string values in each array with a separator.

        Args:
            separator (IntoExprColumn): The separator to use for joining.
            ignore_nulls (bool, optional): Whether to ignore null values. Defaults to True.

        Returns:
            Expr: A new expression that evaluates to the joined string.
        """
        joined = self.aggregate(Lit.STR_AGG, separator).coalesce(Lit.EMPTY_STR)
        match ignore_nulls:
            case True:
                return joined
            case False:
                return (
                    when(self.filter(element().is_null()).arr.length().gt(0))
                    .then(Lit.NONE)
                    .otherwise(joined)
                )

    def count_matches(self, elem: IntoExpr) -> Expr:
        """Count matches in each array.

        Args:
            elem (IntoExpr): The element to count matches for.

        Returns:
            Expr: A new expression that evaluates to the number of matches in each array.
        """
        return self.filter(element().eq(into_expr(elem, as_col=False))).arr.length()

    def drop_nulls(self) -> Expr:
        """Drop null values in each array.

        Returns:
            Expr
        """
        return self.filter(element().is_not_null())

    def filter(self, lambda_arg: IntoExprColumn) -> Expr:
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

        return self._cls(func("ARRAY_FILTER", self.inner, fn_once(lambda_arg)))

    def get(self, index: int) -> Expr:
        """Return the value by index in each array.

        Returns:
            Expr
        """
        return self.extract(index + 1 if index >= 0 else index)

    def all(self) -> Expr:
        """Return whether all values in the list are true.

        Returns:
            Expr
        """
        return self.inner.list.all()

    def any(self) -> Expr:
        """Return whether any value in the listis true.

        Returns:
            Expr
        """
        return self.inner.list.any()

    def len(self) -> Expr:
        """Return the number of elements in each array.

        Returns:
            Expr
        """
        return self.length()

    def first(self) -> Expr:
        """Get the first element of each array.

        Returns:
            Expr
        """
        return self.inner.list.first()

    def last(self) -> Expr:
        """Get the last element of each array.

        Returns:
            Expr
        """
        return self.inner.list.last()

    def min(self) -> Expr:
        """Compute the min value of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.min()

    def max(self) -> Expr:
        """Compute the max value of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.max()

    def mean(self) -> Expr:
        """Compute the mean value of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.mean()

    def median(self) -> Expr:
        """Compute the median value of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.median()

    def sum(self) -> Expr:
        """Compute the sum value of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.sum()

    def std(self, ddof: int = 1) -> Expr:
        """Compute the standard deviation of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.std(ddof)

    def var(self, ddof: int = 1) -> Expr:
        """Compute the variance of the arrays in the column.

        Returns:
            Expr
        """
        return self.inner.list.var(ddof)

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sorts the elements of the list.

        **SQL name**: *array_sort*

        See Also:
            list_sort

        Args:
            descending (bool): Whether to sort in descending order.
            nulls_last (bool): Whether to place nulls last.

        Examples:
            ```sql
            array_sort([3, 6, 1, 2])
            ```

        Returns:
            T
        """
        return self._cls(_sort_expr(self.inner, desc=descending, nulls_last=nulls_last))

    def unique(self) -> Expr:
        """Removes all duplicates and NULL values from a list.

        Does not preserve the original order.

        **SQL name**: *array_distinct*

        See Also:
            list_distinct

        Examples:
            array_distinct([1, 1, NULL, -3, 1, 5])

        Returns:
            Expr
        """
        return self.distinct()


@dataclass(slots=True)
class ExprNameNameSpace(NameSpaceHandler[Expr]):
    """Name operations namespace (equivalent to pl.Expr.name)."""

    def keep(self) -> Expr:
        """Keep the original name of the expression, even if it gets aliased.

        Returns:
            Expr
        """
        expr = self.inner
        meta = expr.meta.unalias()
        return expr.inner.unalias().pipe(Expr, meta)

    def map(self, function: Aliaser) -> Expr:
        """Map the expression name using the provided function.

        Returns:
            Expr
        """
        return self._with_alias_mapper(function)

    def prefix(self, prefix: str) -> Expr:
        """Prefix the expression name with the given string.

        Returns:
            Expr
        """
        return self._with_alias_mapper(lambda name: f"{prefix}{name}")

    def suffix(self, suffix: str) -> Expr:
        """Suffix the expression name with the given string.

        Returns:
            Expr
        """
        return self._with_alias_mapper(lambda name: f"{name}{suffix}")

    def to_lowercase(self) -> Expr:
        """Convert the expression name to lowercase.

        Returns:
            Expr
        """
        return self._with_alias_mapper(str.lower)

    def to_uppercase(self) -> Expr:
        """Convert the expression name to uppercase.

        Returns:
            Expr
        """
        return self._with_alias_mapper(str.upper)

    def replace(self, pattern: str, value: str, *, literal: bool = False) -> Expr:
        """Replace occurrences of a pattern in the expression name with a value.

        Args:
            pattern (str): The pattern to replace.
            value (str): The value to replace the pattern with.
            literal (bool): Whether to treat the pattern as a literal string. Defaults to False.

        Returns:
            Expr
        """
        match literal:
            case True:
                return self._with_alias_mapper(
                    lambda name: name.replace(pattern, value)
                )
            case False:
                regex = re.compile(pattern)
                return self._with_alias_mapper(lambda name: regex.sub(value, name))

    def _with_alias_mapper(self, mapper: Aliaser) -> Expr:
        expr = self.inner
        return expr.inner.pipe(Expr, expr.meta.with_alias_mapper(mapper))


@dataclass(slots=True)
class ExprJsonNameSpace(JsonFns[Expr]):
    """JSON function namespace for SQL expressions."""


@dataclass(slots=True)
class ExprRegexNameSpace(RegexFns[Expr]):
    """Regex function namespace for SQL expressions."""


@dataclass(slots=True)
class ExprMapNameSpace(MapFns[Expr]):
    """Map function namespace for SQL expressions."""


@dataclass(slots=True)
class ExprEnumNameSpace(EnumFns[Expr]):
    """Enum function namespace for SQL expressions."""


@dataclass(slots=True)
class ExprGeoSpatialNameSpace(GeoSpatialFns[Expr]):
    """Geospatial function namespace for SQL expressions."""


def _sort_expr(expr: Expr, *, desc: bool, nulls_last: bool) -> exp.SortArray:
    return exp.SortArray(
        this=expr.inner,
        asc=exp.Boolean(this=not desc),
        nulls_first=exp.Boolean(this=not nulls_last),
    )
