from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, override

import pyochain as pc

from . import _datatypes as dt, sql  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr
from ._meta import Aliaser, ExprPlan
from .sql import SqlExpr, namespaces as nm

if TYPE_CHECKING:
    from ._typing import TransferEncoding
    from .sql.typing import EpochTimeUnit, IntoExpr, IntoExprColumn, TimeUnit
    from .sql.utils import TryIter


class Lit:
    TITLECASE: SqlExpr = sql.lit(r"[a-z]*[^a-z]*")
    NONE: SqlExpr = sql.lit(None)
    G_PARAM: SqlExpr = sql.lit("g")
    EMPTY_STR: SqlExpr = sql.lit("")
    ESCAPE_REGEX: SqlExpr = sql.lit(r"([.^$*+?{}\[\]\\|()])")
    ESCAPE_REPLACE: SqlExpr = sql.lit(r"\\\1")
    ESCAPE: SqlExpr = sql.lit(" ")
    STR_AGG: SqlExpr = sql.lit("string_agg")
    DAY: SqlExpr = sql.lit("day")
    MONTH: SqlExpr = sql.lit("month")
    ZERO: SqlExpr = sql.lit("0")


class Sec(IntEnum):
    TO_NANO = 1_000_000_000
    TO_MICRO = 1_000_000
    TO_MILLI = 1_000
    BY_MINUTE = 60
    BY_HOUR = 3_600
    BY_DAY = 86_400

    @classmethod
    def micro_by_day(cls) -> int:
        return cls.BY_DAY * cls.TO_MICRO


@dataclass(slots=True)
class ExprNameSpaceBase(sql.CoreHandler[Expr]):
    @override
    def _cls(self, value: SqlExpr) -> Expr:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.inner()._cls(value)  # pyright: ignore[reportPrivateUsage]


@dataclass(slots=True)
class ExprStringNameSpace(ExprNameSpaceBase):
    """String operations namespace (equivalent to pl.Expr.str).

    Returns:
        Expr
    """

    def join(
        self, delimiter: IntoExprColumn = Lit.EMPTY_STR, *, ignore_nulls: bool = True
    ) -> Expr:
        """Vertically concatenate string values into a single string.

        Returns:
            Expr
        """
        return self._cls(
            self.inner().inner().str.join(delimiter, ignore_nulls=ignore_nulls)
        )

    def escape_regex(self) -> Expr:
        """Escape all regex meta characters in the string.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.escape_regex())

    def to_uppercase(self) -> Expr:
        """Convert to uppercase.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.upper())

    def to_lowercase(self) -> Expr:
        """Convert to lowercase.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.lower())

    def len_chars(self) -> Expr:
        """Get the length in characters.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.length())

    def contains(self, pattern: IntoExprColumn, *, literal: bool = False) -> Expr:
        """Check if string contains a pattern.

        Returns:
            Expr
        """
        expr = self.inner().inner()
        match literal:
            case True:
                return self._cls(expr.str.contains(expr.new(pattern)))
            case False:
                return self._cls(expr.re.matches(expr.new(pattern)))

    def starts_with(self, prefix: IntoExprColumn) -> Expr:
        """Check if string starts with prefix.

        Returns:
            Expr
        """
        expr = self.inner().inner()
        return self._cls(expr.str.starts_with(expr.new(prefix)))

    def ends_with(self, suffix: IntoExprColumn) -> Expr:
        """Check if string ends with suffix.

        Returns:
            Expr
        """
        expr = self.inner().inner()
        return self._cls(expr.str.ends_with(expr.new(suffix)))

    def replace(
        self, pattern: str, value: IntoExprColumn, *, literal: bool = False, n: int = 1
    ) -> Expr:
        """Replace first matching substring with a new string value.

        Returns:
            Expr
        """
        pattern_expr = sql.lit(re.escape(pattern) if literal else pattern)

        def _replace_once(expr: SqlExpr) -> SqlExpr:
            return expr.str.replace(pattern_expr, value)

        expr = self.inner().inner()
        match n:
            case 0:
                return self._cls(expr)
            case n_val if n_val < 0:
                return self._cls(expr.re.replace(pattern_expr, value, Lit.G_PARAM))
            case _:
                return (
                    pc
                    .Iter(range(n))
                    .fold(expr, lambda acc, _: _replace_once(acc))
                    .pipe(self._cls)
                )

    def _strip_fn(
        self,
        characters: IntoExprColumn | None,
        fn: Callable[[SqlExpr, SqlExpr | None], SqlExpr],
    ) -> Expr:
        expr = self.inner().inner()
        match characters:
            case None:
                return self._cls(fn(expr, None))
            case _:
                return self._cls(fn(expr, expr.new(characters)))

    def strip_chars(self, characters: IntoExprColumn | None = None) -> Expr:
        """Strip leading and trailing characters.

        Returns:
            Expr
        """
        return self._strip_fn(characters, lambda e, c: e.str.trim(c))

    def strip_chars_start(self, characters: IntoExprColumn | None = None) -> Expr:
        """Strip leading characters.

        Returns:
            Expr
        """
        return self._strip_fn(characters, lambda e, c: e.str.ltrim(c))

    def strip_chars_end(self, characters: IntoExprColumn | None = None) -> Expr:
        """Strip trailing characters.

        Returns:
            Expr
        """
        return self._strip_fn(characters, lambda e, c: e.str.rtrim(c))

    def slice(self, offset: int, length: int | None = None) -> Expr:
        """Extract a substring.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.substring(offset + 1, length))

    def len_bytes(self) -> Expr:
        """Get the length in bytes.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().encode().octet_length())

    def split(self, by: IntoExprColumn) -> Expr:
        """Split string by separator.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.split(by))

    def extract_all(self, pattern: IntoExprColumn) -> Expr:
        """Extract all regex matches.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().re.extract_all(pattern))

    def extract(self, pattern: IntoExprColumn, group_index: int = 1) -> Expr:
        """Extract a regex capture group.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().re.extract(pattern, group_index))

    def find(self, pattern: IntoExprColumn, *, literal: bool = False) -> Expr:
        """Return the first match offset as a zero-based index.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.find(pattern, literal=literal))

    def json_path_match(self, json_path: IntoExprColumn) -> Expr:
        """Extract first JSONPath match from string JSON values.

        Returns:
            Expr
        """
        expr = self.inner().inner()
        return self._cls(expr.json.extract_string(expr.new(json_path)))

    def to_date(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as date.

        Returns:
            Expr
        """
        match format:
            case None:
                return self.inner().cast(dt.Date())
            case _:
                return self._cls(self.inner().inner().str.strptime(format)).cast(
                    dt.Date()
                )

    def to_datetime(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as datetime.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().dt.to_datetime(format))

    def to_time(self, format: IntoExprColumn | None = None) -> Expr:  # noqa: A002
        """Parse string values as time.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().dt.to_time(format))

    def strptime(self, format: IntoExprColumn) -> Expr:  # noqa: A002
        """Parse string values into datetime using one or more formats.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.strptime(format))

    def encode(self, encoding: TransferEncoding = "base64") -> Expr:
        """Encode UTF-8 strings as binary values.

        Returns:
            Expr
        """
        match encoding:
            case "base64":
                return self._cls(self.inner().inner().encode().str.to_base64())
            case "hex":
                return self._cls(self.inner().inner().encode().str.to_hex().str.lower())

    def normalize(self) -> Expr:
        """Normalize strings using NFC normalization.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.nfc_normalize())

    def to_decimal(self, *, scale: int) -> Expr:
        """Parse string values as decimal with the requested scale.

        Returns:
            Expr
        """
        return self.inner().cast(dt.Decimal(scale=scale))

    def count_matches(self, pattern: IntoExprColumn, *, literal: bool = False) -> Expr:
        """Count pattern matches.

        Returns:
            Expr
        """
        return self._cls(
            self.inner().inner().str.count_matches(pattern, literal=literal)
        )

    def strip_prefix(self, prefix: IntoExpr) -> Expr:
        """Strip prefix from string.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.strip_prefix(prefix))

    def strip_suffix(self, suffix: IntoExpr) -> Expr:
        """Strip suffix from string.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.strip_suffix(suffix))

    def head(self, n: int) -> Expr:
        """Get first n characters.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.left(n))

    def tail(self, n: int) -> Expr:
        """Get last n characters.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.right(n))

    def reverse(self) -> Expr:
        """Reverse the string.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.reverse())

    def pad_start(self, length: int, fill_char: IntoExprColumn = Lit.ESCAPE) -> Expr:
        expr = self.inner().inner()
        return self._cls(expr.str.lpad(length, expr.new(fill_char)))

    def pad_end(self, length: int, fill_char: IntoExprColumn = Lit.ESCAPE) -> Expr:
        expr = self.inner().inner()
        return self._cls(expr.str.rpad(length, expr.new(fill_char)))

    def zfill(self, length: int) -> Expr:
        return self._cls(self.inner().inner().str.zfill(length))

    def replace_all(
        self, pattern: IntoExprColumn, value: IntoExprColumn, *, literal: bool = False
    ) -> Expr:
        """Replace all occurrences.

        Returns:
            Expr
        """
        return self._cls(
            self.inner().inner().str.replace_all(pattern, value, literal=literal)
        )

    def to_titlecase(self) -> Expr:
        """Convert to title case.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().str.to_titlecase())


@dataclass(slots=True)
class _VecNameSpace[T: nm.SqlExprListNameSpace | nm.SqlExprArrayNameSpace](
    ExprNameSpaceBase, ABC
):
    """Common list/array operations namespace.

    Returns:
        Expr
    """

    @property
    @abstractmethod
    def _vec(self) -> T:  # pragma: no cover
        raise NotImplementedError

    def eval(self, expr: Expr) -> Expr:
        """Run an expression against each array element.

        Returns:
            Expr
        """
        return self._cls(self._vec.eval(expr.inner()))

    def filter(self, predicate: Expr) -> Expr:
        return self._cls(self._vec.filter(predicate.inner()))

    def drop_nulls(self) -> Expr:
        """Drop null values in each list.

        Returns:
            Expr
        """
        return self._cls(self._vec.filter(sql.element().is_not_null()))

    def contains(self, item: IntoExpr) -> Expr:
        """Check if subarrays contain the given item.

        Returns:
            Expr
        """
        return self._cls(self._vec.contains(item))

    def all(self) -> Expr:
        """Return whether all values in the list are true.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.bool_and())

    def any(self) -> Expr:
        """Return whether any value in the listis true.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.bool_or())

    def len(self) -> Expr:
        """Return the number of elements in each array.

        Returns:
            Expr
        """
        return self._cls(self._vec.length())

    def unique(self) -> Expr:
        """Return unique values in each array.

        Returns:
            Expr
        """
        return self._cls(self._vec.distinct())

    def reverse(self) -> Expr:
        """Reverse the arrays of the expression.

        Returns:
            Expr
        """
        return self._cls(self._vec.reverse())

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        """Sort the lists of the column.

        Returns:
            Expr
        """
        return self._cls(
            self._vec.sort(
                sql.lit(sql.SortClause.order(desc=descending)),
                sql.lit(sql.NullsClause.order(last=nulls_last)),
            )
        )

    def first(self) -> Expr:
        """Get the first element of each array.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.first())

    def last(self) -> Expr:
        """Get the last element of each array.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.last())

    def min(self) -> Expr:
        """Compute the min value of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.min())

    def max(self) -> Expr:
        """Compute the max value of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.max())

    def mean(self) -> Expr:
        """Compute the mean value of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.avg())

    def median(self) -> Expr:
        """Compute the median value of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.median())

    def sum(self) -> Expr:
        """Compute the sum value of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.sum())

    def std(self, ddof: int = 1) -> Expr:
        """Compute the standard deviation of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.std(ddof))

    def var(self, ddof: int = 1) -> Expr:
        """Compute the variance of the arrays in the column.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().list.var(ddof))

    def get(self, index: int) -> Expr:
        """Return the value by index in each array.

        Returns:
            Expr
        """
        return self._cls(self._vec.extract(index + 1 if index >= 0 else index))

    def join(self, separator: IntoExprColumn, *, ignore_nulls: bool = True) -> Expr:
        return self._cls(self._vec.join(separator, ignore_nulls=ignore_nulls))

    def n_unique(self) -> Expr:
        return self._cls(self._vec.n_unique())

    def count_matches(self, element: IntoExpr) -> Expr:
        return self._cls(self._vec.count_matches(element))

    def explode(self) -> Expr:
        """Explode arrays into multiple rows.

        Returns:
            Expr
        """
        return self._cls(self._vec.explode())


@dataclass(slots=True)
class ExprArrayNameSpace(_VecNameSpace[nm.SqlExprArrayNameSpace]):
    """Array operations namespace (equivalent to pl.Expr.array).

    Returns:
        Expr
    """

    @property
    @override
    def _vec(self) -> nm.SqlExprArrayNameSpace:
        return self.inner().inner().arr


@dataclass(slots=True)
class ExprListNameSpace(_VecNameSpace[nm.SqlExprListNameSpace]):
    """List operations namespace (equivalent to pl.Expr.list).

    Returns:
        Expr
    """

    @property
    @override
    def _vec(self) -> nm.SqlExprListNameSpace:
        return self.inner().inner().list


@dataclass(slots=True)
class ExprStructNameSpace(ExprNameSpaceBase):
    """Struct operations namespace (equivalent to pl.Expr.struct).

    Returns:
        Expr
    """

    def field(self, name: str) -> Expr:
        """Retrieve a struct field by name.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().struct.extract(sql.lit(name))).alias(name)

    def json_encode(self) -> Expr:
        """Encode struct values as JSON strings.

        Returns:
            Expr
        """
        return self._cls(self.inner().inner().to_json())

    def with_fields(
        self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr, **named_exprs: IntoExpr
    ) -> Expr:
        """Return a new struct with updated or additional fields.

        Returns:
            Expr
        """
        return (
            ExprPlan(pc.Seq[str].new(), exprs, more_exprs, named_exprs)
            .with_fields_ctx(self.inner().inner())
            .pipe(self._cls)
        )


@dataclass(slots=True)
class ExprNameNameSpace(ExprNameSpaceBase):
    """Name operations namespace (equivalent to pl.Expr.name).

    Returns:
        Expr
    """

    def _with_alias_mapper(self, mapper: Aliaser) -> Expr:
        meta = self.inner().meta.with_alias_mapper(mapper)
        return self.inner().inner().pipe(Expr, meta)

    def keep(self) -> Expr:
        expr = self.inner()
        meta = expr.meta.unalias()
        return expr.inner().inner().unalias().pipe(SqlExpr).pipe(Expr, meta)

    def map(self, function: Aliaser) -> Expr:
        return self._with_alias_mapper(function)

    def prefix(self, prefix: str) -> Expr:
        return self._with_alias_mapper(lambda name: f"{prefix}{name}")

    def suffix(self, suffix: str) -> Expr:
        return self._with_alias_mapper(lambda name: f"{name}{suffix}")

    def to_lowercase(self) -> Expr:
        return self._with_alias_mapper(str.lower)

    def to_uppercase(self) -> Expr:
        return self._with_alias_mapper(str.upper)

    def replace(self, pattern: str, value: str, *, literal: bool = False) -> Expr:
        match literal:
            case True:
                return self._with_alias_mapper(
                    lambda name: name.replace(pattern, value)
                )
            case False:
                regex = re.compile(pattern)
                return self._with_alias_mapper(lambda name: regex.sub(value, name))


@dataclass(slots=True)
class ExprDateTimeNameSpace(ExprNameSpaceBase):
    """Date and datetime operations namespace (equivalent to pl.Expr.dt).

    Returns:
        Expr
    """

    def millennium(self) -> Expr:
        return self._cls(self.inner().inner().dt.millennium())

    def century(self) -> Expr:
        return self._cls(self.inner().inner().dt.century())

    def year(self) -> Expr:
        return self._cls(self.inner().inner().dt.year())

    def iso_year(self) -> Expr:
        return self._cls(self.inner().inner().dt.isoyear())

    def quarter(self) -> Expr:
        return self._cls(self.inner().inner().dt.quarter())

    def month(self) -> Expr:
        return self._cls(self.inner().inner().dt.month())

    def week(self) -> Expr:
        return self._cls(self.inner().inner().dt.week())

    def weekday(self) -> Expr:
        return self._cls(self.inner().inner().dt.isodow())

    def day(self) -> Expr:
        return self._cls(self.inner().inner().dt.day())

    def ordinal_day(self) -> Expr:
        return self._cls(self.inner().inner().dt.dayofyear())

    def hour(self) -> Expr:
        return self._cls(self.inner().inner().dt.hour())

    def minute(self) -> Expr:
        return self._cls(self.inner().inner().dt.minute())

    def second(self) -> Expr:
        return self._cls(self.inner().inner().dt.second())

    def millisecond(self) -> Expr:
        return self._cls(self.inner().inner().dt.millisecond().mod(Sec.TO_MILLI))

    def microsecond(self) -> Expr:
        return self._cls(self.inner().inner().dt.microsecond().mod(Sec.TO_MICRO))

    def nanosecond(self) -> Expr:
        return self._cls(self.inner().inner().dt.nanosecond().mod(Sec.TO_NANO))

    def month_start(self) -> Expr:
        return self._cls(self.inner().inner().dt.month_start())

    def month_end(self) -> Expr:
        return self._cls(self.inner().inner().dt.month_end())

    def date(self) -> Expr:
        return self.inner().cast(dt.Date())

    def time(self) -> Expr:
        return self.inner().cast(dt.Time())

    def to_string(self, format: IntoExprColumn) -> Expr:  # noqa: A002
        return self._cls(self.inner().inner().str.strftime(format))

    def strftime(self, format: IntoExprColumn) -> Expr:  # noqa: A002
        return self._cls(self.inner().inner().str.strftime(format))

    def epoch(self, time_unit: EpochTimeUnit = "us") -> Expr:
        expr = self.inner().inner().dt
        match time_unit:
            case "d":
                return self._cls(
                    self
                    .inner()
                    .inner()
                    .dt.epoch_us()
                    .truediv(Sec.micro_by_day())
                    .floor()
                )
            case "s":
                return self._cls(expr.epoch_us().truediv(Sec.TO_MICRO).floor())
            case "ms":
                return self._cls(expr.epoch_ms())
            case "us":
                return self._cls(expr.epoch_us())
            case "ns":
                return self._cls(expr.epoch_ns())

    def timestamp(self, time_unit: TimeUnit = "us") -> Expr:
        return self.epoch(time_unit)

    def truncate(self, every: str) -> Expr:
        return self._cls(self.inner().inner().dt.trunc(sql.lit(every)))

    def round(self, every: str) -> Expr:
        return self._cls(self.inner().inner().dt.trunc(sql.lit(every)))

    def offset_by(self, by: IntoExpr) -> Expr:
        return self._cls(self.inner().inner().dt.offset_by(by))
