"""Column selectors for PQL."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Self, final, overload, override

import pyochain as pc

from . import _datatypes as dt, sql  # pyright: ignore[reportPrivateUsage]
from ._expr import Expr
from ._meta import MultiMeta, Resolver, ResolverFn

if TYPE_CHECKING:
    from .sql.typing import IntoExpr


@final
class Selector(Expr):
    """Column selector based on dtype predicates."""

    meta: MultiMeta  # pyright: ignore[reportIncompatibleVariableOverride]
    __slots__ = ()

    @property
    def _resolver(self) -> ResolverFn:
        return self.meta.resolver

    @classmethod
    def __from_resolver__(cls, resolver: ResolverFn) -> Self:  # noqa: PLW3201
        return cls(sql.all(), MultiMeta(resolver=resolver))

    @overload
    def union(self, other: Self) -> Self: ...
    @overload
    def union(self, other: IntoExpr) -> Expr: ...
    def union(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self.__from_resolver__(
                    Resolver.union(self._resolver, other._resolver)
                )
            case _:
                return Expr.__or__(self, other)

    @overload
    def __or__(self, other: Self) -> Self: ...
    @overload
    def __or__(self, other: IntoExpr) -> Expr: ...
    @override
    def __or__(self, other: IntoExpr) -> Self | Expr:
        return self.union(other)

    @overload
    def intersection(self, other: Self) -> Self: ...
    @overload
    def intersection(self, other: IntoExpr) -> Expr: ...
    def intersection(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self.__from_resolver__(
                    Resolver.intersection(self._resolver, other._resolver)
                )
            case _:
                return Expr.__and__(self, other)

    @overload
    def __and__(self, other: Self) -> Self: ...
    @overload
    def __and__(self, other: IntoExpr) -> Expr: ...
    @override
    def __and__(self, other: IntoExpr) -> Self | Expr:
        return self.intersection(other)

    @overload
    def difference(self, other: Self) -> Self: ...
    @overload
    def difference(self, other: IntoExpr) -> Expr: ...
    def difference(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self.__from_resolver__(
                    Resolver.difference(self._resolver, other._resolver)
                )
            case _:
                return Expr.__sub__(self, other)

    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    @override
    def __sub__(self, other: IntoExpr) -> Self | Expr:
        return self.difference(other)

    def complement(self) -> Selector:
        return self.__from_resolver__(Resolver.complement(self._resolver))

    @override
    def __invert__(self) -> Selector:
        return self.complement()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes.

    Args:
        *dtypes (type[dt.DataType]): One or more dtype classes to match.

    Returns:
        Selector: A selector for columns matching the specified dtypes.
    """
    return Selector.__from_resolver__(Resolver.dtype(*dtypes))


def numeric() -> Selector:
    """Select all numeric columns.

    Returns:
        Selector: A selector for all numeric columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.NumericType))


def string() -> Selector:
    """Select all string columns.

    Returns:
        Selector: A selector for all string columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.StringType))


def boolean() -> Selector:
    """Select all boolean columns.

    Returns:
        Selector: A selector for all boolean columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Boolean))


def all() -> Selector:
    """Select all columns.

    Returns:
        Selector: A selector for all columns.
    """
    return Selector.__from_resolver__(Resolver.all_columns())


def float() -> Selector:
    """Select all float columns.

    Returns:
        Selector: A selector for all float columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.FloatType))


def integer() -> Selector:
    """Select all integer columns.

    Returns:
        Selector: A selector for all integer columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.IntegerType))


def signed_integer() -> Selector:
    """Select all signed integer columns.

    Returns:
        Selector: A selector for all signed integer columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.SignedIntegerType))


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns.

    Returns:
        Selector: A selector for all unsigned integer columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.UnsignedIntegerType))


def temporal() -> Selector:
    """Select all temporal columns.

    Returns:
        Selector: A selector for all temporal columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.TemporalType))


def date() -> Selector:
    """Select all date columns.

    Returns:
        Selector: A selector for all date columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Date))


def time() -> Selector:
    """Select all time columns.

    Returns:
        Selector: A selector for all time columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Time))


def duration() -> Selector:
    """Select all duration columns.

    Returns:
        Selector: A selector for all duration columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Duration))


def binary() -> Selector:
    """Select all binary columns.

    Returns:
        Selector: A selector for all binary columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Binary))


def enum() -> Selector:
    """Select all enum columns.

    Returns:
        Selector: A selector for all enum columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Enum))


def decimal() -> Selector:
    """Select all decimal columns.

    Returns:
        Selector: A selector for all decimal columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Decimal))


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns.

    Returns:
        Selector: A selector for all nested columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.NestedType))


def struct() -> Selector:
    """Select all struct columns.

    Returns:
        Selector: A selector for all struct columns.
    """
    return Selector.__from_resolver__(Resolver.dtype(dt.Struct))


# ──── name-based selectors ────


def matches(pattern: str) -> Selector:
    """Select columns whose names match the given regex pattern.

    Args:
        pattern (str): A regular expression pattern to match column names against.

    Returns:
            Selector: A selector for columns with names matching the pattern.
    """
    compiled = re.compile(pattern)
    return Selector.__from_resolver__(
        Resolver.name(lambda name: compiled.search(name) is not None)
    )


def by_name(*names: str) -> Selector:
    """Select columns by exact name.

    Args:
        names (str): Column names to select.

    Returns:
        Selector: A selector for columns with the given names.
    """
    return Selector.__from_resolver__(Resolver.ordered_name(pc.Seq(names)))


def starts_with(*prefix: str) -> Selector:
    """Select columns whose names start with any of the given prefixes.

    Args:
        prefix (str): Prefixes to match column names against.

    Returns:
        Selector: A selector for columns with names starting with any of the given prefixes.
    """
    return Selector.__from_resolver__(
        Resolver.name(lambda name: name.startswith(prefix))
    )


def ends_with(*suffix: str) -> Selector:
    """Select columns whose names end with any of the given suffixes.

    Args:
        suffix (str): Suffixes to match column names against.

    Returns:
        Selector: A selector for columns with names ending with any of the given suffixes.
    """
    return Selector.__from_resolver__(Resolver.name(lambda name: name.endswith(suffix)))


def contains(*substring: str) -> Selector:
    """Select columns whose names contain any of the given substrings.

    Args:
        substring (str): Substrings to match column names against.

    Returns:
        Selector: A selector for columns with names containing any of the given substrings.
    """
    subs = pc.Seq(substring)
    return Selector.__from_resolver__(
        Resolver.name(lambda name: subs.any(lambda s: s in name))
    )


__all__ = [
    "all",
    "binary",
    "boolean",
    "by_dtype",
    "by_name",
    "contains",
    "date",
    "decimal",
    "duration",
    "ends_with",
    "enum",
    "float",
    "integer",
    "matches",
    "nested",
    "numeric",
    "signed_integer",
    "starts_with",
    "string",
    "struct",
    "temporal",
    "time",
    "unsigned_integer",
]
