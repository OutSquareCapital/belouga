"""Column selectors for PQL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, final, overload, override

from pyochain import Iter, Option, Seq, Set

from . import (
    _funcs as fn,  # pyright: ignore[reportPrivateUsage]
    datatypes as dt,
)
from ._core import into_expr
from ._expr import Expr
from ._meta import MultiMeta
from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pyochain.traits import PyoCollection
    from sqlglot import exp

    from .typing import IntoExpr, IntoExprColumn, Schema


type Cols = PyoCollection[str]


@dataclass(slots=True, repr=False)
class Resolver:
    _fn: Callable[[Schema], Cols]

    @override
    def __repr__(self) -> str:
        fn = self._fn.__name__.replace("_", " ").title()
        return f"{self.__class__.__name__}({fn})"

    def __call__(self, schema: Schema) -> Cols:
        return self._fn(schema)

    def into_selector(self) -> Selector:
        return Selector(fn.all().inner, self.into_meta())

    def into_meta(self) -> MultiMeta:
        return MultiMeta(resolver=self)

    @classmethod
    def all_columns(cls) -> Self:
        def _all_columns(schema: Schema) -> Cols:
            return schema.keys()

        return cls(_all_columns)

    @classmethod
    def fixed(cls, names: Cols) -> Self:
        def _fixed(_: Schema) -> Cols:
            return names

        return cls(_fixed)

    @classmethod
    def all_fn(cls, exclude: Option[TryIter[IntoExprColumn]]) -> Self:
        return exclude.map(
            lambda exc: (
                try_iter(exc)
                .map(lambda value: into_expr(value, as_col=True).name)
                .collect(Set)
                .into(cls.exclude)
            )
        ).unwrap_or_else(cls.all_columns)

    @classmethod
    def exclude(cls, excluded: Cols) -> Self:
        def _exclude(schema: Schema) -> Cols:
            return schema.iter().filter(lambda n: n not in excluded).collect()

        return cls(_exclude)

    @classmethod
    def ordered_name(cls, names: Iterable[str]) -> Self:
        def _ordered(schema: Schema) -> Cols:
            return Iter(names).filter(lambda name: name in schema).collect()

        return cls(_ordered)

    @classmethod
    def name(cls, predicate: Callable[[str], bool]) -> Self:
        def _name(schema: Schema) -> Cols:
            return schema.iter().filter(predicate).collect()

        return cls(_name)

    @classmethod
    def dtype(cls, predicate: Callable[[dt.DataType], bool]) -> Self:
        def _dtype(schema: Schema) -> Cols:
            return (
                schema
                .items()
                .iter()
                .filter_star(
                    lambda _name, dtype: predicate(dt.DataType.from_sql(dtype))
                )
                .map_star(lambda name, _dtype: name)
                .collect()
            )

        return cls(_dtype)

    def difference(self, right_fn: Self) -> Self:
        def _difference(schema: Schema) -> Cols:
            right = right_fn(schema)
            return self(schema).iter().filter(lambda n: n not in right).collect()

        return self.__class__(_difference)

    def complement(self) -> Self:
        def _complement(schema: Schema) -> Cols:
            excluded = self(schema)
            return schema.iter().filter(lambda n: n not in excluded).collect()

        return self.__class__(_complement)

    def intersection(self, right: Self) -> Self:
        def _intersection(schema: Schema) -> Cols:
            right_set = right(schema)
            return self(schema).iter().filter(lambda n: n in right_set).collect()

        return self.__class__(_intersection)

    def union(self, right: Self) -> Self:
        def _union(schema: Schema) -> Cols:
            selected = self(schema).iter().chain(right(schema)).collect(Set)
            return schema.iter().filter(lambda n: n in selected).collect()

        return self.__class__(_union)


@final
class Selector(Expr):
    """Column selector based on dtype predicates."""

    meta: MultiMeta  # pyright: ignore[reportIncompatibleVariableOverride]
    __slots__ = ()

    @override
    def _cls(self, value: exp.Expr) -> Expr:
        return Expr(value, self.meta)

    @property
    def _resolver(self) -> Resolver:
        return self.meta.resolver

    @overload
    def union(self, other: Self) -> Self: ...
    @overload
    def union(self, other: IntoExpr) -> Expr: ...
    def union(self, other: IntoExpr) -> Self | Expr:
        match other:
            case Selector():
                return self._resolver.union(other._resolver).into_selector()
            case _:
                return super().__or__(other)

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
                return self._resolver.intersection(other._resolver).into_selector()
            case _:
                return super().__and__(other)

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
                return self._resolver.difference(other._resolver).into_selector()
            case _:
                return super().__sub__(other)

    @overload
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    @override
    def __sub__(self, other: IntoExpr) -> Self | Expr:
        return self.difference(other)

    @override
    def __invert__(self) -> Selector:
        return self.complement()

    def complement(self) -> Selector:
        return self._resolver.complement().into_selector()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:
    """Select columns matching any of the given dtype classes.

    Args:
        *dtypes (type[dt.DataType]): One or more dtype classes to match.

    Returns:
        Selector: A selector for columns matching the specified dtypes.
    """
    return Resolver.dtype(lambda d: isinstance(d, dtypes)).into_selector()


def numeric() -> Selector:
    """Select all numeric columns.

    Returns:
        Selector: A selector for all numeric columns.
    """
    return by_dtype(dt.NumericType)


def string() -> Selector:
    """Select all string columns.

    Returns:
        Selector: A selector for all string columns.
    """
    return by_dtype(dt.StringType)


def boolean() -> Selector:
    """Select all boolean columns.

    Returns:
        Selector: A selector for all boolean columns.
    """
    return by_dtype(dt.Boolean)


def all() -> Selector:
    """Select all columns.

    Returns:
        Selector: A selector for all columns.
    """
    return Resolver.all_columns().into_selector()


def float() -> Selector:
    """Select all float columns.

    Returns:
        Selector: A selector for all float columns.
    """
    return by_dtype(dt.FloatType)


def integer() -> Selector:
    """Select all integer columns.

    Returns:
        Selector: A selector for all integer columns.
    """
    return by_dtype(dt.IntegerType)


def signed_integer() -> Selector:
    """Select all signed integer columns.

    Returns:
        Selector: A selector for all signed integer columns.
    """
    return by_dtype(dt.SignedIntegerType)


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns.

    Returns:
        Selector: A selector for all unsigned integer columns.
    """
    return by_dtype(dt.UnsignedIntegerType)


def temporal() -> Selector:
    """Select all temporal columns.

    Returns:
        Selector: A selector for all temporal columns.
    """
    return by_dtype(dt.TemporalType)


def date() -> Selector:
    """Select all date columns.

    Returns:
        Selector: A selector for all date columns.
    """
    return by_dtype(dt.Date)


def time() -> Selector:
    """Select all time columns.

    Returns:
        Selector: A selector for all time columns.
    """
    return by_dtype(dt.Time, dt.TimeTZ)


def duration() -> Selector:
    """Select all duration columns.

    Returns:
        Selector: A selector for all duration columns.
    """
    return by_dtype(dt.Duration)


def binary() -> Selector:
    """Select all binary columns.

    Returns:
        Selector: A selector for all binary columns.
    """
    return by_dtype(dt.Binary)


def enum() -> Selector:
    """Select all enum columns.

    Returns:
        Selector: A selector for all enum columns.
    """
    return by_dtype(dt.Enum)


def decimal() -> Selector:
    """Select all decimal columns.

    Returns:
        Selector: A selector for all decimal columns.
    """
    return by_dtype(dt.Decimal)


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns.

    Returns:
        Selector: A selector for all nested columns.
    """
    return by_dtype(dt.NestedType)


def struct() -> Selector:
    """Select all struct columns.

    Returns:
        Selector: A selector for all struct columns.
    """
    return by_dtype(dt.Struct)


# ──── name-based selectors ────


def matches(pattern: str) -> Selector:
    """Select columns whose names match the given regex pattern.

    Args:
        pattern (str): A regular expression pattern to match column names against.

    Returns:
            Selector: A selector for columns with names matching the pattern.
    """
    compiled = re.compile(pattern)
    return Resolver.name(lambda name: compiled.search(name) is not None).into_selector()


def by_name(*names: str) -> Selector:
    """Select columns by exact name.

    Args:
        names (str): Column names to select.

    Returns:
        Selector: A selector for columns with the given names.
    """
    return Resolver.ordered_name(names).into_selector()


def starts_with(*prefix: str) -> Selector:
    """Select columns whose names start with any of the given prefixes.

    Args:
        prefix (str): Prefixes to match column names against.

    Returns:
        Selector: A selector for columns with names starting with any of the given prefixes.
    """
    return Resolver.name(lambda name: name.startswith(prefix)).into_selector()


def ends_with(*suffix: str) -> Selector:
    """Select columns whose names end with any of the given suffixes.

    Args:
        suffix (str): Suffixes to match column names against.

    Returns:
        Selector: A selector for columns with names ending with any of the given suffixes.
    """
    return Resolver.name(lambda name: name.endswith(suffix)).into_selector()


def contains(*substring: str) -> Selector:
    """Select columns whose names contain any of the given substrings.

    Args:
        substring (str): Substrings to match column names against.

    Returns:
        Selector: A selector for columns with names containing any of the given substrings.
    """
    subs = Seq(substring)
    return Resolver.name(lambda name: subs.any(lambda s: s in name)).into_selector()


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
