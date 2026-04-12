"""Column selectors for PQL."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self, final, overload, override

import pyochain as pc

from . import sql
from ._expr import Expr
from ._meta import MultiMeta
from .sql import SqlExpr
from .sql.utils import TryIter, try_iter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pyochain.traits import PyoCollection

    from . import _datatypes as dt  # pyright: ignore[reportPrivateUsage]
    from .sql.typing import IntoExpr, IntoExprColumn


type Cols = PyoCollection[str]


@dataclass(slots=True, repr=False)
class Resolver:
    _fn: Callable[[Cols], Cols]

    @override
    def __repr__(self) -> str:
        fn = self._fn.__name__.replace("_", " ").title()
        return f"{self.__class__.__name__}({fn})"

    def __call__(self, cols: Cols) -> Cols:
        return self._fn(cols)

    def into_selector(self) -> Selector:
        return Selector(sql.all(), MultiMeta(resolver=self))

    @classmethod
    def all_columns(cls) -> Self:
        def _all_columns(cols: Cols) -> Cols:
            return cols

        return cls(_all_columns)

    @classmethod
    def fixed(cls, names: Cols) -> Self:
        def _fixed(_: Cols) -> Cols:
            return names

        return cls(_fixed)

    @classmethod
    def all_fn(cls, exclude: pc.Option[TryIter[IntoExprColumn]]) -> Self:
        return exclude.map(
            lambda exc: (
                try_iter(exc)
                .map(lambda value: SqlExpr.new(value, as_col=True).inner().name)
                .collect(pc.Set)
                .into(cls.exclude)
            )
        ).unwrap_or_else(cls.all_columns)

    @classmethod
    def exclude(cls, excluded: Cols) -> Self:
        def _exclude(cols: Cols) -> Cols:
            return cols.iter().filter(lambda n: n not in excluded).collect()

        return cls(_exclude)

    @classmethod
    def agg_expr(cls, cols: pc.Option[pc.Seq[str]]) -> Self:
        return cols.map(cls.fixed).unwrap_or_else(cls.all_columns)

    @classmethod
    def ordered_name(cls, names: Iterable[str]) -> Self:
        def _ordered(cols: Cols) -> Cols:
            return pc.Iter(names).filter(lambda name: name in cols).collect()

        return cls(_ordered)

    @classmethod
    def name(cls, predicate: Callable[[str], bool]) -> Self:
        def _name(cols: Cols) -> Cols:
            return cols.iter().filter(predicate).collect()

        return cls(_name)

    def difference(self, right_fn: Self) -> Self:
        def _difference(cols: Cols) -> Cols:
            right = right_fn(cols)
            return self(cols).iter().filter(lambda n: n not in right).collect()

        return self.__class__(_difference)

    def complement(self) -> Self:
        def _complement(cols: Cols) -> Cols:
            excluded = self(cols)
            return cols.iter().filter(lambda n: n not in excluded).collect()

        return self.__class__(_complement)

    def intersection(self, right: Self) -> Self:
        def _intersection(cols: Cols) -> Cols:
            right_set = right(cols)
            return self(cols).iter().filter(lambda n: n in right_set).collect()

        return self.__class__(_intersection)

    def union(self, right: Self) -> Self:
        def _union(cols: Cols) -> Cols:
            selected = self(cols).iter().chain(right(cols)).collect(pc.Set)
            return cols.iter().filter(lambda n: n in selected).collect()

        return self.__class__(_union)


@final
class Selector(Expr):
    """Column selector based on dtype predicates."""

    meta: MultiMeta  # pyright: ignore[reportIncompatibleVariableOverride]
    __slots__ = ()

    @override
    def _cls(self, value: SqlExpr) -> Expr:
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

    def complement(self) -> Selector:
        return self._resolver.complement().into_selector()

    @override
    def __invert__(self) -> Selector:
        return self.complement()


def by_dtype(*dtypes: type[dt.DataType]) -> Selector:  # pyright: ignore[reportUnusedParameter]
    """Select columns matching any of the given dtype classes.

    Args:
        *dtypes (type[dt.DataType]): One or more dtype classes to match.

    Returns:
        Selector: A selector for columns matching the specified dtypes.
    """
    raise NotImplementedError


def numeric() -> Selector:
    """Select all numeric columns.

    Returns:
        Selector: A selector for all numeric columns.
    """
    raise NotImplementedError


def string() -> Selector:
    """Select all string columns.

    Returns:
        Selector: A selector for all string columns.
    """
    raise NotImplementedError


def boolean() -> Selector:
    """Select all boolean columns.

    Returns:
        Selector: A selector for all boolean columns.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def integer() -> Selector:
    """Select all integer columns.

    Returns:
        Selector: A selector for all integer columns.
    """
    raise NotImplementedError


def signed_integer() -> Selector:
    """Select all signed integer columns.

    Returns:
        Selector: A selector for all signed integer columns.
    """
    raise NotImplementedError


def unsigned_integer() -> Selector:
    """Select all unsigned integer columns.

    Returns:
        Selector: A selector for all unsigned integer columns.
    """
    raise NotImplementedError


def temporal() -> Selector:
    """Select all temporal columns.

    Returns:
        Selector: A selector for all temporal columns.
    """
    raise NotImplementedError


def date() -> Selector:
    """Select all date columns.

    Returns:
        Selector: A selector for all date columns.
    """
    raise NotImplementedError


def time() -> Selector:
    """Select all time columns.

    Returns:
        Selector: A selector for all time columns.
    """
    raise NotImplementedError


def duration() -> Selector:
    """Select all duration columns.

    Returns:
        Selector: A selector for all duration columns.
    """
    raise NotImplementedError


def binary() -> Selector:
    """Select all binary columns.

    Returns:
        Selector: A selector for all binary columns.
    """
    raise NotImplementedError


def enum() -> Selector:
    """Select all enum columns.

    Returns:
        Selector: A selector for all enum columns.
    """
    raise NotImplementedError


def decimal() -> Selector:
    """Select all decimal columns.

    Returns:
        Selector: A selector for all decimal columns.
    """
    raise NotImplementedError


def nested() -> Selector:
    """Select all nested (list, array, struct, map) columns.

    Returns:
        Selector: A selector for all nested columns.
    """
    raise NotImplementedError


def struct() -> Selector:
    """Select all struct columns.

    Returns:
        Selector: A selector for all struct columns.
    """
    raise NotImplementedError


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
    subs = pc.Seq(substring)
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
