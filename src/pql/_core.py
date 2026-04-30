from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Self, override

from pyochain import Iter, Option
from sqlglot import exp

from ._sqlglot_patch import DUCKDB_FUNCTIONS

if TYPE_CHECKING:
    from .typing import IntoExpr


@dataclass(slots=True, repr=False)
class CoreHandler[T]:
    """A wrapper for an inner value.

    Is used as a base class for Expressions, Relation, LazyFrame, and namespaces, since they all share the same pattern of wrapping an inner value and forwarding method calls to it.
    """

    _inner: T

    @override
    def __repr__(self) -> str:
        return self.inner.__repr__()

    @override
    def __str__(self) -> str:
        return self.inner.__str__()

    def pipe[**P, R](
        self,
        function: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Apply a *function* to *Self* with *args* and *kwargs*.

        Allow to do `x.pipe(func, ...)` instead of `func(x, ...)`.

        This keep a fluent style for UDF, and is shared across `Expr` and `LazyFrame` objects.

        This is similar to **polars** `.pipe` method.

        Args:
            function (Callable[Concatenate[Self, P], R]): The *function* to apply.
            *args (P.args): Positional arguments to pass to *function*.
            **kwargs (P.kwargs): Keyword arguments to pass to *function*.

        Returns:
            R: The result of applying the *function*.
        """
        return function(self, *args, **kwargs)

    def _cls(self, value: T) -> Self:
        """Create a new instance of *Self* with the given value.

        Args:
            value (T): The value to wrap.

        Returns:
            Self: A new instance of *Self* with the given value.
        """
        return self.__class__(value)

    @property
    def inner(self) -> T:
        """Unwrap the underlying value.

        Returns:
            T: The underlying value.
        """
        return self._inner


@dataclass(slots=True, repr=False)
class ExprHandler(CoreHandler[exp.Expr]):
    """A wrapper for `sqlglot` expressions."""


@dataclass(slots=True)
class NameSpaceHandler[T: ExprHandler]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _cls(self, expr: exp.Expr) -> T:
        return self._parent.__class__(expr)

    @property
    def inner(self) -> T:
        """Unwrap the underlying expression.

        Returns:
            T: The parent type of the namespace.
        """
        return self._parent


def anon(name: str, *args: IntoExpr) -> exp.Expr:
    """Create a SQL anonymous function expression.

    Returns:
        exp.Expr: A new expression representing the anonymous function.
    """
    return exp.Anonymous(this=name, expressions=into_expr_list(args))


def anon_agg(name: str, *args: IntoExpr) -> exp.Expr:
    """Create a SQL anonymous aggregate function expression.

    Returns:
        exp.Expr: A new aggregate expression representing the anonymous function.
    """
    return exp.AnonymousAggFunc(this=name, expressions=into_expr_list(args))


def func(name: str, *args: IntoExpr) -> exp.Expr:
    return DUCKDB_FUNCTIONS[name](into_expr_list(args))


def into_expr_list(args: Iterable[IntoExpr], *, as_col: bool = False) -> list[exp.Expr]:
    """Convert an `Iterable` of `IntoExpr` values into a list of sqlglot `Expr` nodes.

    Args:
        args (Iterable[IntoExpr]): The values to convert.
        as_col (bool): Whether to treat string values as column names. Defaults to `False`.

    Returns:
        list[exp.Expr]: A list of sqlglot expressions.
    """
    return (
        Iter(args)
        .filter_map(Option)
        .map(lambda x: into_expr(x, as_col=as_col))
        .collect(list)
    )


def into_expr(value: IntoExpr, *, as_col: bool = True) -> exp.Expr:
    """Convert an `IntoExpr` value into a sqlglot `Expr` node.

    Args:
        value (IntoExpr): The value to convert.
        as_col (bool): Whether to treat string values as column names. Defaults to `True`.

    Returns:
        exp.Expr: The resulting sqlglot expression.
    """
    match value:
        case ExprHandler():
            return value.inner
        case str() if as_col:
            return exp.column(value)
        case _:
            return exp.convert(value)
