from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Self, override

from sqlglot import exp

from ._conversions import into_expr_list
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
class DuckHandler(CoreHandler[exp.Expr]):
    """A wrapper for DuckDB expressions."""


@dataclass(slots=True)
class NameSpaceHandler[T: DuckHandler]:
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
