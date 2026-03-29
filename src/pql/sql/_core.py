from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Self, overload, override

import duckdb
import pyochain as pc
from sqlglot import exp
from sqlglot.parsers.duckdb import DuckDBParser

if TYPE_CHECKING:
    from .typing import (
        IntoDuckExpr,
        IntoDuckExprCol,
        IntoExpr,
        IntoExprColumn,
    )


@dataclass(slots=True, repr=False)
class CoreHandler[T]:
    """A wrapper for an inner value.

    Is used as a base class for Expressions, Relation, LazyFrame, and namespaces, since they all share the same pattern of wrapping an inner value and forwarding method calls to it.
    """

    _inner: T

    @override
    def __repr__(self) -> str:
        return self.inner().__repr__()

    @override
    def __str__(self) -> str:
        return self.inner().__str__()

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

    def _new(self, value: T) -> Self:
        """Create a new instance of *Self* with the given value."""
        return self.__class__(value)

    def inner(self) -> T:
        """Unwrap the underlying value."""
        return self._inner


@dataclass(slots=True, repr=False)
class DuckHandler(CoreHandler[exp.Expr]):
    """A wrapper for DuckDB expressions."""

    def into_duckdb(self) -> duckdb.Expression:
        """Convert the inner expression to a DuckDB expression."""
        return _glot_into_duckdb(self.inner())


def _glot_into_duckdb(expr: exp.Expr) -> duckdb.Expression:  # noqa: C901, PLR0912
    def _alias_name(value: exp.Expr | str | None) -> pc.Result[str, ValueError]:
        match value:
            case exp.Identifier() as ident:
                return pc.Ok(ident.name)
            case exp.Expr() as expr:
                return pc.Ok(expr.sql(dialect="duckdb"))
            case str() as name:
                return pc.Ok(name)
            case _:
                msg = "Alias expression requires a non-empty alias name"
                return pc.Err(ValueError(msg))

    match expr:
        case exp.Alias() as alias_expr:
            return (
                _alias_name(alias_expr.args.get("alias"))
                .map(lambda name: _glot_into_duckdb(alias_expr.this).alias(name))  # pyright: ignore[reportAny]
                .unwrap()
            )
        case exp.Column() as col_expr:
            parts = pc.Iter(col_expr.parts).map(lambda part: part.name)
            return duckdb.ColumnExpression(*parts)
        case exp.Anonymous() | exp.Greatest() | exp.Least() as func_expr:
            match func_expr:
                case exp.Anonymous() as anon_expr:
                    name = anon_expr.name
                    exprs = anon_expr.expressions
                case _ as func_expr:
                    name = func_expr.sql_name()
                    exprs = func_expr.iter_expressions()
            args = pc.Iter(exprs).map(_glot_into_duckdb)
            return duckdb.FunctionExpression(name, *args)

        case exp.Lambda() as lambda_expr:
            match lambda_expr.expressions[0]:
                case exp.Identifier() as identifier:
                    param = identifier.name
                case _ as item:  # pyright: ignore[reportAny]
                    param = str(item)  # pyright: ignore[reportAny]

            return duckdb.LambdaExpression(param, _glot_into_duckdb(lambda_expr.this))  # pyright: ignore[reportAny]
        case exp.Ordered() as ordered_expr:
            ordered = _glot_into_duckdb(ordered_expr.this)  # pyright: ignore[reportAny]
            match ordered_expr.args.get("desc", False):  # pyright: ignore[reportMatchNotExhaustive]
                case True:
                    ordered = ordered.desc()
                case False:
                    ordered = ordered.asc()
            match ordered_expr.args.get("nulls_first", False):  # pyright: ignore[reportMatchNotExhaustive]
                case True:
                    ordered = ordered.nulls_first()
                case False:
                    ordered = ordered.nulls_last()
            return ordered
        case _:
            return duckdb.SQLExpression(expr.sql(dialect="duckdb"))


@overload
def into_duckdb(value: IntoExprColumn) -> IntoDuckExprCol: ...
@overload
def into_duckdb(value: IntoExpr) -> IntoDuckExpr: ...
def into_duckdb(value: IntoExpr | IntoExprColumn) -> IntoDuckExpr | IntoDuckExprCol:
    from .._expr import Expr

    match value:
        case DuckHandler():
            return value.into_duckdb()
        case Expr():
            return value.inner().into_duckdb()
        case exp.Expr():
            return DuckHandler(value).into_duckdb()
        case _:
            return value


@dataclass(slots=True)
class NameSpaceHandler[T: DuckHandler]:
    """A wrapper for expression namespaces that return the parent type."""

    _parent: T

    def _new(self, expr: exp.Expr) -> T:
        return self._parent.__class__(expr)

    def inner(self) -> T:
        """Unwrap the underlying expression."""
        return self._parent


def into_glot(value: IntoExpr) -> exp.Expr:
    """Convert an IntoExpr value into a sqlglot expression node."""
    from .._expr import Expr

    match value:
        case DuckHandler():
            return value.inner()
        case Expr():
            return value.inner().inner()
        case exp.Expr():
            return value
        case str():
            return exp.column(value)
        case _:
            return exp.convert(value)


def args_into_glot(args: Iterable[IntoExpr]) -> list[exp.Expr]:
    return pc.Iter(args).filter_map(pc.Option).map(into_glot).collect(list)


def anon(name: str, *args: IntoExpr) -> exp.Expr:
    """Create a SQL anonymous function expression."""
    return exp.Anonymous(this=name, expressions=args_into_glot(args))


def func(name: str, *args: IntoExpr) -> exp.Expr:
    return DuckDBParser.FUNCTIONS[name](args_into_glot(args))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
