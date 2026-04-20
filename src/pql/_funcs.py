from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc

from . import sql
from ._expr import Expr
from .selectors import Resolver
from .sql import SqlExpr
from .sql.typing import IntoExpr, IntoExprColumn, PythonLiteral
from .sql.utils import TryIter, try_iter


@final
class Col:
    __slots__ = ()

    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name))

    def __getattr__(self, name: str) -> Expr:
        return self(name)


col: Col = Col()


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression.

    Returns:
        Expr: A new expression that evaluates to the literal value.
    """
    return Expr(sql.lit(value))


def len() -> Expr:
    """Return the number of rows.

    Returns:
        Expr: A new expression that evaluates to the number of rows.
    """
    return Expr(sql.len())


def _agg_expr(
    agg: Callable[[TryIter[str], *tuple[str, ...]], SqlExpr],
    cols: TryIter[str],
    more_cols: Iterable[str],
) -> Expr:
    meta = (
        try_iter(cols)
        .chain(more_cols)
        .collect()
        .then(Resolver.fixed)
        .unwrap_or_else(Resolver.all_columns)
        .into_meta()
    )
    return Expr(agg(cols, *more_cols), meta)


def sum(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.sum, cols, more_cols)


def mean(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.mean, cols, more_cols)


def median(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.median, cols, more_cols)


def min(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.min, cols, more_cols)


def max(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(sql.max, cols, more_cols)


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """Create a coalesce expression.

    Returns:
        Expr: A new expression that evaluates to the first non-null value among the given expressions.
    """
    return sql.coalesce(exprs, *more_exprs).pipe(Expr)


def all(exclude: TryIter[IntoExprColumn] = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all()).

    Returns:
        Expr: A new expression that evaluates to all columns.
    """
    return Expr(sql.all(exclude), Resolver.all_fn(pc.Option(exclude)).into_meta())


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.sum_horizontal(exprs, *more_exprs))


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.min_horizontal(exprs, *more_exprs))


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.max_horizontal(exprs, *more_exprs))


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.mean_horizontal(exprs, *more_exprs))


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.all_horizontal(exprs, *more_exprs))


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return Expr(sql.any_horizontal(exprs, *more_exprs))


_ELEMENT = Expr(sql.element())


def element() -> Expr:
    """Alias for an element being evaluated in a list context.

    Returns:
        Expr: A new expression that evaluates to the element being evaluated in a list or array context.
    """
    return _ELEMENT
