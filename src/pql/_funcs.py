from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc

from . import sql
from ._expr import Expr
from ._meta import Marker, SingleMeta
from .selectors import Resolver
from .sql import SqlExpr
from .sql.typing import IntoExpr, IntoExprColumn, PythonLiteral
from .sql.utils import TryIter, try_iter


@final
class Col:
    __slots__ = ()

    def __call__(self, name: str) -> Expr:
        return Expr(sql.col(name), SingleMeta(root_name=name))

    def __getattr__(self, name: str) -> Expr:
        return self(name)


col: Col = Col()


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression.

    Returns:
        Expr: A new expression that evaluates to the literal value.
    """
    return Expr(sql.lit(value), SingleMeta(root_name=Marker.LIT))


def len() -> Expr:
    """Return the number of rows.

    Returns:
        Expr: A new expression that evaluates to the number of rows.
    """
    return Expr(sql.len(), SingleMeta(root_name=Marker.LEN))


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
    all_exprs = try_iter(exprs).chain(more_exprs)
    expr = all_exprs.next().map(SqlExpr.new, as_col=True).unwrap()
    return Expr(expr.coalesce(all_exprs), SingleMeta(root_name=expr.inner.output_name))


def all(exclude: TryIter[IntoExprColumn] = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all()).

    Returns:
        Expr: A new expression that evaluates to all columns.
    """
    return Expr(sql.all(exclude), Resolver.all_fn(pc.Option(exclude)).into_meta())


def _horizontal_fn(
    exprs: TryIter[IntoExpr], more_exprs: Iterable[IntoExpr], fn: Callable[..., SqlExpr]
) -> Expr:
    meta = (
        try_iter(exprs)
        .next()
        .map(lambda v: SqlExpr.new(v, as_col=True).inner.output_name)
        .map(lambda n: SingleMeta(root_name=n))
        .unwrap()
    )
    return Expr(fn(exprs, *more_exprs), meta)


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, sql.sum_horizontal)


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, sql.min_horizontal)


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, sql.max_horizontal)


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, sql.mean_horizontal)


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, sql.all_horizontal)


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, sql.any_horizontal)


_ELEMENT = Expr(sql.element(), SingleMeta(root_name=Marker.ELEMENT))


def element() -> Expr:
    """Alias for an element being evaluated in a list context.

    Returns:
        Expr: A new expression that evaluates to the element being evaluated in a list or array context.
    """
    return _ELEMENT
