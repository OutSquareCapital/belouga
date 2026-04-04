from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc
from sqlglot import exp

from . import sql
from ._expr import Expr
from ._meta import ExprKind, Marker, MultiMeta, Resolver, SingleMeta
from .sql import SqlExpr
from .sql._conversions import args_into_glot
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
    return Expr(sql.lit(1), SingleMeta(root_name=Marker.LEN)).count()


def _agg_expr(
    agg: Callable[[SqlExpr], SqlExpr],
    cols: TryIter[str],
    more_cols: Iterable[str],
) -> Expr:

    def _columns_expr(inner_cols: pc.Seq[str]) -> SqlExpr:
        expr = exp.Columns(this=inner_cols.into(args_into_glot))
        return SqlExpr(expr)

    names = try_iter(cols).chain(more_cols).collect().then_some()
    meta = MultiMeta(
        kind=ExprKind.SCALAR, resolver=Resolver.agg_expr(names), preserve_native=True
    )
    inner_expr = (
        names
        .map(_columns_expr)
        .unwrap_or_else(lambda: SqlExpr(exp.Columns(this=exp.Star())))
        .pipe(agg)
    )
    return Expr(inner_expr, meta)


def sum(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(SqlExpr.sum, cols, more_cols)


def mean(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(SqlExpr.mean, cols, more_cols)


def median(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(SqlExpr.median, cols, more_cols)


def min(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(SqlExpr.min, cols, more_cols)


def max(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(SqlExpr.max, cols, more_cols)


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """Create a coalesce expression.

    Returns:
        Expr: A new expression that evaluates to the first non-null value among the given expressions.
    """
    expr_name = (
        try_iter(exprs).next().map(sql.into_expr, as_col=True).unwrap().get_name()
    )
    return Expr(sql.coalesce(exprs, *more_exprs), SingleMeta(root_name=expr_name))


def all(exclude: TryIter[IntoExprColumn] = None) -> Expr:
    """Create an expression representing all columns (equivalent to pl.all()).

    Returns:
        Expr: A new expression that evaluates to all columns.
    """
    meta = MultiMeta(resolver=Resolver.all_fn(pc.Option(exclude)), preserve_native=True)
    return Expr(sql.all(exclude), meta)


def _horizontal_fn(
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    fn: Callable[..., SqlExpr],
) -> Expr:
    meta = (
        try_iter(exprs)
        .next()
        .map(lambda v: sql.into_expr(v, as_col=True).get_name())
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
