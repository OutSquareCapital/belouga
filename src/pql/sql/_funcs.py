from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc
from sqlglot import exp

from ._conversions import args_into_glot, pql_into_glot
from ._expr import SqlExpr
from .typing import IntoExpr, IntoExprColumn, PythonLiteral
from .utils import TryIter, try_iter


def reduce(
    exprs: Iterable[IntoExpr], function: Callable[[SqlExpr, IntoExpr], SqlExpr]
) -> SqlExpr:
    """Reduces an `Iterable` of `IntoExpr` into a single `SqlExpr`.

    Done by applying a binary *fn* (defaulting to logical `AND`) to each item, after converting them with `into_expr`.

    Args:
        exprs (Iterable[IntoExpr]): The expressions to reduce.
        function (Callable[[SqlExpr, IntoExpr], SqlExpr]): The binary function to apply for reduction.

    Returns:
        SqlExpr: The result of reducing the expressions with the given function.
    """
    return (
        pc
        .Iter(exprs)
        .map(lambda value: SqlExpr.new(value, as_col=True))
        .reduce(function)
    )


def row_number() -> SqlExpr:
    """Create a ROW_NUMBER() expression.

    Returns:
        SqlExpr: An expression representing the ROW_NUMBER() function.
    """
    return SqlExpr(exp.RowNumber())


def unnest(
    col: IntoExprColumn, max_depth: int | None = None, *, recursive: bool = False
) -> SqlExpr:
    """The unnest special function is used to unnest lists or structs by one level.

    The function can be used as a regular scalar function, but only in the SELECT clause.

    Invoking unnest with the recursive parameter will unnest lists and structs of multiple levels.

    The depth of unnesting can be limited using the max_depth parameter (which assumes recursive unnesting by default).

    Using `unnest` on a list emits one row per list entry.

    Regular scalar expressions in the same `SELECT` clause are repeated for every emitted row.

    When multiple lists are unnested in the same `SELECT` clause, the lists are unnested side-by-side.

    If one list is longer than the other, the shorter list is padded with `NULL` values.

    Empty and `NULL` lists both unnest to zero rows.

    Note:
        We use `exp.Explode` altough `DuckDB` document `UNNEST`. `Exp.Unnest()` does not seem to be equivalent when parsed.

    Args:
        col (SqlExpr): The column to unnest.
        max_depth (int | None): Maximum depth of recursive unnesting.
        recursive (bool): Whether to recursively unnest lists and structs (default: `False`).  Note that lists *within* structs are not unnested.

    Returns:
        SqlExpr: An expression representing the unnesting operation.
    """
    expr = exp.Explode(
        this=pql_into_glot(col), max_depth=max_depth, recursive=recursive
    )
    return SqlExpr(expr)


@final
class Col:
    __slots__ = ()

    def __call__(self, name: str, table: str | None = None) -> SqlExpr:
        return SqlExpr(exp.column(name, table=table))

    def __getattr__(self, name: str) -> SqlExpr:
        return self(name)


col = Col()


ELEM_NAME = "element"

ELEMENT = col(ELEM_NAME)
_ELEM_ID = exp.to_identifier(ELEM_NAME)


def element() -> SqlExpr:
    return ELEMENT


def fn_once(rhs: IntoExpr) -> SqlExpr:
    return SqlExpr(exp.Lambda(this=pql_into_glot(rhs), expressions=[_ELEM_ID]))


def all(exclude: TryIter[IntoExprColumn] = None) -> SqlExpr:
    return (
        pc
        .Option(exclude)
        .map(lambda x: try_iter(x).map(pql_into_glot).collect())
        .map(lambda exc: SqlExpr(exp.Star(except_=exc)))
        .unwrap_or_else(lambda: SqlExpr(exp.Star()))
    )


def lit(value: PythonLiteral) -> SqlExpr:
    """Create a literal expression.

    Args:
        value (PythonLiteral): The literal value to create an expression for.

    Returns:
        SqlExpr: An expression representing the literal value.
    """
    return SqlExpr(exp.convert(value))


def len() -> SqlExpr:
    """Return the number of rows.

    Returns:
        SqlExpr
    """
    from .._meta import Marker

    return lit(1).count().alias(Marker.LEN)


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    """Create a COALESCE expression.

    Args:
        exprs (TryIter[IntoExpr]): The expressions to coalesce.
        *more_exprs (IntoExpr): Additional expressions to coalesce.

    Returns:
        SqlExpr: An expression representing the COALESCE operation.
    """
    all_exprs = try_iter(exprs).chain(more_exprs)
    expr = all_exprs.next().map(SqlExpr.new, as_col=True).unwrap()
    return expr.coalesce(all_exprs).alias(expr.inner.output_name)


_HORIZONTAL_ERR = "At least one expression is required."


def _into_col(value: IntoExpr) -> SqlExpr:
    return SqlExpr.new(value, as_col=True)


def _horizontal_fn(
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    fn: Callable[[SqlExpr, *tuple[IntoExpr]], SqlExpr],
) -> SqlExpr:
    all_exprs = try_iter(exprs).chain(more_exprs).map(_into_col)
    return (
        all_exprs
        .next()
        .map(lambda first: first.pipe(fn, *all_exprs).alias(first.inner.output_name))
        .expect(_HORIZONTAL_ERR)
    )


def _horizontal_reduce(
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    fn: Callable[[SqlExpr, IntoExpr], SqlExpr],
) -> SqlExpr:
    all_exprs = try_iter(exprs).chain(more_exprs).map(_into_col).collect()
    return all_exprs.iter().reduce(fn).alias(all_exprs.first().inner.output_name)


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_fn(exprs, more_exprs, SqlExpr.least)


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_fn(exprs, more_exprs, SqlExpr.greatest)


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_reduce(
        exprs, more_exprs, lambda lhs, rhs: lhs.add(_into_col(rhs).coalesce(0))
    )


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_reduce(exprs, more_exprs, SqlExpr.and_)


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_reduce(exprs, more_exprs, SqlExpr.or_)


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    dtype = exp.DType.BIGINT.into_expr()
    return (
        try_iter(exprs)
        .chain(more_exprs)
        .map(_into_col)
        .collect()
        .then(
            lambda vals: (
                vals
                .iter()
                .map(lambda value: value.coalesce(0))
                .reduce(SqlExpr.add)
                .truediv(
                    vals
                    .iter()
                    .map(lambda value: value.is_not_null().cast(dtype))
                    .reduce(SqlExpr.add)
                )
                .alias(vals.first().inner.output_name)
            )
        )
        .expect(_HORIZONTAL_ERR)
    )


def sum(cols: TryIter[str], *more_cols: str) -> SqlExpr:
    return _agg_expr(SqlExpr.sum, cols, more_cols)


def mean(cols: TryIter[str], *more_cols: str) -> SqlExpr:
    return _agg_expr(SqlExpr.mean, cols, more_cols)


def median(cols: TryIter[str], *more_cols: str) -> SqlExpr:
    return _agg_expr(SqlExpr.median, cols, more_cols)


def min(cols: TryIter[str], *more_cols: str) -> SqlExpr:
    return _agg_expr(SqlExpr.min, cols, more_cols)


def max(cols: TryIter[str], *more_cols: str) -> SqlExpr:
    return _agg_expr(SqlExpr.max, cols, more_cols)


def _agg_expr(
    agg: Callable[[SqlExpr], SqlExpr], cols: TryIter[str], more_cols: Iterable[str]
) -> SqlExpr:
    return (
        try_iter(cols)
        .chain(more_cols)
        .then(lambda inner_cols: exp.Columns(this=inner_cols.into(args_into_glot)))
        .unwrap_or_else(lambda: exp.Columns(this=exp.Star()))
        .pipe(SqlExpr)
        .pipe(agg)
    )
