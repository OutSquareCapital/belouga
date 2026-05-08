from collections.abc import Callable, Iterable
from typing import final

from pyochain import Iter, Option
from sqlglot import exp

from ._core import into_expr, into_expr_list
from ._expr import Expr
from .typing import IntoExpr, IntoExprColumn, PythonLiteral
from .utils import TryIter, try_iter


def reduce(
    exprs: Iterable[IntoExpr], function: Callable[[Expr, IntoExpr], Expr]
) -> Expr:
    """Reduces an `Iterable` of `IntoExpr` into a single `Expr`.

    Done by applying a binary *fn* (defaulting to logical `AND`) to each item, after converting them with `into_expr`.

    Args:
        exprs (Iterable[IntoExpr]): The expressions to reduce.
        function (Callable[[Expr, IntoExpr], Expr]): The binary function to apply for reduction.

    Returns:
        Expr: The result of reducing the expressions with the given function.
    """
    return Iter(exprs).map(lambda value: Expr.new(value, as_col=True)).reduce(function)


def row_number() -> Expr:
    """Create a ROW_NUMBER() expression.

    Returns:
        Expr: An expression representing the ROW_NUMBER() function.
    """
    return Expr(exp.RowNumber())


def unnest(
    col: IntoExprColumn, max_depth: int | None = None, *, recursive: bool = False
) -> Expr:
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
        col (Expr): The column to unnest.
        max_depth (int | None): Maximum depth of recursive unnesting.
        recursive (bool): Whether to recursively unnest lists and structs (default: `False`).  Note that lists *within* structs are not unnested.

    Returns:
        Expr: An expression representing the unnesting operation.
    """
    expr = exp.Explode(this=into_expr(col), max_depth=max_depth, recursive=recursive)
    return Expr(expr)


@final
class Col:
    __slots__ = ()

    def __call__(self, name: str, table: str | None = None) -> Expr:
        return Expr(exp.column(name, table=table))

    def __getattr__(self, name: str) -> Expr:
        return self(name)


col = Col()


ELEM_NAME = "element"

ELEMENT = col(ELEM_NAME)
_ELEM_ID = exp.to_identifier(ELEM_NAME)


def element() -> Expr:
    return ELEMENT


def fn_once(rhs: IntoExpr) -> Expr:
    def _bind(node: exp.Expr) -> exp.Expr:
        match node:
            case exp.Column() if node.name == ELEM_NAME:
                return _ELEM_ID
            case _:
                return node

    body = into_expr(rhs).transform(_bind)
    return Expr(exp.Lambda(this=body, expressions=[_ELEM_ID]))


def all(exclude: TryIter[IntoExprColumn] = None) -> Expr:
    from .selectors import Resolver

    exclude_opt: Option[TryIter[IntoExprColumn]] = Option(exclude)
    return (
        exclude_opt
        .map(lambda x: try_iter(x).map(into_expr).collect())
        .map(lambda exc: exp.Star(except_=exc))
        .unwrap_or_else(exp.Star)
        .pipe(Expr, Resolver.all_fn(exclude_opt).into_meta())
    )


def any(names: TryIter[str], *more_names: str, ignore_nulls: bool = True) -> Expr:
    if ignore_nulls:
        return _agg_expr(
            lambda expr: expr.any().fill_null(lit(value=False)),
            names,
            more_names,
        )
    return _agg_expr(Expr.any, names, more_names)


def lit(value: PythonLiteral) -> Expr:
    """Create a literal expression.

    Args:
        value (PythonLiteral): The literal value to create an expression for.

    Returns:
        Expr: An expression representing the literal value.
    """
    return Expr(exp.convert(value))


def len() -> Expr:
    """Return the number of rows.

    Returns:
        Expr
    """
    from ._meta import Marker

    return lit(1).count().alias(Marker.LEN)


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    """Create a COALESCE expression.

    Args:
        exprs (TryIter[IntoExpr]): The expressions to coalesce.
        *more_exprs (IntoExpr): Additional expressions to coalesce.

    Returns:
        Expr: An expression representing the COALESCE operation.
    """
    all_exprs = try_iter(exprs).chain(more_exprs)
    expr = all_exprs.next().map(Expr.new, as_col=True).unwrap()
    return expr.coalesce(all_exprs).alias(expr.inner.output_name)


def arctan2(y: IntoExprColumn | float, x: IntoExprColumn | float) -> Expr:
    return Expr.new(y, as_col=True).atan2(Expr.new(x, as_col=True))


_HORIZONTAL_ERR = "At least one expression is required."


def _into_col(value: IntoExpr) -> Expr:
    return Expr.new(value, as_col=True)


def _horizontal_fn(
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    fn: Callable[[Expr, *tuple[IntoExpr]], Expr],
) -> Expr:
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
    fn: Callable[[Expr, IntoExpr], Expr],
) -> Expr:
    all_exprs = try_iter(exprs).chain(more_exprs).map(_into_col).collect()
    return all_exprs.iter().reduce(fn).alias(all_exprs.first().inner.output_name)


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, Expr.least)


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_fn(exprs, more_exprs, Expr.greatest)


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_reduce(
        exprs, more_exprs, lambda lhs, rhs: lhs.add(_into_col(rhs).coalesce(0))
    )


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_reduce(exprs, more_exprs, Expr.and_)


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    return _horizontal_reduce(exprs, more_exprs, Expr.or_)


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> Expr:
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
                .reduce(Expr.add)
                .truediv(
                    vals
                    .iter()
                    .map(lambda value: value.is_not_null().cast(dtype))
                    .reduce(Expr.add)
                )
                .alias(vals.first().inner.output_name)
            )
        )
        .expect(_HORIZONTAL_ERR)
    )


def sum(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.sum, cols, more_cols)


def count(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.count, cols, more_cols)


def approx_n_unique(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.approx_count_distinct, cols, more_cols)


def n_unique(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.n_unique, cols, more_cols)


def first(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.first, cols, more_cols)


def last(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.last, cols, more_cols)


def cum_count(cols: TryIter[str], *more_cols: str, reverse: bool = False) -> Expr:
    return _agg_expr(lambda expr: expr.cum_count(reverse=reverse), cols, more_cols)


def cum_sum(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.cum_sum, cols, more_cols)


def mean(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.mean, cols, more_cols)


def median(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.median, cols, more_cols)


def min(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.min, cols, more_cols)


def max(cols: TryIter[str], *more_cols: str) -> Expr:
    return _agg_expr(Expr.max, cols, more_cols)


def std(column: str, *, ddof: int = 1) -> Expr:
    return col(column).std(ddof)


def var(column: str, *, ddof: int = 1) -> Expr:
    return col(column).var(ddof)


def _agg_expr(
    agg: Callable[[Expr], Expr], cols: TryIter[str], more_cols: Iterable[str]
) -> Expr:
    from .selectors import Resolver

    all_cols = try_iter(cols).chain(more_cols).collect().then_some()
    meta = all_cols.map(Resolver.fixed).unwrap_or_else(Resolver.all_columns).into_meta()
    return (
        all_cols
        .map(lambda inner_cols: exp.Columns(this=inner_cols.into(into_expr_list)))
        .unwrap_or_else(lambda: exp.Columns(this=exp.Star()))
        .pipe(Expr, meta)
        .pipe(agg)
    )
