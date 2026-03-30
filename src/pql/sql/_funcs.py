from collections.abc import Callable, Iterable
from typing import final

import pyochain as pc
import sqlglot
from duckdb import Expression
from sqlglot import exp

from ._conversions import args_into_glot, pql_into_glot
from ._core import DuckHandler
from ._expr import SqlExpr
from .typing import IntoExpr, IntoExprColumn, PythonLiteral
from .utils import TryIter, try_chain, try_iter


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
        pc.Iter(exprs).map(lambda value: into_expr(value, as_col=True)).reduce(function)
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


def coalesce(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    """Create a COALESCE expression.

    Args:
        exprs (TryIter[IntoExpr]): The expressions to coalesce.
        *more_exprs (IntoExpr): Additional expressions to coalesce.

    Returns:
        SqlExpr: An expression representing the COALESCE operation.
    """
    exprs = try_chain(exprs, more_exprs).into(args_into_glot)
    return SqlExpr(exp.Coalesce(this=exprs[0], expressions=exprs))


_HORIZONTAL_ERR = "At least one expression is required."


def _horizontal_fn(
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    fn: Callable[[SqlExpr, *tuple[IntoExpr]], SqlExpr],
) -> SqlExpr:
    return try_chain(exprs, more_exprs).into(
        lambda all_exprs: (
            all_exprs
            .next()
            .map(lambda value: into_expr(value, as_col=True))
            .map(
                lambda x: fn(
                    x, *all_exprs.map(lambda value: into_expr(value, as_col=True))
                )
            )
        ).expect(_HORIZONTAL_ERR)
    )


def min_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_fn(exprs, more_exprs, SqlExpr.least)


def max_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return _horizontal_fn(exprs, more_exprs, SqlExpr.greatest)


def sum_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return try_chain(exprs, more_exprs).into(
        reduce, lambda lhs, rhs: lhs.add(coalesce(rhs, 0))
    )


def all_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return try_chain(exprs, more_exprs).into(reduce, SqlExpr.and_)


def any_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return try_chain(exprs, more_exprs).into(reduce, SqlExpr.or_)


def mean_horizontal(exprs: TryIter[IntoExpr], *more_exprs: IntoExpr) -> SqlExpr:
    return (
        try_chain(exprs, more_exprs)
        .map(lambda value: into_expr(value, as_col=True))
        .collect()
        .then(
            lambda vals: (
                vals
                .iter()
                .map(lambda value: coalesce(value, 0))
                .reduce(SqlExpr.add)
                .truediv(
                    vals
                    .iter()
                    .map(
                        lambda value: value.is_not_null().cast(
                            exp.DataType.build(exp.DType.BIGINT)  # pyright: ignore[reportUnknownMemberType]
                        )
                    )
                    .reduce(SqlExpr.add)
                )
            )
        )
        .expect(_HORIZONTAL_ERR)
    )


def into_expr(value: IntoExpr, *, as_col: bool = False) -> SqlExpr:  # noqa: PLR0911
    """Convert a value to a `SqlExpr`.

    Args:
        value (IntoExpr): The value to convert.
        as_col (bool): Whether to treat `str` values as column names (default: `False`).

    Returns:
        SqlExpr
    """
    from .._expr import Expr

    match value:
        case SqlExpr():
            return value
        case DuckHandler():
            return SqlExpr(value.inner())
        case Expr():
            return value.inner()
        case str() if as_col:
            return col(value)
        case exp.Expr():
            return SqlExpr(value)
        case Expression():
            return SqlExpr(sqlglot.parse_one(str(value), dialect="duckdb"))
        case _:
            return lit(value)
