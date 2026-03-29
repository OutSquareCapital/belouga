from collections.abc import Iterable

import duckdb
import sqlglot.expressions as exp
from pyochain import Err, Iter, Ok, Option, Result

from .typing import IntoExpr


class PQLConversionError(ValueError):
    """Raised when a conversion from a sqlglot expression to a DuckDB expression fails."""


def args_into_glot(args: Iterable[IntoExpr]) -> list[exp.Expr]:
    return Iter(args).filter_map(Option).map(pql_into_glot).collect(list)


def pql_into_glot(value: IntoExpr) -> exp.Expr:
    """Convert an IntoExpr value into a sqlglot expression node."""
    from .._expr import Expr
    from ._core import DuckHandler

    match value:
        case DuckHandler():
            return value.inner()
        case Expr():
            return value.inner().inner()
        case str():
            return exp.column(value)
        case _:
            return exp.convert(value)


def glot_into_duckdb(expr: exp.Expr) -> duckdb.Expression:

    try:
        match expr:
            case exp.Alias():
                return _alias_expr(expr)
            case exp.Column():
                return _col_expr(expr)
            case exp.Anonymous() | exp.Greatest() | exp.Least():
                return _anon_func_expr(expr)
            case exp.Lambda():
                return _lambda_expr(expr)
            case exp.Ordered():
                return _ordered_expr(expr)
            case _:
                return _raw_expr(expr)
    except duckdb.Error as e:
        msg = f"""Failed to convert expression to DuckDB!
    error:
        {e}
    expression:
        {expr!r}
    SQL:
        {expr.sql(dialect="duckdb", pretty=True)}
        """
        raise PQLConversionError(msg) from e


def _alias_name(value: exp.Expr | str | None) -> Result[str, ValueError]:
    match value:
        case exp.Identifier() as ident:
            return Ok(ident.name)
        case exp.Expr() as expr:
            return Ok(expr.sql(dialect="duckdb"))
        case str() as name:
            return Ok(name)
        case _:
            msg = "Alias expression requires a non-empty alias name"
            return Err(ValueError(msg))


def _ordered_expr(expr: exp.Ordered) -> duckdb.Expression:
    ordered = glot_into_duckdb(expr.this)  # pyright: ignore[reportAny]
    match expr.args.get("desc", False):  # pyright: ignore[reportMatchNotExhaustive]
        case True:
            ordered = ordered.desc()
        case False:
            ordered = ordered.asc()
    match expr.args.get("nulls_first", False):  # pyright: ignore[reportMatchNotExhaustive]
        case True:
            ordered = ordered.nulls_first()
        case False:
            ordered = ordered.nulls_last()
    return ordered


def _lambda_expr(expr: exp.Lambda) -> duckdb.Expression:
    match expr.expressions[0]:
        case exp.Identifier() as identifier:
            param = identifier.name
        case _ as item:  # pyright: ignore[reportAny]
            param = str(item)  # pyright: ignore[reportAny]

    return duckdb.LambdaExpression(param, glot_into_duckdb(expr.this))  # pyright: ignore[reportAny]


def _col_expr(expr: exp.Column) -> duckdb.Expression:
    parts = Iter(expr.parts).map(lambda part: part.name)
    return duckdb.ColumnExpression(*parts)


def _alias_expr(expr: exp.Alias) -> duckdb.Expression:

    return (
        _alias_name(expr.args.get("alias"))
        .map(lambda name: glot_into_duckdb(expr.this).alias(name))  # pyright: ignore[reportAny]
        .unwrap()
    )


def _anon_func_expr(
    expr: exp.Anonymous | exp.Greatest | exp.Least,
) -> duckdb.Expression:
    match expr:
        case exp.Anonymous() as anon_expr:
            name = anon_expr.name
            exprs = anon_expr.expressions
        case _ as func_expr:
            name = func_expr.sql_name()
            exprs = func_expr.iter_expressions()
    args = Iter(exprs).map(glot_into_duckdb)
    return duckdb.FunctionExpression(name, *args)


def _raw_expr(expr: exp.Expr) -> duckdb.Expression:
    return duckdb.SQLExpression(expr.sql(dialect="duckdb"))
