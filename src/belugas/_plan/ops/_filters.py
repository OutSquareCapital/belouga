from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Set, option
from sqlglot import exp

from ..._core import Tables, into_expr
from ..._funcs import col
from ...utils import try_iter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ..._expr import Expr
    from ...typing import IntoExpr, IntoExprColumn, Schema, TryIter


def filter(
    predicates: TryIter[IntoExprColumn],
    more_predicates: Iterable[IntoExprColumn],
    constraints: Dict[str, IntoExpr],
) -> exp.Condition:

    def _constraint(k: str, val: IntoExpr) -> exp.Expr:
        return exp.column(k).eq(into_expr(val, as_col=False))

    return (
        try_iter(predicates)
        .chain(more_predicates)
        .map(lambda value: into_expr(value, as_col=True))
        .chain(constraints.items().iter().map_star(_constraint))
        .unpack_into(exp.and_)
    )


def drop_rows(
    schema: Schema, subset: TryIter[str], fn: Callable[[Expr], Expr]
) -> exp.Condition:
    return (
        option(subset)
        .map(try_iter)
        .unwrap_or_else(schema.iter)
        .map(lambda name: col(name).pipe(fn).inner)
        .unpack_into(exp.and_)
    )


def drop(
    ast: exp.Select,
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> exp.Select:

    def _process(e: IntoExprColumn) -> exp.Expr:
        expr = into_expr(e, as_col=True)
        name = expr.output_name
        _ = schema.pop(name)
        return expr

    excluded = try_iter(columns).chain(more_columns).map(_process)

    def _as_subquery() -> exp.Select:
        return exp.select(exp.Star(except_=excluded.collect(list))).from_(
            ast.subquery(Tables.SRC, copy=False), copy=False
        )

    match ast.args.get("distinct"), ast.selects:
        case None, [exp.Star()]:
            return ast.select(
                exp.Star(except_=excluded.collect(list)), copy=False, append=False
            )
        case None, current_cols if not ast.is_star:
            excluded_names = excluded.map(lambda expr: expr.output_name).collect(Set)
            exprs = (
                Iter(current_cols)
                .filter(lambda expr: expr.output_name not in excluded_names)
                .collect(list)
            )
            ast.set("expressions", exprs)
            return ast
        case _:
            return _as_subquery()
