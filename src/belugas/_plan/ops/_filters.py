from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pyochain import Iter, Option
from sqlglot import exp

from ..._core import Tables, into_expr
from ..._funcs import all, col
from ...utils import try_iter

if TYPE_CHECKING:
    from ..._expr import Expr
    from ...typing import IntoExpr, IntoExprColumn, Schema, TryIter


def filter(
    predicates: TryIter[IntoExprColumn],
    more_predicates: Iterable[IntoExprColumn],
    constraints: dict[str, IntoExpr],
) -> exp.Condition:

    def _constraint(k: str, val: IntoExpr) -> exp.Expr:
        return exp.column(k).eq(into_expr(val, as_col=False))

    return (
        try_iter(predicates)
        .chain(more_predicates)
        .map(lambda value: into_expr(value, as_col=True))
        .chain(Iter(constraints.items()).map_star(_constraint))
        .unpack_into(exp.and_)
    )


def drop_rows(
    schema: Schema, subset: TryIter[str], fn: Callable[[Expr], Expr]
) -> exp.Condition:
    return (
        Option(subset)
        .map(try_iter)
        .unwrap_or_else(schema.iter)
        .map(lambda name: col(name).pipe(fn))
        .into(lambda predicates: filter(predicates, (), {}))
    )


def drop(
    src_ast: exp.Select | exp.Union,
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> tuple[exp.Select, Schema]:

    def _process(e: IntoExprColumn) -> exp.Expr:
        expr = into_expr(e, as_col=True)
        name = expr.output_name
        _ = schema.pop(name)
        return expr

    selected = try_iter(columns).chain(more_columns).map(_process).into(all).inner
    return (
        exp.select(selected).from_(
            src_ast.subquery(Tables.SRC, copy=False), copy=False
        ),
        schema,
    )
