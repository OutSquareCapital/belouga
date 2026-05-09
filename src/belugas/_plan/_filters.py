from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Option, Set
from sqlglot import exp

from .._core import into_expr
from ..utils import TryIter, try_iter
from ._meta import Tables

if TYPE_CHECKING:
    from .._expr import Expr
    from ..typing import IntoExpr, IntoExprColumn, Schema


def filter(
    predicates: TryIter[IntoExprColumn],
    *more_predicates: IntoExprColumn,
    **constraints: IntoExpr,
) -> exp.Select:
    from .._expr import Expr
    from .._funcs import col

    def _constraint(k: str, val: IntoExpr) -> Expr:
        return col(k).eq(into_expr(val, as_col=False))

    condition = (
        try_iter(predicates)
        .chain(more_predicates)
        .map(lambda value: Expr.new(value, as_col=True))
        .chain(Iter(constraints.items()).map_star(_constraint))
        .reduce(Expr.and_)
        .inner
    )
    return (
        exp
        .select(exp.Star())
        .from_(Tables.SRC, copy=False)
        .where(condition, copy=False)
    )


def drop_rows(
    schema: Schema, subset: TryIter[str], fn: Callable[[Expr], Expr]
) -> exp.Select:
    from .._funcs import col

    return (
        Option(subset)
        .map(try_iter)
        .unwrap_or_else(lambda: schema.keys().iter())
        .map(lambda name: col(name).pipe(fn))
        .into(filter)
    )


def drop(
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> tuple[exp.Select, Schema]:
    from .._expr import Expr
    from .._funcs import all

    cols = (
        try_iter(columns)
        .chain(more_columns)
        .map(lambda e: Expr.new(e, as_col=True))
        .collect()
    )
    to_drop = cols.iter().map(lambda e: e.inner.output_name).collect(Set)
    new_schema = (
        schema
        .items()
        .iter()
        .filter_star(lambda name, _: name not in to_drop)
        .collect(Dict)
    )
    return (
        exp.select(cols.into(all).inner).from_(Tables.SRC, copy=False),
        new_schema,
    )
