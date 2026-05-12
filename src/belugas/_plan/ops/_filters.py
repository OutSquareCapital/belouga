from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Option, Seq, Set, Some
from sqlglot import exp

from ..._core import into_expr
from ..._expr import Expr
from ..._funcs import all, col
from ...utils import try_iter
from .._deferred import DeferredDelta, ProjectionSpec

if TYPE_CHECKING:
    from ...typing import IntoExpr, IntoExprColumn, Schema, TryIter


def filter(
    predicates: TryIter[IntoExprColumn],
    more_predicates: Iterable[IntoExprColumn],
    constraints: dict[str, IntoExpr],
) -> DeferredDelta:

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
    return DeferredDelta(where=Some(condition))


def drop_rows(
    schema: Schema, subset: TryIter[str], fn: Callable[[Expr], Expr]
) -> DeferredDelta:
    return (
        Option(subset)
        .map(try_iter)
        .unwrap_or_else(schema.iter)
        .map(lambda name: col(name).pipe(fn))
        .into(lambda predicates: filter(predicates, (), {}))
    )


def limit(n: int) -> DeferredDelta:
    return DeferredDelta(limit=Some(exp.Literal.number(n)))


def drop(
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> DeferredDelta:

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
    return DeferredDelta(
        schema=Some(new_schema),
        projection=Some(ProjectionSpec(Exprs=Seq((cols.into(all).inner,)))),
    )
