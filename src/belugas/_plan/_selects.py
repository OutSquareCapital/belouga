from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Seq, Some
from sqlglot import exp

from ._meta import Marker, ResolvedExpr, Tables, find_all, lookup_type, resolve_all

if TYPE_CHECKING:
    from pyochain.traits import PyoIterable

    from .._expr import Expr
    from ..typing import IntoExpr, Schema
    from ..utils import TryIter


def select(
    schema: Schema,
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    named_exprs: dict[str, IntoExpr],
) -> tuple[exp.Selectable, Schema]:
    projections = resolve_all(schema, exprs, more_exprs, named_exprs)

    def aliased(*, broadcast_agg: bool) -> exp.Select:
        def _into_expr(resolved: ResolvedExpr) -> exp.Expr:
            return resolved.as_aliased(broadcast_agg=broadcast_agg).inner

        return exp.select(*projections.iter().map(_into_expr))

    match projections.then_some():
        case Some(projs):
            new_schema = _select_schema(schema, projs)
            source = _into_windowed(projs)
            if projs.all(lambda resolved: resolved.has_projection_distinct):
                ast = aliased(broadcast_agg=False).from_(source).distinct()
            ast = aliased(
                broadcast_agg=_should_broadcast_agg(
                    include_source_cols=False, projections=projections
                )
            ).from_(source)
            return ast, new_schema
        case _:
            msg = "Need to decouple from ScanSource"
            raise NotImplementedError(msg)


def with_columns(
    schema: Schema,
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    named_exprs: dict[str, IntoExpr],
) -> tuple[exp.Select, Schema]:
    def _resolved(updates: Dict[str, Expr]) -> Iter[exp.Expr]:
        update_iter = updates.items().iter()
        if not updates.any(lambda name: name in schema):
            return update_iter.map_star(lambda _name, expr: expr.inner).insert(
                exp.Star()
            )
        return (
            schema
            .iter()
            .map(
                lambda name: updates.get_item(name).map_or(
                    exp.column(name), lambda expr: expr.inner
                )
            )
            .chain(
                update_iter.filter_star(
                    lambda name, _expr: name not in schema
                ).map_star(lambda _name, expr: expr.inner)
            )
        )

    projections = resolve_all(schema, exprs, more_exprs, named_exprs)
    broadcast_agg = _should_broadcast_agg(
        include_source_cols=True, projections=projections
    )
    updates = (
        projections
        .iter()
        .map(
            lambda proj: (
                proj.name,
                proj.as_aliased(broadcast_agg=broadcast_agg),
            )
        )
        .collect(Dict)
    )
    return exp.select(*updates.into(_resolved)).from_(
        projections.into(_into_windowed), copy=False
    ), _with_columns_schema(schema, projections)


def _with_columns_schema(schema: Schema, projections: Seq[ResolvedExpr]) -> Schema:
    updates = _select_schema(schema, projections)
    return (
        schema
        .items()
        .iter()
        .map_star(lambda name, dtype: (name, updates.get_item(name).unwrap_or(dtype)))
        .chain(updates.items().iter().filter_star(lambda name, _: name not in schema))
        .collect(Dict)
    )


def _select_schema(schema: Schema, projections: Seq[ResolvedExpr]) -> Schema:
    return (
        projections
        .iter()
        .map(lambda proj: (proj.name, lookup_type(proj.expr.inner, schema)))
        .collect(Dict)
    )


def _should_broadcast_agg(
    *, include_source_cols: bool, projections: Seq[ResolvedExpr]
) -> bool:
    return include_source_cols or not projections.all(
        lambda resolved: resolved.is_pure_reducer
    )


def _into_windowed(cols: PyoIterable[ResolvedExpr]) -> exp.Expr:
    from .._funcs import row_number

    def _is_windowed(p: ResolvedExpr) -> bool:
        return p.name != Marker.TEMP and p.expr.inner.pipe(find_all, exp.Column).any(
            lambda col: col.parts[-1].name == Marker.TEMP
        )

    if cols.any(_is_windowed):
        row_nb = row_number().window().sub(1).alias(Marker.TEMP).inner
        return (
            exp
            .select(row_nb, exp.Star())
            .from_(Tables.SRC)
            .subquery(Tables.SRC.name, copy=False)
        )
    return Tables.SRC
