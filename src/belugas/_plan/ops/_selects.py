from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Seq, Some
from pyochain.traits import PyoIterable
from sqlglot import exp

from ..._core import Marker, Tables
from ..._funcs import col, row_number
from .._common import as_relation
from .._resolve import (
    ResolvedExpr,
    find_all,
    has_window_ancestor,
    lookup_type,
    resolve_all,
)

if TYPE_CHECKING:
    from pyochain.traits import PyoIterable

    from ..._expr import Expr
    from ...datatypes import DataType
    from ...typing import IntoExpr, Schema, TryIter


def with_columns(
    src_ast: exp.Selectable | exp.Table,
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
                _maybe_broadcast(proj.expr, broadcast_agg=broadcast_agg).alias(
                    proj.name
                ),
            )
        )
        .collect(Dict)
    )
    source = as_relation(src_ast)
    return exp.select(*updates.into(_resolved)).from_(
        projections.into(_into_windowed, source), copy=False
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


def rename(
    src_ast: exp.Selectable, schema: Schema, mapping: Mapping[str, str]
) -> tuple[exp.Selectable, Schema]:
    exprs = schema.iter().map(lambda c: exp.column(c).as_(mapping.get(c, c)))
    new_schema = (
        schema
        .items()
        .iter()
        .map_star(lambda name, dtype: (mapping.get(name, name), dtype))
        .collect(Dict)
    )
    return exp.select(*exprs).from_(as_relation(src_ast), copy=False), new_schema


def with_row_index(
    src_ast: exp.Selectable, schema: Schema, name: str, order_by: TryIter[str]
) -> tuple[exp.Selectable, Schema]:
    row_nb = row_number().window(order_by=order_by).sub(1).alias(name).inner
    new_schema = (
        Iter
        .once((name, exp.DType.BIGINT.into_expr()))
        .chain(schema.items())
        .collect(Dict)
    )
    return (
        exp.select(row_nb, exp.Star()).from_(as_relation(src_ast), copy=False),
        new_schema,
    )


def union(lhs_ast: exp.Selectable, rhs_ast: exp.Selectable) -> exp.Union:
    slct = exp.select(exp.Star()).from_
    lhs = slct(as_relation(lhs_ast, Tables.LHS.name), copy=False)
    rhs = slct(as_relation(rhs_ast, Tables.RHS.name), copy=False)
    return exp.union(lhs, rhs)


def cast(
    src_ast: exp.Selectable, schema: Schema, dtypes: Mapping[str, DataType] | DataType
) -> tuple[exp.Selectable, Schema]:
    match dtypes:
        case Mapping():
            dtype_map = Dict(dtypes)
            return select_all(
                src_ast,
                schema,
                lambda c: (
                    dtype_map
                    .get_item(c.inner.output_name)
                    .map(lambda dtype: c.cast(dtype.raw))
                    .unwrap_or(c)
                ),
            )
        case _:
            return select_all(src_ast, schema, lambda c: c.cast(dtypes.raw))


def select_all(
    src_ast: exp.Selectable, schema: Schema, func: Callable[[Expr], Expr]
) -> tuple[exp.Selectable, Schema]:

    exprs = schema.iter().map(lambda c: col(c).pipe(func).alias(c).inner)

    return select(src_ast, schema, exprs, (), {})


def select(
    src_ast: exp.Selectable,
    schema: Schema,
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    named_exprs: dict[str, IntoExpr],
) -> tuple[exp.Selectable, Schema]:
    projections = resolve_all(schema, exprs, more_exprs, named_exprs)
    broadcast_agg = _should_broadcast_agg(
        include_source_cols=False, projections=projections
    )

    match projections.then_some():
        case Some(projs):
            new_schema = _select_schema(schema, projs)
            rel = _into_windowed(projs, as_relation(src_ast))
            select_exprs = (
                projs
                .iter()
                .map(
                    lambda resolved: (
                        _maybe_broadcast(
                            resolved.expr,
                            broadcast_agg=broadcast_agg,
                        )
                        .alias(resolved.name)
                        .inner
                    )
                )
                .collect()
            )
            if projs.all(lambda resolved: resolved.has_distinct):
                ast = exp.select(*select_exprs).from_(rel, copy=False).distinct()
                return ast, new_schema
            ast = exp.select(*select_exprs).from_(rel, copy=False)
            return ast, new_schema
        case _:
            new_schema: Schema = Dict.from_ref({
                Marker.TEMP: exp.DType.NULL.into_expr()
            })
            return (
                exp.select(exp.null().as_(Marker.TEMP)).from_(as_relation(src_ast)),
                new_schema,
            )


def _select_schema(schema: Schema, projections: Seq[ResolvedExpr]) -> Schema:
    return (
        projections
        .iter()
        .map(lambda proj: (proj.name, lookup_type(proj.expr.inner, schema)))
        .collect(Dict)
    )


def _into_windowed(
    cols: PyoIterable[ResolvedExpr], source: exp.Table | exp.Subquery
) -> exp.Expr:
    def _is_windowed(p: ResolvedExpr) -> bool:
        return p.name != Marker.TEMP and p.expr.inner.pipe(find_all, exp.Column).any(
            lambda col: col.parts[-1].name == Marker.TEMP
        )

    if cols.any(_is_windowed):
        row_nb = row_number().window().sub(1).alias(Marker.TEMP).inner
        return (
            exp
            .select(row_nb, exp.Star())
            .from_(source, copy=False)
            .subquery(Tables.SRC.name, copy=False)
        )
    return source


def _should_broadcast_agg(
    *, include_source_cols: bool, projections: Seq[ResolvedExpr]
) -> bool:
    return include_source_cols or not projections.all(
        lambda resolved: resolved.is_pure_reducer
    )


def _maybe_broadcast(expr: Expr, *, broadcast_agg: bool) -> Expr:
    if broadcast_agg:
        return broadcast_aggs(expr)
    return expr


def broadcast_aggs(expr: Expr) -> Expr:
    def _window_agg(node: exp.Expr) -> exp.Expr:
        match node:
            case exp.AggFunc() | exp.List() if not has_window_ancestor(node):
                return expr.__class__(node, expr.aliaser).window().inner
            case _:
                return node

    return expr.inner.transform(_window_agg).pipe(expr.__class__)
