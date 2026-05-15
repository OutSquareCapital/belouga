from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Seq, Some, Vec
from sqlglot import exp

from ..._core import Marker, Tables
from ..._funcs import col, row_number
from .._resolve import (
    ResolvedExpr,
    can_inline_select,
    find_all,
    has_window_ancestor,
    lookup_type,
    resolve_all,
)

if TYPE_CHECKING:
    from ..._expr import Expr
    from ...datatypes import DataType
    from ...typing import IntoExpr, Schema, TryIter


def with_columns(
    src_ast: exp.Select,
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
    has_windowed = projections.any(_is_windowed)
    broadcastr = _maybe_broadcast(include_source_cols=True, projections=projections)
    updates = (
        projections
        .iter()
        .map(lambda proj: (proj.name, broadcastr(proj.expr).alias(proj.name)))
        .collect(Dict)
    )
    new_schema = _with_columns_schema(schema, projections)
    if can_inline_select(src_ast) and not has_windowed:
        replaced = list[exp.Expr]()
        added = Vec[exp.Expr].new()
        (
            updates
            .items()
            .iter()
            .for_each_star(
                lambda name, expr: (
                    replaced.append(expr.inner)
                    if name in schema
                    else added.append(expr.inner)
                )
            )
        )
        star = exp.Star(replace=replaced) if replaced else exp.Star()
        new_ast = src_ast.select(star, *added, append=False, copy=False)
    else:
        source = src_ast.subquery(Tables.SRC, copy=False)
        source = _into_windowed(source) if has_windowed else source
        new_ast = (
            updates.into(_resolved).unpack_into(exp.select).from_(source, copy=False)
        )
    return new_ast, new_schema


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


def rename(schema: Schema, mapping: Mapping[str, str]) -> tuple[Iter[exp.Expr], Schema]:
    exprs = schema.iter().map(lambda c: exp.column(c).as_(mapping.get(c, c)))
    new_schema = (
        schema
        .items()
        .iter()
        .map_star(lambda name, dtype: (mapping.get(name, name), dtype))
        .collect(Dict)
    )
    return exprs, new_schema


def with_row_index(
    schema: Schema, name: str, order_by: TryIter[str]
) -> tuple[exp.Expr, Schema]:
    row_nb = row_number().window(order_by=order_by).sub(1).alias(name).inner
    new_schema = (
        Iter
        .once((name, exp.DType.BIGINT.into_expr()))
        .chain(schema.items())
        .collect(Dict)
    )
    return row_nb, new_schema


def union(lhs_ast: exp.Select, rhs_ast: exp.Select) -> exp.Select:
    slct = exp.select(exp.Star()).from_
    lhs = slct(lhs_ast.subquery(Tables.LHS, copy=False), copy=False)
    rhs = slct(rhs_ast.subquery(Tables.RHS, copy=False), copy=False)
    return exp.select(exp.Star()).from_(exp.union(lhs, rhs))


def cast(
    src_ast: exp.Select, schema: Schema, dtypes: Mapping[str, DataType] | DataType
) -> tuple[exp.Select, Schema]:
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
    src_ast: exp.Select, schema: Schema, func: Callable[[Expr], Expr]
) -> tuple[exp.Select, Schema]:

    exprs = schema.iter().map(lambda c: col(c).pipe(func).alias(c).inner)

    return select(src_ast, schema, exprs, (), {})


def select(
    src_ast: exp.Select,
    schema: Schema,
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    named_exprs: dict[str, IntoExpr],
) -> tuple[exp.Select, Schema]:
    projections = resolve_all(schema, exprs, more_exprs, named_exprs)
    has_windowed = projections.any(_is_windowed)
    broadcaster = _maybe_broadcast(include_source_cols=False, projections=projections)

    match projections.then_some():
        case Some(projs):
            new_schema = _select_schema(schema, projs)
            select_exprs = (
                projs
                .iter()
                .map(
                    lambda resolved: (
                        broadcaster(resolved.expr).alias(resolved.name).inner
                    )
                )
                .collect()
            )
            match src_ast:
                case source if can_inline_select(source) and not has_windowed:
                    ast = source.select(*select_exprs, append=False, copy=False)
                    if projs.all(lambda resolved: resolved.has_distinct):
                        return ast.distinct(copy=False), new_schema
                    return ast, new_schema
                case _:
                    rel = src_ast.subquery(Tables.SRC, copy=False).pipe(
                        lambda r: _into_windowed(r) if has_windowed else r
                    )
                    if projs.all(lambda resolved: resolved.has_distinct):
                        ast = (
                            exp.select(*select_exprs).from_(rel, copy=False).distinct()
                        )
                        return ast, new_schema
                    ast = exp.select(*select_exprs).from_(rel, copy=False)
                    return ast, new_schema
        case _:
            new_schema: Schema = Dict.from_ref({
                Marker.TEMP: exp.DType.NULL.into_expr()
            })
            return (
                exp.select(exp.null().as_(Marker.TEMP)).from_(
                    src_ast.subquery(Tables.SRC, copy=False), copy=False
                ),
                new_schema,
            )


def _select_schema(schema: Schema, projections: Seq[ResolvedExpr]) -> Schema:
    return (
        projections
        .iter()
        .map(lambda proj: (proj.name, lookup_type(proj.expr.inner, schema)))
        .collect(Dict)
    )


def _is_windowed(p: ResolvedExpr) -> bool:
    return p.name != Marker.TEMP and p.expr.inner.pipe(find_all, exp.Column).any(
        lambda col: col.parts[-1].name == Marker.TEMP
    )


def _into_windowed(source: exp.Expr) -> exp.Expr:
    row_nb = row_number().window().sub(1).alias(Marker.TEMP).inner
    return (
        exp
        .select(row_nb, exp.Star())
        .from_(source, copy=False)
        .subquery(Tables.SRC, copy=False)
    )


def _maybe_broadcast(
    *, include_source_cols: bool, projections: Seq[ResolvedExpr]
) -> Callable[[Expr], Expr]:
    if include_source_cols or not projections.all(
        lambda resolved: resolved.is_pure_reducer
    ):
        return broadcast_aggs
    return lambda x: x


def broadcast_aggs(expr: Expr) -> Expr:
    def _window_agg(node: exp.Expr) -> exp.Expr:
        match node:
            case exp.AggFunc() | exp.List() if not has_window_ancestor(node):
                return expr.__class__(node, expr.aliaser).window().inner
            case _:
                return node

    return expr.inner.transform(_window_agg).pipe(expr.__class__)
