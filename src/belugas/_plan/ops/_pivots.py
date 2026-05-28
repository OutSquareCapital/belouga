from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Dict, Err, Iter, Null, Ok, Result, Seq, Some
from sqlglot import exp

from ..._core import Tables
from ..._expr import Expr
from ..._funcs import col
from ...utils import try_iter, try_seq

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ...typing import PivotAgg, PythonLiteral, Schema, TryIter

PIVOT_AGG: dict[PivotAgg, Callable[[Expr], Expr]] = {
    "min": Expr.min,
    "max": Expr.max,
    "first": Expr.first,
    "last": Expr.last,
    "sum": Expr.sum,
    "mean": Expr.mean,
    "median": Expr.median,
    "len": Expr.count,
    "count": Expr.count,
}


def pivot(  # noqa: PLR0913, PLR0914, PLR0917
    ast: exp.Select,
    schema: Schema,
    on: TryIter[str],
    on_columns: TryIter[PythonLiteral],
    index: TryIter[str],
    values: TryIter[str],
    aggregate_function: PivotAgg,
    *,
    maintain_order: bool,
    separator: str,
) -> tuple[exp.Select, Schema]:
    def _cols_not_in(cols: Iterable[str]) -> Seq[str]:
        return (
            schema
            .keys()
            .iter()
            .filter(lambda c: c not in on_cols and c not in cols)
            .collect()
        )

    def _get_idx_and_vals() -> Result[tuple[Seq[str], Seq[str]], ValueError]:
        match (try_seq(index), try_seq(values)):
            case (Some(idx), Some(vals)):
                return Ok((idx, vals))
            case (Some(idx), Null()):
                return Ok((idx, _cols_not_in(idx)))
            case (Null(), Some(vals)):
                return Ok((_cols_not_in(vals), vals))
            case _:
                msg = "`pivot` needs either `index` or `values` to be specified"
                return Err(ValueError(msg))

    on_cols = try_iter(on).collect()
    idx_cols, val_cols = _get_idx_and_vals().unwrap()
    on_values = try_iter(on_columns).map(str).collect()

    multi = val_cols.len() > 1
    agg = PIVOT_AGG[aggregate_function]

    def _aliased(name: str) -> Expr:

        expr = col(name).pipe(agg)
        return expr.alias(name) if multi else expr

    field_exprs = try_iter(on_columns).map(exp.convert).collect(list)
    pivot_field = exp.In(this=exp.column(on_cols.first()), expressions=field_exprs)

    group = idx_cols.then(
        lambda cols: exp.Group(expressions=cols.iter().map(exp.column).collect(list))
    ).unwrap_or_none()

    pivot_exprs = val_cols.iter().map(_aliased).map(lambda c: c.inner).collect(list)
    pivot_cols = (
        idx_cols
        .iter()
        .chain(
            on_values.iter().flat_map(
                lambda ov: val_cols.iter().map(lambda vc: f"{ov}_{vc}")
            )
            if multi
            else on_values
        )
        .map(lambda name: exp.to_identifier(name, quoted=True))
        .collect(list)
    )
    pivot_alias = exp.TableAlias(
        this=exp.to_identifier("_pivot", quoted=False), columns=pivot_cols
    )
    pivot_node = exp.Pivot(
        expressions=pivot_exprs, fields=[pivot_field], group=group, alias=pivot_alias
    )
    table = ast.subquery(Tables.SRC, copy=False)
    table.set("pivots", [pivot_node])

    ordered = (
        exp
        .select(exp.Star())
        .from_(table, copy=False)
        .pipe(
            lambda selected: (
                try_iter(idx_cols if maintain_order else None)
                .collect()
                .then(
                    lambda cols: selected.order_by(
                        *cols.iter().map(exp.column), copy=False
                    )
                )
                .unwrap_or(selected)
            )
        )
    )

    unknown = exp.DType.UNKNOWN.into_expr()

    if multi:

        def _idx_expr(name: str) -> exp.Expr:
            return exp.column(name)

        def _rename(vc: str) -> Iter[exp.Expr]:
            def _renamed(ov: str) -> exp.Expr:
                return exp.column(f"{ov}_{vc}", quoted=True).as_(
                    f"{vc}{separator}{ov}", quoted=True
                )

            return on_values.iter().map(_renamed)

        final_schema: Schema = (
            idx_cols
            .iter()
            .map(lambda name: (name, schema.get_item(name).unwrap()))
            .chain(
                val_cols.iter().flat_map(
                    lambda vc: on_values.iter().map(
                        lambda ov: (f"{vc}{separator}{ov}", unknown)
                    )
                )
            )
            .collect(Dict)
        )
        final_ast = (
            idx_cols
            .iter()
            .map(_idx_expr)
            .chain(val_cols.iter().flat_map(_rename))
            .unpack_into(exp.select, copy=False)
            .from_(ordered.subquery(alias="_pivot", copy=False), copy=False)
        )
        return final_ast, final_schema

    return ordered, (
        idx_cols
        .iter()
        .map(lambda name: (name, schema.get_item(name).unwrap()))
        .chain(on_values.iter().map(lambda ov: (ov, unknown)))
        .collect(Dict)
    )


def unpivot(  # noqa: PLR0913, PLR0917
    ast: exp.Select,
    schema: Schema,
    on: TryIter[str],
    index: TryIter[str],
    variable_name: str,
    value_name: str,
    order_by: TryIter[str],
) -> tuple[exp.Select, Schema]:
    index_set = try_iter(index).collect(dict.fromkeys)
    match on:
        case None:
            first_dtype = schema.iter().next()
        case _:
            first_dtype = try_iter(on).next()
    value_dtype = first_dtype.and_then(schema.get_item).unwrap_or_else(
        exp.DType.UNKNOWN.into_expr
    )
    new_schema = (
        schema
        .items()
        .iter()
        .filter_star(lambda name, _: name in index_set)
        .chain((
            (variable_name, exp.DType.VARCHAR.into_expr()),
            (value_name, value_dtype),
        ))
        .collect(Dict)
    )

    index_cols = try_iter(index).collect(dict.fromkeys)
    unpivot_cols = (
        try_iter(on)
        .then_some()
        .unwrap_or_else(
            lambda: schema.iter().filter(lambda name: name not in index_cols)
        )
        .collect(list)
    )

    into = exp.UnpivotColumns(this=variable_name, expressions=[value_name])
    pivot = (
        ast
        .subquery(Tables.SRC, copy=False)
        .pipe(
            lambda e: exp.Pivot(
                this=e, expressions=unpivot_cols, unpivot=True, into=into
            )
        )
        .pipe(lambda e: exp.Subquery(this=e))
    )

    new_ast = (
        exp
        .select(*index_cols, variable_name, value_name)
        .from_(pivot, copy=False)
        .pipe(
            lambda selected: (
                try_iter(order_by)
                .then(lambda cols: selected.order_by(*cols, copy=False))
                .unwrap_or(selected)
            )
        )
    )
    return new_ast, new_schema
