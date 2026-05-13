from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

from pyochain import Dict, Err, Iter, Null, Ok, Result, Seq, Some
from sqlglot import exp

from ..._core import Tables
from ..._expr import Expr
from ..._funcs import col
from ...utils import try_iter, try_seq
from .._common import as_relation

if TYPE_CHECKING:
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
    ast: exp.Selectable,
    schema: Schema,
    on: TryIter[str],
    on_columns: Sequence[PythonLiteral],
    index: TryIter[str],
    values: TryIter[str],
    aggregate_function: PivotAgg,
    *,
    maintain_order: bool,
    separator: str,
) -> tuple[exp.Selectable, Schema]:
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
    on_values = Iter(on_columns).map(str).collect()

    multi = val_cols.length() > 1
    agg = PIVOT_AGG[aggregate_function]

    def _aliased(name: str) -> Expr:

        expr = col(name).pipe(agg)
        return expr.alias(name) if multi else expr

    tail = (
        on_values.iter()
        if not multi
        else val_cols.iter().flat_map(
            lambda vc: on_values.iter().map(lambda ov: f"{ov}_{vc}")
        )
    )
    pivoted_cols = idx_cols.iter().chain(tail).collect()

    field_exprs = Iter(on_columns).map(exp.convert).collect(list)
    pivot_field = exp.In(this=exp.column(on_cols.first()), expressions=field_exprs)

    group_opt = idx_cols.then(
        lambda cols: exp.Group(expressions=cols.iter().map(exp.column).collect(list))
    )
    group = group_opt.unwrap() if group_opt.is_some() else None

    pivot_exprs = val_cols.iter().map(_aliased).map(lambda c: c.inner).collect(list)
    pivot_cols = (
        pivoted_cols
        .iter()
        .skip(idx_cols.length())
        .map(_case_sensitive_id)
        .collect(list)
    )
    pivot_node = exp.Pivot(
        expressions=pivot_exprs, fields=[pivot_field], group=group, columns=pivot_cols
    )

    match as_relation(ast):
        case exp.Table() as table:
            table = exp.Table(
                this=exp.to_identifier(table.name),
                db=table.args.get("db"),
                catalog=table.args.get("catalog"),
                alias=table.args.get("alias"),
                pivots=[pivot_node],
            )
        case exp.Subquery(this=exp.Selectable() as inner):
            alias = exp.TableAlias(this=exp.to_identifier(Tables.SRC.name))
            table = exp.Subquery(this=inner, alias=alias, pivots=[pivot_node])
        case _:
            msg = f"Unexpected AST node type for pivot input: {ast.__class__.__name__}"
            raise ValueError(msg)

    selected = (
        pivoted_cols
        .iter()
        .map(lambda n: exp.column(_case_sensitive_id(n)))
        .into(_select)
        .from_(table, copy=False)
    )

    ordered = (
        try_iter(idx_cols if maintain_order else None)
        .collect()
        .then(lambda cols: selected.order_by(*cols.iter().map(exp.column), copy=False))
        .unwrap_or(selected)
    )

    unknown = exp.DType.UNKNOWN.into_expr()

    if multi:
        subq = exp.Subquery(
            this=ordered,
            alias=exp.TableAlias(this=exp.to_identifier("_pivot")),
        )

        def _idx_expr(name: str) -> exp.Expr:
            return exp.column(name)

        def _rename(vc: str) -> Iter[exp.Expr]:
            def _renamed(ov: str) -> exp.Expr:
                return exp.column(_case_sensitive_id(f"{ov}_{vc}")).as_(
                    f"{vc}{separator}{ov}", quoted=True
                )

            return on_values.iter().map(_renamed)

        rename_exprs = (
            idx_cols
            .iter()
            .map(_idx_expr)
            .chain(val_cols.iter().flat_map(_rename))
            .collect(list)
        )
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
        return exp.select(*rename_exprs).from_(subq, copy=False), final_schema

    return ordered, (
        idx_cols
        .iter()
        .map(lambda name: (name, schema.get_item(name).unwrap()))
        .chain(on_values.iter().map(lambda ov: (ov, unknown)))
        .collect(Dict)
    )


def _case_sensitive_id(name: str) -> exp.Identifier:
    """Build a quoted identifier that survives `qualify` normalization.

    In DuckDB, all identifiers (even quoted) are normalized to
    lowercase by `sqlglot.optimizer.normalize_identifiers`, which is
    run by `qualify` and `annotate_types` during schema inference in
    `_compute_schema`.

    For pivoted output columns whose names mirror the user-provided
    `on_columns` literals (e.g. ``"Engineering"``, ``"Sales"``), we
    want the post-pivot column names to preserve their original
    case rather than be downcased into ``"engineering"`` /
    ``"sales"``. The literals inside the ``IN (...)`` clause already
    survive normalization (they are `exp.Literal`, not identifiers),
    but the identifiers we wire into ``Pivot.args["columns"]`` and
    the explicit projection ``SELECT "Engineering", "Sales" ...``
    that replaces ``SELECT *`` after the pivot are subject to it.

    The escape hatch documented by sqlglot for this exact case is
    the per-node ``meta["case_sensitive"] = True`` flag, which makes
    `normalize_identifiers` skip the node entirely (see
    `sqlglot.optimizer.normalize_identifiers.normalize_identifiers`).

    Note:
        Once https://github.com/tobymao/sqlglot/pull/7586 is merged
        and released, the cleaner alternative is to drop both
        ``Pivot.args["columns"]`` and this meta flag, and instead
        rename the pivot output positionally via the standard
        ``PIVOT(...) AS alias(c1, c2, ...)`` mechanism (a
        `TableAlias(columns=[...])` on the Pivot's alias). The PR
        teaches `Pivot.output_columns` and `annotate_types` to
        propagate those alias-renamed names with their proper
        types, removing the need to bypass normalization manually.

    Returns:
        exp.Identifier
    """
    ident = exp.to_identifier(name, quoted=True)
    ident.meta["case_sensitive"] = True
    return ident


def _select(exprs: Iterable[exp.Expr | str]) -> exp.Select:
    return exp.select(*exprs)


def unpivot(  # noqa: PLR0913, PLR0917
    ast: exp.Selectable,
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
        .pipe(as_relation)
        .pipe(
            lambda e: exp.Pivot(
                this=e, expressions=unpivot_cols, unpivot=True, into=into
            )
        )
        .pipe(lambda e: exp.Subquery(this=e))
    )

    selected = exp.select(*index_cols, variable_name, value_name).from_(
        pivot, copy=False
    )
    return (
        try_iter(order_by)
        .then(lambda cols: selected.order_by(*cols, copy=False))
        .unwrap_or(selected),
        new_schema,
    )
