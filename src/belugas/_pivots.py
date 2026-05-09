from collections.abc import Callable, Iterable, Sequence

from pyochain import Err, Iter, Null, Ok, Result, Seq, Some
from pyochain.traits import PyoIterable
from sqlglot import exp

from ._expr import Expr
from ._funcs import col
from ._meta import Tables
from .typing import PivotAgg, PythonLiteral
from .utils import TryIter, try_iter, try_seq

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


def pivot(  # noqa: ANN202, PLR0913, PLR0914, PLR0917
    base_columns: PyoIterable[str],
    on: TryIter[str],
    on_columns: Sequence[PythonLiteral],
    index: TryIter[str],
    values: TryIter[str],
    aggregate_function: PivotAgg,
    *,
    maintain_order: bool,
):
    def _cols_not_in(cols: Iterable[str]) -> Seq[str]:
        return (
            base_columns
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

    multi = val_cols.length() > 1
    agg = PIVOT_AGG[aggregate_function]

    def _aliased(name: str) -> Expr:
        expr = col(name).pipe(agg)
        return expr.alias(name) if multi else expr

    on_strs = Iter(on_columns).map(str)
    tail = (
        on_strs
        if not multi
        else val_cols.iter().flat_map(
            lambda vc: Iter(on_columns).map(lambda ov: f"{ov}_{vc}")
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

    table = exp.Table(this=Tables.SRC, pivots=[pivot_node])

    selected = (
        pivoted_cols
        .iter()
        .map(lambda n: exp.column(_case_sensitive_id(n)))
        .into(_select)
        .from_(table, copy=False)
    )

    return (
        (
            try_iter(idx_cols if maintain_order else None)
            .collect()
            .then(
                lambda cols: selected.order_by(*cols.iter().map(exp.column), copy=False)
            )
            .unwrap_or(selected)
        ),
        multi,
        idx_cols,
        val_cols,
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
    base_cols: PyoIterable[str],
    on: TryIter[str],
    index: TryIter[str],
    variable_name: str,
    value_name: str,
    order_by: TryIter[str],
) -> exp.Select:
    index_cols = try_iter(index).collect(dict.fromkeys)
    unpivot_cols = (
        try_iter(on)
        .then_some()
        .unwrap_or_else(
            lambda: base_cols.iter().filter(lambda name: name not in index_cols)
        )
        .collect(list)
    )

    into = exp.UnpivotColumns(this=variable_name, expressions=[value_name])
    pivot = Tables.SRC.pipe(
        lambda e: exp.Pivot(this=e, expressions=unpivot_cols, unpivot=True, into=into)
    ).pipe(lambda e: exp.Subquery(this=e))
    selected = exp.select(*index_cols, variable_name, value_name).from_(
        pivot, copy=False
    )
    return (
        try_iter(order_by)
        .then(lambda cols: selected.order_by(*cols, copy=False))
        .unwrap_or(selected)
    )
