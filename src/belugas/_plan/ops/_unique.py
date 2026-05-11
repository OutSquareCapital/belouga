from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Err, Null, Ok, Result, Seq, Some
from sqlglot import exp

from ..._core import Tables
from ..._expr import Expr
from ..._funcs import col, lit
from ...utils import try_seq

if TYPE_CHECKING:
    from ...typing import TryIter, TrySeq, UniqueKeepStrategy


def unique(
    subset: TryIter[str], keep: UniqueKeepStrategy, order_by: TrySeq[str]
) -> Result[exp.Selectable, ValueError]:
    match (keep, try_seq(order_by), try_seq(subset)):
        case ("none", _, Null()):
            return Ok(_none_on_all())
        case ("any", _, Null()) | ("first" | "last", Some(_), Null()):
            return Ok(exp.select(exp.Star()).from_(Tables.SRC, copy=False).distinct())
        case ("none", _, Some(subset_names)):
            res = _none_on_subset(subset_names)
            return Ok(res)
        case ("last", Some(order_cols), Some(subset_names)):
            return Ok(
                _distinct_on(subset_names, order_cols, descending=True, nulls_last=True)
            )
        case ("any" | "first", order_cols, Some(subset_names)):
            return Ok(
                _distinct_on(
                    subset_names,
                    order_cols.unwrap_or_else(Seq[str].new),
                    descending=False,
                    nulls_last=False,
                )
            )
        case _:
            msg = """`order_by` must be specified when `keep` is 'first' or 'last'
            because LazyFrame makes no assumptions about row order."""
            return Err(ValueError(msg))


def _none_on_subset(subset_names: Seq[str]) -> exp.Select:

    subset_exprs = subset_names.iter().map(exp.column).collect()
    rhs = (
        exp
        .select(*subset_exprs)
        .from_(Tables.SRC, copy=False)
        .group_by(*subset_exprs)
        .having(lit(1).count().eq(1).inner)
        .subquery(Tables.RHS.name, copy=False)
    )
    condition = (
        subset_names
        .iter()
        .map(
            lambda name: exp.NullSafeEQ(
                this=col(name, table=Tables.LHS.name).inner,
                expression=col(name, table=Tables.RHS.name).inner,
            )
        )
        .map(Expr)
        .reduce(Expr.and_)
        .inner
    )
    return (
        exp
        .select("lhs.*")
        .from_("src AS lhs")
        .join(rhs, on=condition, join_type="semi")
    )


def _none_on_all() -> exp.Select:
    return (
        exp
        .select(exp.Star())
        .from_(Tables.SRC, copy=False)
        .group_by("ALL")
        .having(lit(1).count().eq(1).inner)
    )


def _distinct_on(
    subset_names: Seq[str],
    order_names: Seq[str],
    *,
    descending: bool,
    nulls_last: bool,
) -> exp.Select:

    order_exprs = (
        subset_names
        .iter()
        .map(col)
        .chain(
            order_names.iter().map(
                lambda name: col(name).order_by(
                    descending=descending, nulls_last=nulls_last
                )
            )
        )
        .map(lambda expr: expr.inner)
    )
    return (
        exp
        .select(exp.Star())
        .from_(Tables.SRC, copy=False)
        .distinct(*subset_names)
        .order_by(*order_exprs)
    )
