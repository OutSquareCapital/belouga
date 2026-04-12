from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, final

import pyochain as pc
from sqlglot import exp

from ._expr import Expr
from ._funcs import col, len
from ._meta import ExprPlan

if TYPE_CHECKING:
    from ._frame import LazyFrame
    from .sql import SqlExpr
    from .sql.typing import IntoExpr
    from .sql.utils import TryIter


def _root_column_name(expr: SqlExpr) -> pc.Option[str]:
    match expr.inner().unalias():
        case exp.Column() as column:
            return pc.Option.if_some(column.parts[-1]).map(lambda part: part.name)
        case _:
            return pc.NONE


@final
class LazyGroupBy:
    __slots__ = ("_aggregator", "_cols", "_constructor", "_keys")

    def __init__(
        self, frame: LazyFrame, keys: pc.Seq[SqlExpr], group_expr: pc.Option[str]
    ) -> None:
        self._constructor = frame.__class__
        self._keys = keys
        keys_names = keys.iter().filter_map(_root_column_name).collect(pc.Set)
        self._cols = (
            frame.columns.iter().filter(lambda name: name not in keys_names).collect()
        )
        self._aggregator = partial(
            frame.inner().relation.aggregate,
            group_expr=group_expr.unwrap_or_else(
                lambda: keys.iter().map(str).join(", ")
            ),
        )

    def _agg_columns(self, func: Callable[[Expr], Expr]) -> LazyFrame:
        return (
            self._cols
            .iter()
            .map(lambda name: col(name).pipe(func).alias(name))
            .into(self.agg)
        )

    def len(self, name: str | None = None) -> LazyFrame:
        return self.agg(pc.Option(name).map(len().alias).unwrap_or_else(len))

    def all(self) -> LazyFrame:
        return self._agg_columns(Expr.implode)

    def sum(self) -> LazyFrame:
        return self._agg_columns(Expr.sum)

    def mean(self) -> LazyFrame:
        return self._agg_columns(Expr.mean)

    def median(self) -> LazyFrame:
        return self._agg_columns(Expr.median)

    def min(self) -> LazyFrame:
        return self._agg_columns(Expr.min)

    def max(self) -> LazyFrame:
        return self._agg_columns(Expr.max)

    def first(self) -> LazyFrame:
        return self._agg_columns(Expr.first)

    def last(self) -> LazyFrame:
        return self._agg_columns(Expr.last)

    def n_unique(self) -> LazyFrame:
        return self._agg_columns(Expr.n_unique)

    def quantile(self, quantile: float, *, interpolation: bool = True) -> LazyFrame:
        return self._agg_columns(
            lambda expr: expr.quantile(quantile, interpolation=interpolation)
        )

    def agg(
        self,
        aggs: TryIter[IntoExpr] = None,
        *more_aggs: IntoExpr,
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
        keys = self._keys.iter().map(lambda c: c.into_duckdb())
        rel = self._cols.into(ExprPlan, aggs, more_aggs, named_aggs).agg_context(
            keys, self._aggregator
        )
        return self._constructor(rel)
