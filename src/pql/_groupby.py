from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, final

import pyochain as pc

from ._expr import Expr
from ._funcs import col, len
from ._meta import ExprPlan
from ._schema import Schema
from .sql import SqlExpr

if TYPE_CHECKING:
    from ._frame import LazyFrame
    from .sql.typing import IntoExpr
    from .sql.utils import TryIter


@final
class LazyGroupBy:
    __slots__ = ("_aggregator", "_frame", "_keys", "_schema")

    def __init__(
        self, frame: LazyFrame, keys: pc.Seq[SqlExpr], group_expr: pc.Option[str]
    ) -> None:
        self._frame = frame
        self._keys = keys
        keys_names = keys.iter().filter_map(SqlExpr.root_column_name).collect(pc.Set)
        self._schema = (
            frame.schema
            .items()
            .iter()
            .filter_star(lambda name, _: name not in keys_names)
            .collect(Schema)
        )
        self._aggregator = partial(
            self._frame.inner().aggregate,
            group_expr=group_expr.unwrap_or_else(
                lambda: keys.iter().map(str).join(", ")
            ),
        )

    def _agg_columns(self, func: Callable[[Expr], Expr]) -> LazyFrame:
        return (
            self._schema
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
        rel = self._schema.into(ExprPlan, aggs, more_aggs, named_aggs).agg_context(
            keys, self._aggregator
        )
        return self._frame.__class__(rel)
