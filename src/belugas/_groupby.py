from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyochain import Option

from ._expr import Expr
from ._frame import LazyFrame
from ._funcs import len
from ._plan import nodes

if TYPE_CHECKING:
    from .typing import IntoExpr, TryIter


@dataclass(slots=True)
class LazyGroupBy:
    _inner: nodes.GroupBy

    def len(self, name: str | None = None) -> LazyFrame:
        return self.agg(Option(name).map(len().alias).unwrap_or_else(len))

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

    def _agg_columns(self, func: Callable[[Expr], Expr]) -> LazyFrame:
        node = nodes.AggColumns(self._inner, func=func)
        return _from_node(node)

    def agg(
        self,
        aggs: TryIter[IntoExpr] = None,
        *more_aggs: IntoExpr,
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
        node = nodes.Agg(self._inner, aggs, more_aggs, named_aggs)
        return _from_node(node)


def _from_node(scan: nodes.Agg | nodes.AggColumns) -> LazyFrame:
    out = LazyFrame.__new__(LazyFrame)
    out._inner = scan  # pyright: ignore[reportPrivateUsage]
    return out
