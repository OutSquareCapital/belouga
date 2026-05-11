from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyochain import Option, Seq

from ._expr import Expr
from ._funcs import len
from ._plan import nodes

if TYPE_CHECKING:
    from ._frame import LazyFrame
    from .typing import GroupByClause, IntoExpr, TryIter


@dataclass(slots=True)
class LazyGroupBy:
    _frame: LazyFrame
    _keys: Seq[Expr]
    _strategy: GroupByClause | None
    _drop_null_keys: bool

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
        node = nodes.AggColumns(
            self._frame.inner,
            self._keys,
            self._strategy,
            drop_null_keys=self._drop_null_keys,
            func=func,
        )
        return self._frame._cls(node)  # pyright: ignore[reportPrivateUsage]

    def agg(
        self,
        aggs: TryIter[IntoExpr] = None,
        *more_aggs: IntoExpr,
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
        node = nodes.Agg(
            self._frame.inner,
            aggs,
            more_aggs,
            named_aggs,
            self._keys,
            self._strategy,
            drop_null_keys=self._drop_null_keys,
        )
        return self._frame._cls(node)  # pyright: ignore[reportPrivateUsage]
