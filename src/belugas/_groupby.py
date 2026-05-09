from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, final

from pyochain import NONE, Dict, Iter, Option, Seq, Set, Some
from sqlglot import exp

from ._expr import Expr
from ._funcs import col, len
from ._plan import ExprPlan

if TYPE_CHECKING:
    from ._frame import LazyFrame
    from .typing import GroupByClause, IntoExpr
    from .utils import TryIter


def _root_column_name(expr: Expr) -> Option[str]:
    match expr.inner:
        case exp.Column() as column:
            return Some(column.output_name)
        case _:
            return NONE


@final
class LazyGroupBy:
    __slots__ = ("_frame", "_keys", "_schema", "_strategy")

    def __init__(
        self, frame: LazyFrame, keys: Seq[Expr], strategy: GroupByClause | None
    ) -> None:
        self._frame = frame
        self._keys = keys
        self._strategy: GroupByClause | None = strategy
        keys_names = keys.iter().filter_map(_root_column_name).collect(Set)
        self._schema = (
            frame._schema  # pyright: ignore[reportPrivateUsage]
            .items()
            .iter()
            .filter_star(lambda name, _dt: name not in keys_names)
            .collect(Dict)
        )

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
        return (
            self._schema
            .iter()
            .map(lambda name: col(name).pipe(func).alias(name))
            .into(self.agg)
        )

    def agg(
        self,
        aggs: TryIter[IntoExpr] = None,
        *more_aggs: IntoExpr,
        **named_aggs: IntoExpr,
    ) -> LazyFrame:
        key_glots = self._keys.iter().map(lambda c: c.inner).collect(list)

        def _group_by_clause() -> Iterable[exp.Expr]:
            match self._strategy:
                case "CUBE":
                    return Iter.once(exp.Cube(expressions=key_glots))
                case "ROLLUP":
                    return Iter.once(exp.Rollup(expressions=key_glots))
                case None:
                    return key_glots

        plan = self._schema.into(ExprPlan, aggs, more_aggs, named_aggs)
        key_schema = (
            self._keys
            .iter()
            .filter_map(_root_column_name)
            .map(lambda name: (name, self._frame._schema.get_item(name).unwrap()))  # pyright: ignore[reportPrivateUsage]
            .collect(Dict)
        )
        return (
            plan
            .agg_ctx(Iter(key_glots))
            .group_by(*_group_by_clause())
            .pipe(
                self._frame._from_ast,  # pyright: ignore[reportPrivateUsage]
                schema=plan.agg_schema(key_schema),
                src=self._frame,
            )
        )
