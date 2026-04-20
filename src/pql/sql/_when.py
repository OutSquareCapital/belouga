from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlglot import exp

from ._conversions import pql_into_glot
from ._expr import SqlExpr
from ._funcs import reduce
from .utils import try_iter

if TYPE_CHECKING:
    from .typing import IntoExpr
    from .utils import TryIter


def when(predicates: TryIter[IntoExpr], *more_predicates: IntoExpr) -> When:
    return When(_into_pred(predicates, more_predicates))


def _into_pred(preds: TryIter[IntoExpr], more_preds: Iterable[IntoExpr]) -> SqlExpr:
    return try_iter(preds).chain(more_preds).into(reduce, function=SqlExpr.and_)


@dataclass(slots=True)
class When:
    _when: SqlExpr

    def then(self, value: IntoExpr) -> Then:
        """Attach the value for the initial WHEN condition.

        Returns:
            Then: An object that allows chaining additional WHEN conditions or specifying an OTHERWISE clause.
        """
        return Then(
            exp.Case(ifs=[exp.If(this=self._when.inner, true=pql_into_glot(value))])
        )


@dataclass(slots=True)
class Then(SqlExpr):
    def when(
        self, predicates: TryIter[IntoExpr], *more_predicates: IntoExpr
    ) -> ChainedWhen:
        return ChainedWhen(self, _into_pred(predicates, more_predicates))

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        case = self.inner.copy()
        case.set("default", pql_into_glot(statement))
        return SqlExpr(case)


@dataclass(slots=True)
class ChainedWhen:
    _chained_when: SqlExpr
    _predicate: SqlExpr

    def then(self, statement: IntoExpr) -> ChainedThen:
        case = self._chained_when.inner.copy()
        if_expr = exp.If(this=self._predicate.inner, true=pql_into_glot(statement))
        case.append("ifs", if_expr)
        return ChainedThen(case)


@dataclass(slots=True)
class ChainedThen(SqlExpr):
    def when(
        self, predicates: TryIter[IntoExpr], *more_predicates: IntoExpr
    ) -> ChainedWhen:
        return ChainedWhen(self, _into_pred(predicates, more_predicates))

    def otherwise(self, statement: IntoExpr) -> SqlExpr:
        case = self.inner.copy()
        case.set("default", pql_into_glot(statement))
        return SqlExpr(case)
