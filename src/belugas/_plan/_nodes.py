from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyochain import Dict, Option, Seq

    from belugas.utils import TryIter, TrySeq

    from .._expr import Expr
    from .._frame import LazyFrame
    from ..typing import (
        AsofJoinStrategy,
        IntoExpr,
        IntoExprColumn,
        JoinStrategy,
        PivotAgg,
        PythonLiteral,
        UniqueKeepStrategy,
    )


@dataclass(slots=True)
class Node:
    """Base class for all plan nodes."""


@dataclass(slots=True)
class Select(Node):
    exprs: Seq[IntoExpr]
    named: Dict[str, IntoExpr]


@dataclass(slots=True)
class IterSlct(Node):
    func: Callable[[Expr], Expr]


@dataclass(slots=True)
class WithColumns(Node):
    exprs: Seq[IntoExpr]
    named: Dict[str, IntoExpr]


@dataclass(slots=True)
class Filter(Node):
    predicates: Seq[IntoExprColumn]
    constraints: Dict[str, IntoExpr]


@dataclass(slots=True)
class GroupByAll(Node):
    exprs: Seq[IntoExpr]
    named: Dict[str, IntoExpr]


@dataclass(slots=True)
class Sort(Node):
    by: Seq[IntoExpr]
    descending: TrySeq[bool]
    nulls_last: TrySeq[bool]


@dataclass(slots=True)
class Limit(Node):
    n: int


@dataclass(slots=True)
class Slice(Node):
    offset: int
    length: Option[int]


@dataclass(slots=True)
class Drop(Node):
    columns: Seq[IntoExprColumn]


@dataclass(slots=True)
class DropRows(Node):
    subset: TryIter[str]
    fn: Callable[[Expr], Expr]


@dataclass(slots=True)
class Explode(Node):
    columns: Seq[IntoExprColumn]


@dataclass(slots=True)
class Union(Node):
    other: LazyFrame


@dataclass(slots=True)
class Unnest(Node):
    columns: Seq[IntoExprColumn]


@dataclass(slots=True)
class Rename(Node):
    mapping: Dict[str, str]


@dataclass(slots=True)
class Join(Node):
    other: LazyFrame
    on: TrySeq[str]
    how: JoinStrategy
    left_on: TrySeq[str]
    right_on: TrySeq[str]
    suffix: str


@dataclass(slots=True)
class JoinCross(Node):
    other: LazyFrame
    suffix: str


@dataclass(slots=True)
class JoinAsof(Node):
    other: LazyFrame
    left_on: Option[str]
    right_on: Option[str]
    on: Option[str]
    by_left: TrySeq[str]
    by_right: TrySeq[str]
    by: TrySeq[str]
    strategy: AsofJoinStrategy
    suffix: str


@dataclass(slots=True)
class Unique(Node):
    subset: TrySeq[str]
    keep: UniqueKeepStrategy
    order_by: TrySeq[str]


@dataclass(slots=True)
class Pivot(Node):
    on: TryIter[str]
    on_columns: Sequence[PythonLiteral]
    index: TryIter[str]
    values: TryIter[str]
    aggregate_function: PivotAgg
    maintain_order: bool
    separator: str


@dataclass(slots=True)
class Unpivot(Node):
    on: TryIter[str]
    index: TryIter[str]
    variable_name: str
    value_name: str
    order_by: TryIter[str]


@dataclass(slots=True)
class WithRowIndex(Node):
    name: str
    order_by: TryIter[str]


type PlanNode = (
    Select
    | WithColumns
    | Filter
    | Sort
    | Limit
    | Slice
    | Drop
    | DropRows
    | Explode
    | Unnest
    | Rename
    | GroupByAll
    | Join
    | JoinCross
    | JoinAsof
    | Unique
    | Pivot
    | Unpivot
    | WithRowIndex
    | IterSlct
    | Union
)
