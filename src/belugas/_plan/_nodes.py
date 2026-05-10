from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyochain import Option, Seq

    from .._expr import Expr
    from .._frame import LazyFrame
    from ..datatypes import DataType
    from ..typing import (
        AsofJoinStrategy,
        GroupByClause,
        IntoExpr,
        IntoExprColumn,
        JoinStrategy,
        PivotAgg,
        PythonLiteral,
        TryIter,
        TrySeq,
        UniqueKeepStrategy,
    )

type ExprFn = Callable[[Expr], Expr]
"""Function type for operations that take a single `Expr` and return anoter.

Useful to defer method calls on `Expr` until the plan is being compiled.
"""


@dataclass(slots=True)
class Node:
    """Base class for all plan nodes."""


@dataclass(slots=True)
class _Expressions:
    exprs: TryIter[IntoExpr]
    more_exprs: Iterable[IntoExpr]
    named: dict[str, IntoExpr]


@dataclass(slots=True)
class Select(_Expressions, Node):
    """Node representing a select operation."""


@dataclass(slots=True)
class SelectAll(Node):
    func: ExprFn


@dataclass(slots=True)
class WithColumns(_Expressions, Node):
    """Node representing a with_columns operation."""


@dataclass(slots=True)
class Filter(Node):
    predicates: TryIter[IntoExprColumn]
    more_predicates: Iterable[IntoExprColumn]
    constraints: dict[str, IntoExpr]


@dataclass(slots=True)
class GroupBy(Node):
    keys: Seq[Expr]
    strategy: GroupByClause | None
    drop_null_keys: bool


# NOTE: should we herit from GroupBy here?
@dataclass(slots=True)
class Agg(_Expressions, GroupBy):
    """Node representing an aggregation operation."""


@dataclass(slots=True)
class AggColumns(Node):
    keys: Seq[Expr]
    func: ExprFn
    drop_null_keys: bool


@dataclass(slots=True)
class GroupByAll(_Expressions, Node):
    """Node representing a group_by_all operation."""


@dataclass(slots=True)
class Sort(Node):
    by: TryIter[IntoExpr]
    more_by: Iterable[IntoExpr]
    descending: TrySeq[bool]
    nulls_last: TrySeq[bool]


@dataclass(slots=True)
class Limit(Node):
    n: int


@dataclass(slots=True)
class Slice(Node):
    length: Option[int]
    offset: int


@dataclass(slots=True)
class DropRows(Node):
    subset: TryIter[str]
    fn: ExprFn


@dataclass(slots=True)
class Drop(Node):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True)
class Explode(Node):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True)
class Unnest(Node):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True)
class Union(Node):
    other: LazyFrame


@dataclass(slots=True)
class _JoinBase(Node):
    other: LazyFrame
    suffix: str


@dataclass(slots=True)
class Join(_JoinBase):
    on: TryIter[str]
    how: JoinStrategy
    left_on: TryIter[str]
    right_on: TryIter[str]


@dataclass(slots=True)
class JoinCross(_JoinBase):
    """Node representing a cross join operation."""


@dataclass(slots=True)
class JoinAsof(Node):
    left_on: Option[str]
    right_on: Option[str]
    on: Option[str]
    by_left: TryIter[str]
    by_right: TryIter[str]
    by: TryIter[str]
    strategy: AsofJoinStrategy


@dataclass(slots=True)
class Unique(Node):
    subset: TryIter[str]
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


@dataclass(slots=True)
class Cast(Node):
    dtypes: Mapping[str, DataType] | DataType


@dataclass(slots=True)
class Rename(Node):
    mapping: Mapping[str, str]


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
    | SelectAll
    | Union
)
