from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyochain import Option, Seq, Vec

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
type Plan = Vec[Node]
"""Logical plan represented as a sequence of `Node`s."""


@dataclass(slots=True)
class BaseNode:
    """Base class for all plan nodes."""


@dataclass(slots=True)
class _Expressions(BaseNode):
    exprs: TryIter[IntoExpr]
    more_exprs: Iterable[IntoExpr]
    named: dict[str, IntoExpr]


@dataclass(slots=True)
class Scan(BaseNode):
    """Node representing a scan operation."""


@dataclass(slots=True)
class Select(_Expressions):
    """Node representing a select operation."""


@dataclass(slots=True)
class SelectAll(BaseNode):
    func: ExprFn


@dataclass(slots=True)
class WithColumns(_Expressions):
    """Node representing a with_columns operation."""


@dataclass(slots=True)
class Filter(BaseNode):
    predicates: TryIter[IntoExprColumn]
    more_predicates: Iterable[IntoExprColumn]
    constraints: dict[str, IntoExpr]


@dataclass
class _GroupByBase(BaseNode):
    keys: Seq[Expr]
    strategy: GroupByClause | None
    drop_null_keys: bool


@dataclass(slots=True)
class GroupBy(_GroupByBase):
    """Node representing a group_by operation."""


@dataclass(slots=True)
class Agg(_GroupByBase, _Expressions):
    """Node representing an aggregation operation."""


@dataclass(slots=True)
class AggColumns(_GroupByBase):
    """Node representing an aggregation operation that applies the same function to all columns."""

    func: ExprFn


@dataclass(slots=True)
class GroupByAll(_Expressions):
    """Node representing a group_by_all operation."""


@dataclass(slots=True)
class Sort(BaseNode):
    by: TryIter[IntoExpr]
    more_by: Iterable[IntoExpr]
    descending: TrySeq[bool]
    nulls_last: TrySeq[bool]


@dataclass(slots=True)
class Limit(BaseNode):
    n: int


@dataclass(slots=True)
class Slice(BaseNode):
    length: Option[int]
    offset: int


@dataclass(slots=True)
class DropRows(BaseNode):
    subset: TryIter[str]
    fn: ExprFn


@dataclass(slots=True)
class Drop(BaseNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True)
class Explode(BaseNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True)
class Unnest(BaseNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True)
class Union(BaseNode):
    other: LazyFrame


@dataclass(slots=True)
class Join(BaseNode):
    other: LazyFrame
    on: TryIter[str]
    how: JoinStrategy
    left_on: TryIter[str]
    right_on: TryIter[str]
    suffix: str


@dataclass(slots=True)
class JoinCross(BaseNode):
    """Node representing a cross join operation."""

    other: LazyFrame
    suffix: str


@dataclass(slots=True)
class JoinAsof(BaseNode):
    other: LazyFrame
    left_on: Option[str]
    right_on: Option[str]
    on: Option[str]
    by_left: TryIter[str]
    by_right: TryIter[str]
    by: TryIter[str]
    strategy: AsofJoinStrategy
    suffix: str


@dataclass(slots=True)
class Unique(BaseNode):
    subset: TryIter[str]
    keep: UniqueKeepStrategy
    order_by: TrySeq[str]


@dataclass(slots=True)
class Pivot(BaseNode):
    on: TryIter[str]
    on_columns: Sequence[PythonLiteral]
    index: TryIter[str]
    values: TryIter[str]
    aggregate_function: PivotAgg
    maintain_order: bool
    separator: str


@dataclass(slots=True)
class Unpivot(BaseNode):
    on: TryIter[str]
    index: TryIter[str]
    variable_name: str
    value_name: str
    order_by: TryIter[str]


@dataclass(slots=True)
class WithRowIndex(BaseNode):
    name: str
    order_by: TryIter[str]


@dataclass(slots=True)
class Cast(BaseNode):
    dtypes: Mapping[str, DataType] | DataType


@dataclass(slots=True)
class Rename(BaseNode):
    mapping: Mapping[str, str]


# plan node marker START
Node = (
    Agg
    | AggColumns
    | Cast
    | Drop
    | DropRows
    | Explode
    | Filter
    | GroupBy
    | GroupByAll
    | Join
    | JoinAsof
    | JoinCross
    | Limit
    | Pivot
    | Rename
    | Scan
    | Select
    | SelectAll
    | Slice
    | Sort
    | Union
    | Unique
    | Unnest
    | Unpivot
    | WithColumns
    | WithRowIndex
)
"""All nodes that can be part of the logical plan.
Each node represents a logical operation to be performed on the data, such as filtering, joining, or aggregating.

They just hold the input arguments for the operation, and are used to build the logical plan before it's compiled into executable code.

Note:
    - `Node` is a type alias for the union of all public `Node` subclasses.
    - `BaseNode` is the base class for all nodes, but should not be used directly.

    We do this because Python does not have built-in support for Enums with associated data like Rust does.

    Using dataclasses for each node allows us to easily store the necessary information for each operation,

    while still being able to treat them as a unified type when building the plan, relying on the type checker to ensure exhaustiveness when matching on them.


Dynamically generated by `scripts/__main__.py`.

Do NOT edit manually.
"""
# plan node marker END
