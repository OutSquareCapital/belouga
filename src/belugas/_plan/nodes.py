from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from ..typing import FileGlob, PathOrBuffer

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
    from pyochain import Option, Seq
    from rich.console import RenderableType

    from .._expr import Expr
    from ..datatypes import DataType
    from ..typing import (
        AsofJoinStrategy,
        CSVOptions,
        GroupByClause,
        IntoExpr,
        IntoExprColumn,
        IntoRel,
        JoinStrategy,
        JsonOptions,
        Orientation,
        ParquetOptions,
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


@dataclass(slots=True, repr=False)
class BaseNode:
    """Base class for all plan nodes."""

    @override
    def __repr__(self) -> str:
        from .._show import node_structure

        return node_structure(self)

    def __rich__(self) -> RenderableType:  # noqa: PLW3201
        from .._show import node_tree

        return node_tree(self)


@dataclass(slots=True, repr=False)
class BaseScan(BaseNode):
    """Base class for all scan nodes."""

    connection: Option[DuckDBPyConnection]


@dataclass(slots=True, repr=False)
class ScanInMemory[T: IntoRel](BaseScan):
    data: T | None
    orient: Orientation = "col"


@dataclass(slots=True, repr=False)
class ScanTable(BaseScan):
    table: str


@dataclass(slots=True, repr=False)
class ScanTableFunction(BaseScan):
    function: str


@dataclass(slots=True, repr=False)
class _ScanFile[F: FileGlob | PathOrBuffer](BaseScan):
    path: F


@dataclass(slots=True, repr=False)
class ScanParquet(_ScanFile[FileGlob]):
    options: ParquetOptions


@dataclass(slots=True, repr=False)
class ScanCSV(_ScanFile[PathOrBuffer]):
    options: CSVOptions


@dataclass(slots=True, repr=False)
class ScanJson(_ScanFile[PathOrBuffer]):
    options: JsonOptions


@dataclass(slots=True, repr=False)
class LogicalNode(BaseNode):
    inner: Node


@dataclass(slots=True, repr=False)
class _Expressions(LogicalNode):
    exprs: TryIter[IntoExpr]
    more_exprs: Iterable[IntoExpr]
    named: dict[str, IntoExpr]


_LogicalNode = LogicalNode


class Select(_Expressions):
    """Node representing a select operation."""


@dataclass(slots=True, repr=False)
class SelectAll(LogicalNode):
    func: ExprFn


@dataclass(slots=True, repr=False)
class WithColumns(_Expressions):
    """Node representing a with_columns operation."""


@dataclass(slots=True, repr=False)
class Filter(LogicalNode):
    predicates: TryIter[IntoExprColumn]
    more_predicates: Iterable[IntoExprColumn]
    constraints: dict[str, IntoExpr]


@dataclass(slots=True, repr=False)
class GroupBy(LogicalNode):
    """Node representing a group_by operation."""

    keys: Seq[Expr]
    strategy: GroupByClause | None
    drop_null_keys: bool


@dataclass(slots=True, repr=False)
class Agg(_Expressions):
    """Node representing an aggregation operation."""


@dataclass(slots=True, repr=False)
class AggColumns(LogicalNode):
    """Node representing an aggregation operation that applies the same function to all columns."""

    func: ExprFn


@dataclass(slots=True, repr=False)
class GroupByAll(_Expressions):
    """Node representing a group_by_all operation."""


@dataclass(slots=True, repr=False)
class Sort(LogicalNode):
    by: TryIter[IntoExpr]
    more_by: Iterable[IntoExpr]
    descending: TrySeq[bool]
    nulls_last: TrySeq[bool]


@dataclass(slots=True, repr=False)
class Limit(LogicalNode):
    n: int


@dataclass(slots=True, repr=False)
class Slice(LogicalNode):
    length: Option[int]
    offset: int


@dataclass(slots=True, repr=False)
class DropRows(LogicalNode):
    subset: TryIter[str]
    fn: ExprFn


@dataclass(slots=True, repr=False)
class Drop(LogicalNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True, repr=False)
class Explode(LogicalNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True, repr=False)
class Unnest(LogicalNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True, repr=False)
class Union(LogicalNode):
    other: Node


@dataclass(slots=True, repr=False)
class Join(LogicalNode):
    other: Node
    on: TryIter[str]
    how: JoinStrategy
    left_on: TryIter[str]
    right_on: TryIter[str]
    suffix: str


@dataclass(slots=True, repr=False)
class JoinCross(LogicalNode):
    """Node representing a cross join operation."""

    other: Node
    suffix: str


@dataclass(slots=True, repr=False)
class JoinAsof(LogicalNode):
    other: Node
    left_on: Option[str]
    right_on: Option[str]
    on: Option[str]
    by_left: TryIter[str]
    by_right: TryIter[str]
    by: TryIter[str]
    strategy: AsofJoinStrategy
    suffix: str


@dataclass(slots=True, repr=False)
class Unique(LogicalNode):
    subset: TryIter[str]
    keep: UniqueKeepStrategy
    order_by: TrySeq[str]


@dataclass(slots=True, repr=False)
class Pivot(LogicalNode):
    on: TryIter[str]
    on_columns: Sequence[PythonLiteral]
    index: TryIter[str]
    values: TryIter[str]
    aggregate_function: PivotAgg
    maintain_order: bool
    separator: str


@dataclass(slots=True, repr=False)
class Unpivot(LogicalNode):
    on: TryIter[str]
    index: TryIter[str]
    variable_name: str
    value_name: str
    order_by: TryIter[str]


@dataclass(slots=True, repr=False)
class WithRowIndex(LogicalNode):
    name: str
    order_by: TryIter[str]


@dataclass(slots=True, repr=False)
class Cast(LogicalNode):
    dtypes: Mapping[str, DataType] | DataType


@dataclass(slots=True, repr=False)
class Rename(LogicalNode):
    mapping: Mapping[str, str]


# plan scan marker START
type Scan = (
    ScanCSV
    | ScanInMemory[IntoRel]
    | ScanJson
    | ScanParquet
    | ScanTable
    | ScanTableFunction
)
"""All nodes that represent logical scan sources.

The `Scan` union is generated from all public `BaseNode` subclasses with names starting with `Scan`.


Dynamically generated by `scripts/__main__.py`.

Do NOT edit manually.
"""
# plan scan marker END
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
    | BaseScan
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
