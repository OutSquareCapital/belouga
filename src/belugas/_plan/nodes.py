from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, override

from pyochain import NONE, Seq

from ..typing import FileGlob, PathOrBuffer

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection
    from pyochain import Option
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
        return node_structure(self)

    def __rich__(self) -> RenderableType:  # noqa: PLW3201
        from .._show import node_tree

        return node_tree(self)


def node_structure(node: object, level: int = 0) -> str:

    indent = "\n" + ("  " * (level + 1))
    delim = f",{indent}"

    def _is_nested_node(node: BaseNode, name: str) -> bool:
        value: object = getattr(node, name)  # pyright: ignore[reportAny]
        return isinstance(value, BaseNode)

    match node:
        case BaseNode():
            node_fields = Seq(fields(node))
            is_leaf = node_fields.all(
                lambda field: not _is_nested_node(node, field.name)
            )

            if is_leaf:
                indent = ""
                delim = ", "

            items = (
                node_fields
                .iter()
                .filter(lambda field: field.name != "inner")
                .chain(node_fields.iter().filter(lambda field: field.name == "inner"))
                .map(lambda field: _field_to_s(node, field.name, level + 1))
                .join(delim)
            )
            return f"{node.__class__.__name__}({indent}{items})"
        case _:
            return repr(node)


def _field_to_s(node: BaseNode, name: str, level: int) -> str:
    value: object = getattr(node, name)  # pyright: ignore[reportAny]
    return f"{name}={node_structure(value, level)}"


@dataclass(slots=True, repr=False)
class BaseScan(BaseNode):
    """Base class for all scan nodes."""

    connection: Option[DuckDBPyConnection]


@dataclass(slots=True, repr=False)
class ScanInMemory[T: IntoRel](BaseScan):
    data: T | None
    orient: Orientation = "col"
    projected_columns: Option[Seq[str]] = field(default_factory=lambda: NONE)


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
class _UnaryNode(LogicalNode):
    """Base class for nodes that transform a single input relation."""


@dataclass(slots=True, repr=False)
class DeferredNode(_UnaryNode):
    """Unary nodes that can be accumulated before the scan is resolved."""


@dataclass(slots=True, repr=False)
class _DescriptorNode(_UnaryNode):
    """Unary nodes that describe context and are consumed by another node."""


@dataclass(slots=True, repr=False)
class _AggregateNode(_UnaryNode):
    """Unary nodes that consume grouping context and change row grain."""


@dataclass(slots=True, repr=False)
class _BoundaryNode(_UnaryNode):
    """Unary nodes that force a relational boundary when compiled."""


@dataclass(slots=True, repr=False)
class _ConditionalBoundaryNode(_UnaryNode):
    """Unary nodes whose compilation strategy depends on their arguments."""


@dataclass(slots=True, repr=False)
class _BinaryNode(LogicalNode):
    """Base class for nodes that compose two child relations."""

    other: Node


@dataclass(repr=False)
class _Expressions:
    exprs: TryIter[IntoExpr]
    more_exprs: Iterable[IntoExpr]
    named: dict[str, IntoExpr]


@dataclass(slots=True, repr=False)
class _DeferredExpressions(_Expressions, DeferredNode):
    pass


@dataclass(slots=True, repr=False)
class _AggregateExpressions(_Expressions, _AggregateNode):
    pass


@dataclass(slots=True, repr=False)
class _BoundaryExpressions(_Expressions, _BoundaryNode):
    pass


_LogicalNode = LogicalNode


class Select(_DeferredExpressions):
    """Node representing a select operation."""


@dataclass(slots=True, repr=False)
class SelectAll(DeferredNode):
    func: ExprFn


@dataclass(slots=True, repr=False)
class WithColumns(_DeferredExpressions):
    """Node representing a with_columns operation."""


@dataclass(slots=True, repr=False)
class Filter(DeferredNode):
    predicates: TryIter[IntoExprColumn]
    more_predicates: Iterable[IntoExprColumn]
    constraints: dict[str, IntoExpr]


@dataclass(slots=True, repr=False)
class GroupBy(_DescriptorNode):
    """Node representing a group_by operation."""

    keys: Seq[Expr]
    strategy: GroupByClause | None
    drop_null_keys: bool


@dataclass(slots=True, repr=False)
class Agg(_AggregateExpressions):
    """Node representing an aggregation operation."""


@dataclass(slots=True, repr=False)
class AggColumns(_AggregateNode):
    """Node representing an aggregation operation that applies the same function to all columns."""

    func: ExprFn


@dataclass(slots=True, repr=False)
class GroupByAll(_BoundaryExpressions):
    """Node representing a group_by_all operation."""


@dataclass(slots=True, repr=False)
class Sort(DeferredNode):
    by: TryIter[IntoExpr]
    more_by: Iterable[IntoExpr]
    descending: TrySeq[bool]
    nulls_last: TrySeq[bool]


@dataclass(slots=True, repr=False)
class Limit(DeferredNode):
    n: int


@dataclass(slots=True, repr=False)
class Slice(_ConditionalBoundaryNode):
    length: Option[int]
    offset: int


@dataclass(slots=True, repr=False)
class DropRows(DeferredNode):
    subset: TryIter[str]
    fn: ExprFn


@dataclass(slots=True, repr=False)
class Drop(DeferredNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True, repr=False)
class Explode(_BoundaryNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True, repr=False)
class Unnest(_BoundaryNode):
    columns: TryIter[IntoExprColumn]
    more_columns: Iterable[IntoExprColumn]


@dataclass(slots=True, repr=False)
class Union(_BinaryNode):
    pass


@dataclass(slots=True, repr=False)
class Join(_BinaryNode):
    on: TryIter[str]
    how: JoinStrategy
    left_on: TryIter[str]
    right_on: TryIter[str]
    suffix: str


@dataclass(slots=True, repr=False)
class JoinCross(_BinaryNode):
    """Node representing a cross join operation."""

    suffix: str


@dataclass(slots=True, repr=False)
class JoinAsof(_BinaryNode):
    left_on: Option[str]
    right_on: Option[str]
    on: Option[str]
    by_left: TryIter[str]
    by_right: TryIter[str]
    by: TryIter[str]
    strategy: AsofJoinStrategy
    suffix: str


@dataclass(slots=True, repr=False)
class Unique(_ConditionalBoundaryNode):
    subset: TryIter[str]
    keep: UniqueKeepStrategy
    order_by: TrySeq[str]


@dataclass(slots=True, repr=False)
class Pivot(_BoundaryNode):
    on: TryIter[str]
    on_columns: Sequence[PythonLiteral]
    index: TryIter[str]
    values: TryIter[str]
    aggregate_function: PivotAgg
    maintain_order: bool
    separator: str


@dataclass(slots=True, repr=False)
class Unpivot(_BoundaryNode):
    on: TryIter[str]
    index: TryIter[str]
    variable_name: str
    value_name: str
    order_by: TryIter[str]


@dataclass(slots=True, repr=False)
class WithRowIndex(DeferredNode):
    name: str
    order_by: TryIter[str]


@dataclass(slots=True, repr=False)
class Cast(DeferredNode):
    dtypes: Mapping[str, DataType] | DataType


@dataclass(slots=True, repr=False)
class Rename(DeferredNode):
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
