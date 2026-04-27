"""Typing definitions for the SQL module."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, runtime_checkable

from pyochain import Result, Seq
from sqlglot import exp

if TYPE_CHECKING:
    from _duckdb._typing import (  # pyright: ignore[reportMissingModuleSource]
        BlobLiteral as DuckBlobLit,
        IntoExpr as DuckIntoExpr,
        IntoExprColumn as DuckIntoExprColumn,
        NestedLiteral as DuckNestedLit,
        NonNestedLiteral as DuckNonNestedLit,
        ParquetCompression as DuckParquetCompression,
        PythonLiteral as DuckPyLit,
        PyTypeIds as DuckPyTypeIds,
        StrIntoPyType as DuckStrIntoPyType,
    )
    from narwhals.typing import IntoFrame

    from ._core import DuckHandler
    from ._expr import Expr
    from ._joins import JoinKeys
    from ._scans import ScanSource
    from .datatypes import DataType


@runtime_checkable
class FrameLike(Protocol):
    """Credits to `narwhals` for the Protocols definitions."""

    @property
    def columns(self) -> Any: ...  # noqa: ANN401, D102  # pyright: ignore[reportAny, reportExplicitAny]
    def join(self, *args: Any, **kwargs: Any) -> Any: ...  # pyright: ignore[reportExplicitAny, reportAny]  # noqa: ANN401, D102


class NPProtocol(Protocol):
    """Base Protocol for numpy objects."""

    @property
    def dtype(self) -> Any: ...  # noqa: ANN401, D102  # pyright: ignore[reportExplicitAny, reportAny]
    @property
    def ndim(self) -> int: ...  # noqa: D102
    def __array__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105, PLW3201  # pyright: ignore[reportExplicitAny, reportAny]
    def __array_wrap__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105, PLW3201  # pyright: ignore[reportExplicitAny, reportAny]
    @property
    def __array_interface__(self) -> dict[str, Any]: ...  # noqa: D105, PLW3201  # pyright: ignore[reportExplicitAny]
    @property
    def __array_priority__(self) -> float: ...  # noqa: D105, PLW3201


class NPScalarTypeLike(NPProtocol, Protocol):  # noqa: D101
    @property
    def itemsize(self) -> int: ...  # noqa: D102


@runtime_checkable
class NPArrayLike[S: tuple[Any, ...], D](NPProtocol, Protocol):
    """Protocol for `numpy` ndarrays."""

    def __len__(self) -> int: ...  # noqa: D105
    def __contains__(self, value: object, /) -> bool: ...  # noqa: D105
    def __iter__(self) -> Iterator[D]: ...  # noqa: D105
    def __array_finalize__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401, D105, PLW3201  # pyright: ignore[reportExplicitAny, reportAny]
    def __getitem__(self, *args: Any, **kwargs: Any) -> Any: ...  # noqa: ANN401, D105  # pyright: ignore[reportExplicitAny, reportAny]
    def __setitem__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401, D105  # pyright: ignore[reportExplicitAny, reportAny]
    @property
    def shape(self) -> S: ...  # noqa: D102
    @property
    def size(self) -> int: ...  # noqa: D102
    @property
    def T(self) -> Self: ...  # noqa: D102, N802


type AnyArray = NPArrayLike[Any, Any]  # pyright: ignore[reportExplicitAny]


type IntoDict[K, V] = Mapping[K, V] | Iterable[tuple[K, V]]
type ExprLike = Expr | DuckHandler
"""Types that are already expressions wrappers and can be used directly as expressions."""
type BlobLiteral = DuckBlobLit
type NonNestedLiteral = DuckNonNestedLit
type SeqLiteral[T: NonNestedLiteral] = list[T] | tuple[T, ...]
"""Sequence of non-nested literals of the same type."""
type PythonLiteral = DuckPyLit
type NestedLiteral = DuckNestedLit
"""Python literal types (can convert into a `lit` expression)."""
type LitSeq = Sequence[PythonLiteral]
type NestedSeq = Sequence[LitSeq]
type SeqIntoVals = Sequence[Mapping[str, PythonLiteral]] | NestedSeq | LitSeq | AnyArray

type IntoValues = Mapping[str, LitSeq] | SeqIntoVals
"""Types that can be converted into a `values` relation (either an expression, a mapping, or a sequence)."""
type IntoRel = IntoFrame | IntoValues | ScanSource
""""Types that can be converted into a relation (either a frame or values)."""
type IntoExprColumn = str | ExprLike
"""Inputs that can convert into a `col` expression."""
type IntoExpr = PythonLiteral | IntoExprColumn | exp.Expr
"""Inputs that can convert into an expression (either a `lit` or a `col`)."""
type IntoDuckExpr = DuckIntoExpr
type IntoDuckExprCol = DuckIntoExprColumn
type DTypeIds = DuckPyTypeIds
type StrIntoDType = DuckStrIntoPyType
# TODO: add this to parse_{dirname, dirpath, filename, path} fns arg
type Separator = Literal["system", "both_slash", "forward_slash", "backslash"]
# TODO: add this to date_{trunc, part, diff, sub} fns with a part arg
type IntervalPart = Literal[
    "century",
    "day",
    "decade",
    "hour",
    "microseconds",
    "millenium",
    "milliseconds",
    "minute",
    "month",
    "quarter",
    "second",
    "year",
]
type DatePart = Literal[
    "dayofweek",
    "dayofyear",
    "epoch",
    "era",
    "isodow",
    "isoyear",
    "julian",
    "timezone_hour",
    "timezone_minute",
    "timezone",
    "week",
    "yearweek",
]
type AllDateParts = IntervalPart | DatePart

RoundMode = Literal["half_to_even", "half_away_from_zero"]
type ParquetCompression = DuckParquetCompression
type Orientation = Literal["row", "col"]
type FrameMode = Literal["ROWS", "RANGE", "GROUPS"]
type WindowExclude = Literal["CURRENT ROW", "GROUP", "TIES", "NO OTHERS"]
ClosedInterval = Literal["both", "left", "right", "none"]

TimeUnit = Literal["ms", "us", "ns"]
EpochTimeUnit = Literal["ms", "us", "ns", "s", "d"]
FillNullStrategy = Literal["forward", "backward", "min", "max", "mean", "zero", "one"]
RankMethod = Literal["average", "min", "max", "dense", "ordinal"]
type IntoDataType = exp.DataType | DataType
"""Types that can be converted into a `DataType` instance."""
type TransferEncoding = Literal["hex", "base64"]
JoinStrategy = Literal["inner", "left", "right", "outer", "semi", "anti"]
AsofJoinStrategy = Literal["backward", "forward"]
UniqueKeepStrategy = Literal["any", "none", "first", "last"]
PivotAgg = Literal[
    "min", "max", "first", "last", "sum", "mean", "median", "len", "count"
]
type JoinKeysRes[T: Seq[str] | str] = Result[JoinKeys[T], ValueError]
type GroupByClause = Literal["ROLLUP", "CUBE"]

# theme marker START
Themes = Literal[
    "abap",
    "algol",
    "algol_nu",
    "arduino",
    "autumn",
    "bw",
    "borland",
    "coffee",
    "colorful",
    "default",
    "dracula",
    "emacs",
    "friendly_grayscale",
    "friendly",
    "fruity",
    "github-dark",
    "gruvbox-dark",
    "gruvbox-light",
    "igor",
    "inkpot",
    "lightbulb",
    "lilypond",
    "lovelace",
    "manni",
    "material",
    "monokai",
    "murphy",
    "native",
    "nord-darker",
    "nord",
    "one-dark",
    "paraiso-dark",
    "paraiso-light",
    "pastie",
    "perldoc",
    "rainbow_dash",
    "rrt",
    "sas",
    "solarized-dark",
    "solarized-light",
    "staroffice",
    "stata-dark",
    "stata-light",
    "tango",
    "trac",
    "vim",
    "vs",
    "xcode",
    "zenburn",
]
"""Themes available for SQL syntax highlighting in the `sql_query` method.

Dynamically generated from the available styles in the `pygments` library by `scripts/__main__.py`.

Do NOT edit manually."""
# theme marker END
