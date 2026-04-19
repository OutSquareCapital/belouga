from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .sql import ScanSource

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame
    from sqlglot import exp

    from ._frame import LazyFrame
    from .sql.typing import (
        AnyArray,
        IntoDict,
        IntoRel,
        Orientation,
        PythonLiteral,
        SeqIntoVals,
    )


def from_query(query: exp.Expr, **relations: IntoRel) -> LazyFrame:
    return ScanSource.from_query(query, **relations).into_frame()


def from_table(table: str) -> LazyFrame:
    return ScanSource.from_table(table).into_frame()


def from_table_function(function: str) -> LazyFrame:
    return ScanSource.from_table_function(function).into_frame()


def from_df(df: IntoFrame) -> LazyFrame:
    return ScanSource.from_df(df).into_frame()


def from_numpy(arr: AnyArray, orient: Orientation = "col") -> LazyFrame:
    return ScanSource.from_numpy(arr, orient=orient).into_frame()


def from_dict(mapping: IntoDict[str, PythonLiteral]) -> LazyFrame:
    return ScanSource.from_dict(mapping).into_frame()


def from_dicts(data: Sequence[Mapping[str, PythonLiteral]]) -> LazyFrame:
    return ScanSource.from_dicts(data).into_frame()


def from_records(data: SeqIntoVals, orient: Orientation = "col") -> LazyFrame:
    return ScanSource.from_records(data, orient=orient).into_frame()
