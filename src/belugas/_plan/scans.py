from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from pyochain import Dict, Iter, Null, Option, Range, Seq, Some

from .._core import BelugasConversionError
from ..datatypes import DataType
from ..typing import CSVOptions, JsonOptions, LitSeq, NestedSeq, ParquetOptions, Schema

if TYPE_CHECKING:
    from sqlglot import exp

    from ..typing import (
        AnyArray,
        IntoArrow,
        IntoDict,
        IntoPolars,
        Orientation,
        PathOrBuffer,
        PythonLiteral,
        SeqIntoVals,
    )

# TODO: see if we can pushdown belugas IR into in-memory scans.
# DuckDB will handle file reads just fine, but we should aim to minimize the python objects conversions to duckdb expressions, as well as the sqlglot AST size.
# Polars pushdown seem like a quick free win, espcially with lazyframe: simply select the needed columns before calling collect. same for arrow, and duckdb relations
# Numpy could also be very nice, but seems harder to handle
# The best gain would be from python object: avoid converting unneded dict keys, sequences, etc... if possible.
# This is where taking an approach where we go from the last operations backwards to the scan would be interesting, as we could directly see what columns/keys are needed at the scan level, and only convert those to duckdb expressions. This would also allow us to pushdown filters and projections into the scan, which would be a big win for in-memory scans.

COL0 = "column_0"


@dataclass(slots=True)
class ScanResult:
    relation: DuckDBPyRelation
    schema: Schema

    @property
    def identity(self) -> str:
        return f"bl_scan_{id(self.relation)}"

    def set_alias(self) -> Self:
        self.relation = self.relation.set_alias(self.identity)
        return self


def run_query(
    query: exp.Select, relations: Mapping[str, DuckDBPyRelation]
) -> ScanResult:
    try:
        parsed = query.sql(dialect="duckdb", identify=True)
        namespace = {"duckdb": duckdb, "parsed": parsed, **relations}
        exec("relation = duckdb.from_query(parsed)", namespace)
        relation = cast(DuckDBPyRelation, namespace["relation"])
        schema = (
            Iter(relation.columns)
            .zip(relation.dtypes, strict=True)
            .map_star(lambda k, d: (k, DataType.from_duckdb(d).raw))
            .collect(Dict)
        )
        return ScanResult(relation, schema)

    except duckdb.Error as e:
        raise BelugasConversionError(e, query) from e


def from_dict(data: IntoDict[str, Any]) -> ScanResult:  # pyright: ignore[reportExplicitAny]
    data = Dict(data)

    raw_vals = data.items().iter().map_star(_to_expr).collect(tuple)
    rel = duckdb.values(raw_vals).select(*data.iter().map(_unnest))
    return from_query(rel)


def from_numpy(data: AnyArray, orient: Orientation = "col") -> ScanResult:

    match data.ndim:
        case 1:
            rel = duckdb.values(_to_expr(COL0, data)).select(_unnest(COL0))
            return from_query(rel)
        case _:
            arr = data.T if orient == "col" else data

            def _array_strategy() -> tuple[int, Callable[[int], AnyArray]]:
                match (arr.ndim, orient):
                    case (2, _) | (_, "row"):
                        return 1, lambda j: arr[:, j]  # pyright: ignore[reportAny]
                    case _:
                        return 0, lambda j: arr[j]  # pyright: ignore[reportAny]

            def _named_array(names: Seq[str]) -> DuckDBPyRelation:
                vals = (
                    names
                    .iter()
                    .enumerate()
                    .map_star(lambda j, name: _to_expr(name, arr_getter(j)))
                    .collect(tuple)
                )

                return duckdb.values(vals).select(*names.iter().map(_unnest))

            axis, arr_getter = _array_strategy()
            names_nb: int = arr.shape[axis]  # pyright: ignore[reportAny]
            cols = Range(0, names_nb).iter().map(_named).collect()
            return from_query(_named_array(cols))


def from_polars(
    df: IntoPolars, connection: DuckDBPyConnection | None = None
) -> ScanResult:
    """Create a relation from a Polars DataFrame or LazyFrame.

    Note:
        Two big improvements here would be to:

        1)  Exploit `polars::LazyFrame::collect_batches` to avoid materializing the entire DataFrame in memory at once.
            This would require managing the lifecycle of the Iterator. If we do it naively, it will just freeze once the Iterator is empty.

        2)  Exploit `sqlglot` and the sql capabilities of polars to push down the AST into polars directly.

    Returns:
        Self
    """
    return from_arrow(df.lazy().collect(), connection=connection)


def from_arrow(
    df: IntoArrow, connection: DuckDBPyConnection | None = None
) -> ScanResult:
    return from_query(duckdb.from_arrow(df, connection=connection))


def from_records(data: SeqIntoVals, orient: Orientation = "col") -> ScanResult:
    match data[0]:
        case Mapping():
            vals = cast(Sequence[Mapping[str, Any]], data)  # pyright: ignore[reportExplicitAny]
            return from_dicts(vals)
        case Sequence() as value if not isinstance(value, str | bytes | bytearray):  # pyright: ignore[reportUnknownVariableType]
            vals = cast(NestedSeq, data)
            match orient:
                case "col":
                    return from_seq_col(vals)

                case "row":
                    return from_seq_row(vals)
        case _:
            vals = cast(LitSeq, data)
            return from_seq_lit(vals)


def from_dicts(data: Sequence[Mapping[str, PythonLiteral]]) -> ScanResult:
    return Iter(data[0]).map(lambda key: (key, _into_tup(data, key))).into(from_dict)


def from_seq_lit(data: LitSeq) -> ScanResult:
    rel = duckdb.values(_to_expr(COL0, tuple(data))).select(_unnest(COL0))
    return from_query(rel)


def _unnest(k: str) -> duckdb.Expression:
    return duckdb.FunctionExpression("unnest", duckdb.ColumnExpression(k)).alias(k)


def _to_expr(k: str, v: PythonLiteral) -> duckdb.Expression:
    return duckdb.ConstantExpression(v).alias(k)


def from_seq_col(data: NestedSeq) -> ScanResult:
    return Iter(data).enumerate().map_star(lambda k, v: (_named(k), v)).into(from_dict)


def from_seq_row(data: NestedSeq) -> ScanResult:
    width = len(data[0])
    return (
        Iter(range(width))
        .map(lambda j: (_named(j), _into_tup(data, j)))
        .into(from_dict)
    )


def _named(j: object) -> str:
    return f"column_{j}"


def _into_tup[T](
    vals: Iterable[Sequence[T]] | Iterable[Mapping[T, object]], key: T
) -> tuple[T, ...]:
    return Iter(vals).map(operator.itemgetter(key)).collect(tuple)


def from_table(name: str) -> ScanResult:
    return from_query(duckdb.table(name))


def from_table_function(name: str, *args: object) -> ScanResult:
    return from_query(duckdb.table_function(name, *args))


def from_parquet(
    path: Path | str | Iterable[str | Path],
    connection: Option[DuckDBPyConnection],
    options: ParquetOptions,
) -> ScanResult:
    match path:
        case Path():
            target = str(path)
        case str():
            target = path
        case Iterable():
            target = (
                Iter(path).map(lambda x: x if isinstance(x, str) else str(x)).collect()
            )
    rel = _get_conn(connection).from_parquet(target, **options)
    return from_query(rel)


def from_csv(
    path: PathOrBuffer, connection: Option[DuckDBPyConnection], options: CSVOptions
) -> ScanResult:
    rel = _get_conn(connection).from_csv_auto(path, **options)
    return from_query(rel)


def from_json(
    path: PathOrBuffer, connection: Option[DuckDBPyConnection], options: JsonOptions
) -> ScanResult:
    rel = _get_conn(connection).read_json(path, **options)
    return from_query(rel)


def _get_conn(connection: Option[DuckDBPyConnection]) -> DuckDBPyConnection:
    match connection:
        case Some(conn):
            return conn
        case Null():
            return duckdb.default_connection()


def from_query(relation: DuckDBPyRelation) -> ScanResult:
    schema = (
        Iter(relation.columns)
        .zip(relation.dtypes, strict=True)
        .map_star(lambda k, d: (k, DataType.from_duckdb(d).raw))
        .collect(Dict)
    )

    return ScanResult(relation, schema)
