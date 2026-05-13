from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import duckdb
from duckdb import sqltypes
from pyochain import Range, Seq, Vec
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

import belugas as bl

if TYPE_CHECKING:
    from duckdb.sqltypes import DuckDBPyType


class Variant(NamedTuple):
    label: str
    fn: Callable[[DuckDBPyType], bl.DataType]


class BenchResult(NamedTuple):
    case: DuckDBPyType
    times: Seq[float]

    @property
    def speedup(self) -> float:
        return self.times[-1] / self.times[0]

    @property
    def name(self) -> str:
        return bl.DataType.from_duckdb(self.case).__class__.__name__


CASES: Seq[DuckDBPyType] = Seq((
    duckdb.decimal_type(18, 3),
    duckdb.decimal_type(38, 10),
    duckdb.type("ENUM('a','b','c')"),
    duckdb.list_type(sqltypes.INTEGER),
    duckdb.list_type(
        duckdb.struct_type({"x": sqltypes.INTEGER, "y": sqltypes.VARCHAR})
    ),
    duckdb.array_type(sqltypes.INTEGER, 3),
    duckdb.array_type(sqltypes.DOUBLE, 5),
    duckdb.struct_type({
        "a": duckdb.list_type(sqltypes.INTEGER),
        "b": duckdb.decimal_type(10, 2),
    }),
    duckdb.struct_type({
        "a": sqltypes.INTEGER,
        "b": duckdb.struct_type({
            "c": duckdb.list_type(sqltypes.DOUBLE),
            "d": duckdb.map_type(sqltypes.INTEGER, sqltypes.VARCHAR),
        }),
    }),
    duckdb.map_type(sqltypes.INTEGER, sqltypes.VARCHAR),
    duckdb.map_type(
        sqltypes.VARCHAR,
        duckdb.struct_type({"a": sqltypes.INTEGER, "b": sqltypes.DOUBLE}),
    ),
    duckdb.union_type({"v0": sqltypes.INTEGER, "v1": sqltypes.VARCHAR}),
))

VARIANTS: Seq[Variant] = Seq((
    Variant("impl", bl.DataType.from_duckdb),
    Variant("baseline (from_str)", lambda dtype: bl.DataType.from_str(str(dtype))),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
))


def main(n: int) -> None:
    table = _init_table(n)
    _compute_results(table, n)
    Console().print(table)


def _init_table(n: int) -> Table:
    table = Table(title=f"_build_* benchmark (N={n})")
    table.add_column("type", style="cyan", no_wrap=True)
    table.add_column("result", style="dim", no_wrap=True)
    VARIANTS.iter().for_each(lambda v: table.add_column(v.label, justify="right"))
    table.add_column("impl vs baseline", justify="right", style="green")
    return table


def _compute_results(table: Table, n: int) -> None:
    results = Vec[BenchResult].new()

    def _bench(fn: Callable[[DuckDBPyType], object], dtype: DuckDBPyType) -> float:
        def _once() -> float:
            t0 = time.perf_counter()
            _ = fn(dtype)
            return (time.perf_counter() - t0) * 1e6

        return Range(0, n).iter().map(lambda _: _once()).into(statistics.median)

    with Progress() as progress:
        task = progress.add_task(
            "benchmarking...", total=CASES.length() * VARIANTS.length()
        )
        _ = (
            CASES
            .iter()
            .map(
                lambda case: BenchResult(
                    case,
                    VARIANTS
                    .iter()
                    .map(lambda v: _bench(v.fn, case))
                    .inspect(lambda _: progress.advance(task))
                    .collect(),
                )
            )
            .collect_into(results)
        )
    return (
        results
        .sort(key=lambda r: r.speedup, reverse=True)
        .rev()
        .for_each(lambda r: _add_to_table(r, table))
    )


def _add_to_table(result: BenchResult, table: Table) -> None:
    table.add_row(
        str(result.case),
        result.name,
        *[f"{t:.1f}μ" for t in result.times],
        f"{result.speedup:.1f}x",
    )
