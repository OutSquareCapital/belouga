from __future__ import annotations

import operator
import statistics
import timeit
from collections.abc import Callable

import polars as pl
from pyochain import Dict, Iter, Option, Range, Seq
from rich import print
from rich.progress import Progress
from rich.table import Table

import belugas as bl

type BenchFn = Callable[[], bl.LazyFrame]
N_COLS = 500
N_UNNEST_COLS = 250
N_GROUPS = Range(0, 10)


COLS: Seq[str] = Range(0, N_COLS).iter().map(lambda i: f"c{i}").collect()
UNNEST_COLS: Seq[str] = Range(0, N_UNNEST_COLS).iter().map(lambda i: f"s{i}").collect()

_DATA = pl.DataFrame({
    "c0": N_GROUPS,
    **Range(1, N_COLS)
    .iter()
    .map(lambda i: (f"c{i}", [1] * N_GROUPS.length()))
    .collect(Dict),
})
_RHS_DATA = pl.DataFrame({
    "c0": Range(0, 5),
    "k": Range(5, 10),
    "l": Range(10, 15),
    "m": Range(15, 20),
})
_STRUCT_DATA = pl.DataFrame(
    UNNEST_COLS
    .iter()
    .enumerate()
    .map_star(
        lambda i, name: (
            name,
            [{f"{name}_left": i, f"{name}_right": i}],
        )
    )
    .collect(dict)
)
_ASOF_L_DATA = pl.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
_ASOF_R_DATA = pl.DataFrame({"key": [1, 2, 3], "rval": [100, 200, 300]})
_PIVOT_DATA = pl.DataFrame({
    "idx": [1, 1],
    "col": ["a", "b"],
    "val": [10, 20],
})
_EXPLODE_DATA = pl.DataFrame({
    "id": N_GROUPS,
    **Range(1, N_COLS)
    .iter()
    .map(
        lambda i: (
            f"c{i}",
            Range(0, 3).iter().cycle().take(N_GROUPS.length()).collect(),
        )
    )
    .collect(Dict),
})
# NOTE: arrow is badly typed, so polars best-effort can't go beyong Series | DataFrame when converting from arrow, even if we know it's a DataFrame. Hence the pyright ignores below.
BASE = bl.from_arrow(_DATA)
RHS = bl.from_arrow(_RHS_DATA)
STRUCT_BL = bl.from_arrow(_STRUCT_DATA)
ASOF_L_BL = bl.from_arrow(_ASOF_L_DATA)
ASOF_R_BL = bl.from_arrow(_ASOF_R_DATA)
PIVOT_BL = bl.from_arrow(_PIVOT_DATA)
EXPLODE_BL = bl.from_arrow(_EXPLODE_DATA)

AGG: Seq[bl.Expr] = (
    COLS
    .iter()
    .map(
        lambda c: (
            bl
            .when(bl.col(c).gt(bl.lit(0)))
            .then(bl.col(c).add(1))
            .otherwise(bl.lit(0))
            .mean()
            .alias(f"{c}_agg")
        )
    )
    .collect()
)

MUL: Seq[bl.Expr] = (
    COLS
    .iter()
    .enumerate()
    .map_star(
        lambda i, c: (
            bl
            .col(c)
            .mul(bl.col("c0"))
            .add(bl.lit(i))
            .cast(bl.UInt32())
            .alias(f"{c}_mul")
        )
    )
    .collect()
)
PREDS = COLS.iter().map(lambda c: bl.col(c).eq(1)).collect()
UNPIVOT_ON = COLS.iter().skip(1).collect()
COLS_UNIQUE = COLS.iter().take(10).collect()

BENCHS = Dict[str, BenchFn].from_ref({
    "with_columns": lambda: BASE.with_columns(MUL),
    "filter": lambda: BASE.filter(PREDS),
    "group_by": lambda: BASE.group_by("c0").agg(AGG),
    "join": lambda: BASE.join(RHS, on="c0", how="left"),
    "drop": lambda: BASE.drop(COLS),
    "unnest": lambda: STRUCT_BL.unnest(UNNEST_COLS),
    "join_asof": lambda: ASOF_L_BL.join_asof(ASOF_R_BL, on="key"),
    "pivot": lambda: PIVOT_BL.pivot(
        on="col", on_columns=["a", "b"], index="idx", values="val"
    ),
    "unpivot": lambda: BASE.unpivot(on=UNPIVOT_ON, index=["c0"]),
    "explode": lambda: EXPLODE_BL.explode(UNPIVOT_ON),
    "unique": lambda: BASE.unique(COLS_UNIQUE),
    "slice": lambda: BASE.slice(1, 5),
    "drop_nulls": lambda: BASE.drop_nulls(COLS),
})


def run_benchmark(runs: int, names: Option[list[str]]) -> None:
    table = _get_table()

    benchmarks = _get_benchmarks(names)
    with Progress() as progress:
        _run_all(progress, benchmarks, runs, table)
    print(table)


def _run_all(
    progress: Progress, benchmarks: Dict[str, BenchFn], runs: int, table: Table
) -> None:
    def _run_bench(name: str, bl_fn: BenchFn) -> tuple[str, float]:
        return name, _get_timing(runs, bl_fn)

    def _process_benchmark(name: str, bl_t: float) -> None:
        table.add_row(name, f"{bl_t:.4f}")

    descr = f"[cyan]Running benchmarks on {runs} runs per test..."

    tracker = benchmarks.items().into(
        progress.track, benchmarks.length(), description=descr
    )
    return (
        Iter(tracker)
        .map_star(_run_bench)
        .sort_by(operator.itemgetter(1))
        .iter()
        .for_each_star(_process_benchmark)
    )


def _get_benchmarks(names: Option[list[str]]) -> Dict[str, BenchFn]:
    return names.map(
        lambda ns: Iter(ns).map(lambda t: (t, BENCHS.pop(t))).collect(Dict)
    ).unwrap_or(BENCHS)


def _get_table() -> Table:
    table = Table(
        title="Belugas Benchmark", show_header=True, header_style="bold magenta"
    )
    table.add_column("Benchmark", justify="left")
    table.add_column("Median time (ms)", justify="right")
    return table


def _get_timing(runs: int, fn: BenchFn) -> float:
    return (
        Range(0, runs)
        .iter()
        .map(lambda _: timeit.timeit(fn().query.logical, number=1) * 1000)
        .into(statistics.median)
    )
