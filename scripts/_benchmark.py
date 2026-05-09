from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable

import polars as pl
from pyochain import Iter, Seq
from rich import print
from rich.progress import Progress
from rich.table import Table

import belugas as bl

COLS = Seq("abcdefghij")
BASE = bl.LazyFrame(COLS.iter().map(lambda c: (c, [1])).collect(dict))
RHS = bl.LazyFrame({"a": [1], "k": [10], "l": [20], "m": [30]})

AGG = COLS.iter().map(lambda c: bl.col(c).sum().alias(f"{c}_sum")).collect()
MUL = (
    COLS
    .iter()
    .enumerate()
    .map_star(lambda i, c: bl.col(c).mul(bl.lit(i + 1)).alias(f"{c}_x{i + 1}"))
    .collect()
)

PL_BASE = pl.DataFrame(COLS.iter().map(lambda c: (c, [1])).collect(dict))
PL_RHS = pl.DataFrame({"a": [1], "k": [10], "l": [20], "m": [30]})
PL_AGG = COLS.iter().map(lambda c: pl.col(c).sum().alias(f"{c}_sum")).collect()
PL_MUL = (
    COLS
    .iter()
    .enumerate()
    .map_star(lambda i, c: pl.col(c).mul(pl.lit(i + 1)).alias(f"{c}_x{i + 1}"))
    .collect()
)

MIXED_DATA = COLS.iter().map(lambda c: (c, [1])).collect(dict)
BL_MIXED = bl.LazyFrame(MIXED_DATA)
PL_MIXED = pl.DataFrame(MIXED_DATA)

STRUCT_BL = bl.LazyFrame({"s": [{"x": 1, "y": 2}]})
STRUCT_PL = pl.DataFrame({"s": [{"x": 1, "y": 2}]})

ASOF_L_BL = bl.LazyFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
ASOF_R_BL = bl.LazyFrame({"key": [1, 2, 3], "rval": [100, 200, 300]})
ASOF_L_PL = pl.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]})
ASOF_R_PL = pl.DataFrame({"key": [1, 2, 3], "rval": [100, 200, 300]})

PIVOT_BL = bl.LazyFrame({"idx": [1, 1], "col": ["a", "b"], "val": [10, 20]})
PIVOT_PL = pl.DataFrame({"idx": [1, 1], "col": ["a", "b"], "val": [10, 20]})


def bench_select() -> None:
    _ = BASE.select(COLS)


def bench_pl_select() -> None:
    _ = PL_BASE.select(COLS)


def bench_with_columns() -> None:
    _ = BASE.with_columns(MUL)


def bench_pl_with_columns() -> None:
    _ = PL_BASE.with_columns(PL_MUL)


def bench_agg() -> None:
    _ = BASE.group_by("a").agg(AGG)


def bench_pl_agg() -> None:
    _ = PL_BASE.group_by("a").agg(PL_AGG)


def bench_mixed() -> None:
    _ = (
        BL_MIXED
        .select(COLS)
        .with_columns(MUL)
        .filter(bl.col("a_x1").gt(0))
        .group_by("a")
        .agg(AGG)
    )


def bench_pl_mixed() -> None:
    _ = (
        PL_MIXED
        .select(COLS)
        .with_columns(PL_MUL)
        .filter(pl.col("a_x1").gt(0))
        .group_by("a")
        .agg(PL_AGG)
    )


def bench_join() -> None:
    _ = BASE.join(RHS, on="a", how="left")


def bench_pl_join() -> None:
    _ = PL_BASE.join(PL_RHS, on="a", how="left")


def bench_drop() -> None:
    _ = BASE.drop("a")


def bench_pl_drop() -> None:
    _ = PL_BASE.drop("a")


def bench_unnest() -> None:
    _ = STRUCT_BL.unnest("s")


def bench_pl_unnest() -> None:
    _ = STRUCT_PL.unnest("s")


def bench_join_asof() -> None:
    _ = ASOF_L_BL.join_asof(ASOF_R_BL, on="key")


def bench_pl_join_asof() -> None:
    _ = ASOF_L_PL.join_asof(ASOF_R_PL, on="key", strategy="backward")


def bench_pivot() -> None:
    _ = PIVOT_BL.pivot(on="col", on_columns=["a", "b"], index="idx", values="val")


def bench_pl_pivot() -> None:
    _ = PIVOT_PL.pivot(on="col", index="idx", values="val")


def bench_unpivot() -> None:
    _ = BASE.unpivot(on=list("bcdefghij"), index=["a"])


def bench_pl_unpivot() -> None:
    _ = PL_BASE.unpivot(on=list("bcdefghij"), index=["a"])


def run_benchmark(runs: int) -> None:

    table = _get_table()
    benchmarks = _get_benchmarks()
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Running benchmarks...", total=benchmarks.length()
        )

        def _process_benchmark(name: str, bl_t: float, pl_t: float) -> None:
            table.add_row(
                name, f"{bl_t:.2f} ms", f"{pl_t:.2f} ms", f"{bl_t / pl_t:.1f}x"
            )
            progress.update(task, advance=1)

        benchmarks.iter().map_star(
            lambda name, bl_fn, pl_fn: (
                name,
                _get_timing(runs, bl_fn),
                _get_timing(runs, pl_fn),
            )
        ).for_each_star(_process_benchmark)
    print(table)


def _get_table() -> Table:
    table = Table(
        title="Belugas Benchmark", show_header=True, header_style="bold magenta"
    )
    table.add_column("Benchmark", justify="left")
    table.add_column("Belugas (ms)", justify="right")
    table.add_column("Polars (ms)", justify="right")
    table.add_column("Ratio (bl/pl)", justify="right")
    return table


def _get_benchmarks() -> Seq[tuple[str, Callable[[], None], Callable[[], None]]]:
    return Seq((
        ("select", bench_select, bench_pl_select),
        ("with_columns", bench_with_columns, bench_pl_with_columns),
        ("agg", bench_agg, bench_pl_agg),
        ("mixed", bench_mixed, bench_pl_mixed),
        ("join", bench_join, bench_pl_join),
        ("drop", bench_drop, bench_pl_drop),
        ("unnest", bench_unnest, bench_pl_unnest),
        ("join_asof", bench_join_asof, bench_pl_join_asof),
        ("pivot", bench_pivot, bench_pl_pivot),
        ("unpivot", bench_unpivot, bench_pl_unpivot),
    ))


def _get_timing(runs: int, fn: Callable[[], None]) -> float:
    return (
        Iter(range(runs))
        .map(lambda _: timeit.timeit(fn, number=1) * 1000)
        .into(statistics.median)
    )
