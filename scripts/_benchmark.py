from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable
from typing import Any

import polars as pl
from polars.testing import assert_frame_equal
from pyochain import Dict, Iter, Seq
from rich import print
from rich.progress import Progress
from rich.table import Table

import belugas as bl

type BenchFn[T: pl.DataFrame | bl.LazyFrame] = Callable[[], T]
N_COLS = 25
N_GROUPS = 10


COLS: Seq[str] = Iter(range(N_COLS)).map(lambda i: f"c{i}").collect()

_DATA: dict[str, list[int]] = {
    "c0": list(range(N_GROUPS)),
    **Iter(range(1, N_COLS)).map(lambda i: (f"c{i}", [1] * N_GROUPS)).collect(Dict),
}
_RHS_DATA: dict[str, list[int]] = {
    "c0": list(range(5)),
    "k": list(range(5, 10)),
    "l": list(range(10, 15)),
    "m": list(range(15, 20)),
}
_STRUCT_DATA: dict[str, list[dict[str, int]]] = {"s": [{"x": 1, "y": 2}]}
_ASOF_L_DATA: dict[str, list[int]] = {"key": [1, 2, 3], "val": [10, 20, 30]}
_ASOF_R_DATA: dict[str, list[int]] = {"key": [1, 2, 3], "rval": [100, 200, 300]}
_PIVOT_DATA: dict[str, list[int] | list[str]] = {
    "idx": [1, 1],
    "col": ["a", "b"],
    "val": [10, 20],
}
_EXPLODE_DATA = {
    "id": tuple(range(N_GROUPS)),
    **Iter(range(1, N_COLS))
    .map(
        lambda i: (
            f"c{i}",
            Iter.once(tuple(range(3))).cycle().take(N_GROUPS).collect(tuple),
        )
    )
    .collect(Dict),
}

BASE = bl.LazyFrame(_DATA)
PL_BASE = pl.DataFrame(_DATA)
RHS = bl.LazyFrame(_RHS_DATA)
PL_RHS = pl.DataFrame(_RHS_DATA)
STRUCT_BL = bl.LazyFrame(_STRUCT_DATA)
STRUCT_PL = pl.DataFrame(_STRUCT_DATA)
ASOF_L_BL = bl.LazyFrame(_ASOF_L_DATA)
ASOF_R_BL = bl.LazyFrame(_ASOF_R_DATA)
ASOF_L_PL = pl.DataFrame(_ASOF_L_DATA)
ASOF_R_PL = pl.DataFrame(_ASOF_R_DATA)
PIVOT_BL = bl.LazyFrame(_PIVOT_DATA)
PIVOT_PL = pl.DataFrame(_PIVOT_DATA)
EXPLODE_BL = bl.LazyFrame(_EXPLODE_DATA)
EXPLODE_PL = pl.DataFrame(_EXPLODE_DATA)

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

PL_AGG: Seq[pl.Expr] = (
    COLS
    .iter()
    .map(
        lambda c: (
            pl
            .when(pl.col(c).gt(pl.lit(0)))
            .then(pl.col(c).add(1))
            .otherwise(pl.lit(0))
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
PL_MUL: Seq[pl.Expr] = (
    COLS
    .iter()
    .enumerate()
    .map_star(
        lambda i, c: (
            pl.col(c).mul(pl.col("c0")).add(pl.lit(i)).cast(pl.UInt32).alias(f"{c}_mul")
        )
    )
    .collect()
)
UNPIVOT_ON = COLS.iter().skip(1).collect()
COLS_UNIQUE = COLS.iter().take(10).collect()


def run_benchmark(runs: int) -> None:
    table = _get_table()
    benchmarks = _get_benchmarks()
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Running benchmarks...", total=benchmarks.length()
        )

        def _run_bench(
            name: str, bl_fn: BenchFn[bl.LazyFrame], pl_fn: BenchFn[pl.DataFrame]
        ) -> tuple[str, float, float]:
            assert_frame_equal(
                bl_fn().collect(), pl_fn(), check_dtypes=False, check_row_order=False
            )
            return name, _get_timing(runs, bl_fn), _get_timing(runs, pl_fn)

        def _process_benchmark(name: str, bl_t: float, pl_t: float) -> None:
            table.add_row(
                name, f"{bl_t:.2f} ms", f"{pl_t:.2f} ms", f"{bl_t / pl_t:.1f}x"
            )
            progress.update(task, advance=1)

        benchmarks.iter().map_star(_run_bench).for_each_star(_process_benchmark)
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


def _get_timing(runs: int, fn: BenchFn[Any]) -> float:  # pyright: ignore[reportExplicitAny]
    return (
        Iter(range(runs))
        .map(lambda _: timeit.timeit(fn, number=1) * 1000)
        .into(statistics.median)
    )


def _get_benchmarks() -> Seq[tuple[str, BenchFn[bl.LazyFrame], BenchFn[pl.DataFrame]]]:
    return Seq[tuple[str, BenchFn[bl.LazyFrame], BenchFn[pl.DataFrame]]]((
        (
            "select",
            lambda: BASE.select(COLS),
            lambda: PL_BASE.select(COLS),
        ),
        (
            "with_columns",
            lambda: BASE.with_columns(MUL),
            lambda: PL_BASE.with_columns(PL_MUL),
        ),
        (
            "group_by",
            lambda: BASE.group_by("c0").agg(AGG),
            lambda: PL_BASE.group_by("c0").agg(PL_AGG),
        ),
        (
            "join",
            lambda: BASE.join(RHS, on="c0", how="left"),
            lambda: PL_BASE.join(PL_RHS, on="c0", how="left"),
        ),
        (
            "drop",
            lambda: BASE.drop("c1"),
            lambda: PL_BASE.drop("c1"),
        ),
        (
            "unnest",
            lambda: STRUCT_BL.unnest("s"),
            lambda: STRUCT_PL.unnest("s"),
        ),
        (
            "join_asof",
            lambda: ASOF_L_BL.join_asof(ASOF_R_BL, on="key"),
            lambda: ASOF_L_PL.join_asof(ASOF_R_PL, on="key"),
        ),
        (
            "pivot",
            lambda: PIVOT_BL.pivot(
                on="col", on_columns=["a", "b"], index="idx", values="val"
            ),
            lambda: PIVOT_PL.pivot(on="col", index="idx", values="val"),
        ),
        (
            "unpivot",
            lambda: BASE.unpivot(on=UNPIVOT_ON, index=["c0"]),
            lambda: PL_BASE.unpivot(on=UNPIVOT_ON, index=["c0"]),
        ),
        (
            "explode",
            lambda: EXPLODE_BL.explode(UNPIVOT_ON),
            lambda: EXPLODE_PL.explode(UNPIVOT_ON),
        ),
        (
            "unique",
            lambda: BASE.unique(COLS_UNIQUE),
            lambda: PL_BASE.unique(COLS_UNIQUE),
        ),
        (
            "slice",
            lambda: BASE.slice(1, 5),
            lambda: PL_BASE.slice(1, 5),
        ),
    ))
