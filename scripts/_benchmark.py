from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable

import polars as pl
from polars.testing import assert_frame_equal
from pyochain import Dict, Iter, Option, Range, Seq
from rich import print
from rich.progress import Progress
from rich.table import Table

import belugas as bl

type Frame = pl.LazyFrame | bl.LazyFrame
type BenchFn[T: pl.LazyFrame | bl.LazyFrame] = Callable[[], T]
N_COLS = 25
N_GROUPS = 10


COLS: Seq[str] = Range(0, N_COLS).iter().map(lambda i: f"c{i}").collect()

_DATA = pl.DataFrame({
    "c0": list(range(N_GROUPS)),
    **Range(1, N_COLS).iter().map(lambda i: (f"c{i}", [1] * N_GROUPS)).collect(Dict),
}).to_arrow()
_RHS_DATA = pl.DataFrame({
    "c0": list(range(5)),
    "k": list(range(5, 10)),
    "l": list(range(10, 15)),
    "m": list(range(15, 20)),
}).to_arrow()
_STRUCT_DATA = pl.DataFrame({"s": [{"x": 1, "y": 2}]}).to_arrow()
_ASOF_L_DATA = pl.DataFrame({"key": [1, 2, 3], "val": [10, 20, 30]}).to_arrow()
_ASOF_R_DATA = pl.DataFrame({"key": [1, 2, 3], "rval": [100, 200, 300]}).to_arrow()
_PIVOT_DATA = pl.DataFrame({
    "idx": [1, 1],
    "col": ["a", "b"],
    "val": [10, 20],
}).to_arrow()
_EXPLODE_DATA = pl.DataFrame({
    "id": tuple(range(N_GROUPS)),
    **Range(1, N_COLS)
    .iter()
    .map(
        lambda i: (
            f"c{i}",
            Iter.once(tuple(range(3))).cycle().take(N_GROUPS).collect(tuple),
        )
    )
    .collect(Dict),
}).to_arrow()
# NOTE: arrow is badly typed, so polars best-effort can't go beyong Series | DataFrame when converting from arrow, even if we know it's a DataFrame. Hence the pyright ignores below.
BASE = bl.from_arrow(_DATA)
PL_BASE: pl.LazyFrame = pl.from_arrow(_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
RHS = bl.from_arrow(_RHS_DATA)
PL_RHS: pl.LazyFrame = pl.from_arrow(_RHS_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
STRUCT_BL = bl.from_arrow(_STRUCT_DATA)
STRUCT_PL: pl.LazyFrame = pl.from_arrow(_STRUCT_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
ASOF_L_BL = bl.from_arrow(_ASOF_L_DATA)
ASOF_R_BL = bl.from_arrow(_ASOF_R_DATA)
ASOF_L_PL: pl.LazyFrame = pl.from_arrow(_ASOF_L_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
ASOF_R_PL: pl.LazyFrame = pl.from_arrow(_ASOF_R_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
PIVOT_BL = bl.from_arrow(_PIVOT_DATA)
PIVOT_PL: pl.LazyFrame = pl.from_arrow(_PIVOT_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]
EXPLODE_BL = bl.from_arrow(_EXPLODE_DATA)
EXPLODE_PL: pl.LazyFrame = pl.from_arrow(_EXPLODE_DATA).lazy()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]

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

BENCHS = Dict.from_ref({
    "select": (lambda: BASE.select(COLS), lambda: PL_BASE.select(COLS)),
    "with_columns": (
        lambda: BASE.with_columns(MUL),
        lambda: PL_BASE.with_columns(PL_MUL),
    ),
    "filter": (
        lambda: BASE.filter(bl.col("c1").gt(0)),
        lambda: PL_BASE.filter(pl.col("c1").gt(0)),
    ),
    "group_by": (
        lambda: BASE.group_by("c0").agg(AGG),
        lambda: PL_BASE.group_by("c0").agg(PL_AGG),
    ),
    "join": (
        lambda: BASE.join(RHS, on="c0", how="left"),
        lambda: PL_BASE.join(PL_RHS, on="c0", how="left"),
    ),
    "drop": (
        lambda: BASE.drop("c1"),
        lambda: PL_BASE.drop("c1"),
    ),
    "unnest": (
        lambda: STRUCT_BL.unnest("s"),
        lambda: STRUCT_PL.unnest("s"),
    ),
    "join_asof": (
        lambda: ASOF_L_BL.join_asof(ASOF_R_BL, on="key"),
        lambda: ASOF_L_PL.join_asof(ASOF_R_PL, on="key"),
    ),
    "pivot": (
        lambda: PIVOT_BL.pivot(
            on="col", on_columns=["a", "b"], index="idx", values="val"
        ),
        lambda: PIVOT_PL.pivot(
            on="col", on_columns=["a", "b"], index="idx", values="val"
        ),
    ),
    "unpivot": (
        lambda: BASE.unpivot(on=UNPIVOT_ON, index=["c0"]),
        lambda: PL_BASE.unpivot(on=UNPIVOT_ON, index=["c0"]),
    ),
    "explode": (
        lambda: EXPLODE_BL.explode(UNPIVOT_ON),
        lambda: EXPLODE_PL.explode(UNPIVOT_ON),
    ),
    "unique": (
        lambda: BASE.unique(COLS_UNIQUE),
        lambda: PL_BASE.unique(COLS_UNIQUE),
    ),
    "slice": (
        lambda: BASE.slice(1, 5),
        lambda: PL_BASE.slice(1, 5),
    ),
})


def run_benchmark(runs: int, names: Option[list[str]]) -> None:
    table = _get_table()

    benchmarks = _get_benchmarks(names)
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Running benchmarks...", total=benchmarks.length()
        )

        def _run_bench(
            name: str, bl_fn: BenchFn[bl.LazyFrame], pl_fn: BenchFn[pl.LazyFrame]
        ) -> tuple[str, float, float]:
            assert_frame_equal(
                bl_fn().collect(),
                pl_fn().collect(),
                check_dtypes=False,
                check_row_order=False,
            )
            return name, _get_timing(runs, bl_fn), _get_timing(runs, pl_fn)

        def _process_benchmark(name: str, bl_t: float, pl_t: float) -> None:
            table.add_row(
                name, f"{bl_t:.2f} ms", f"{pl_t:.2f} ms", f"{bl_t / pl_t:.1f}x"
            )
            progress.update(task, advance=1)

        benchmarks.items().iter().map_star(
            lambda k, v: _run_bench(k, *v)
        ).for_each_star(_process_benchmark)
    print(table)


def _get_benchmarks(
    names: Option[list[str]],
) -> Dict[str, tuple[BenchFn[bl.LazyFrame], BenchFn[pl.LazyFrame]]]:
    return names.map(
        lambda ns: Iter(ns).map(lambda t: (t, BENCHS.pop(t))).collect(Dict)
    ).unwrap_or(BENCHS)


def _get_table() -> Table:
    table = Table(
        title="Belugas Benchmark", show_header=True, header_style="bold magenta"
    )
    table.add_column("Benchmark", justify="left")
    table.add_column("Belugas (ms)", justify="right")
    table.add_column("Polars (ms)", justify="right")
    table.add_column("Ratio (bl/pl)", justify="right")
    return table


def _get_timing(runs: int, fn: BenchFn[Frame]) -> float:
    return (
        Range(0, runs)
        .iter()
        .map(lambda _: timeit.timeit(fn, number=1) * 1000)
        .into(statistics.median)
    )
