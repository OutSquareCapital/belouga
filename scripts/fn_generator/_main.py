"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pyochain as pc
from polars.exceptions import ComputeError
from rich import print
from rich.text import Text

from ._query import run_qry
from ._schemas import TableSchema
from ._sections import FunctionInfo, build_file


def get_data(path: Path) -> None:

    import duckdb

    qry = """--sql
    LOAD spatial;
    SELECT *
    FROM duckdb_functions()
    """

    return duckdb.connect().sql(qry).pl().cast(TableSchema).write_parquet(path)


def _inspect(lf: pl.LazyFrame) -> pl.LazyFrame:
    try:
        lf.profile()[1].with_columns(
            pl.col("end").sub(pl.col("start")).alias("duration")
        ).sort("duration", descending=True).show(10, fmt_str_lengths=100)
    except ComputeError:
        return lf
    return lf


def run_pipeline(caller: Path, source: Path, *, profile: bool = False) -> str:
    return (
        pl
        .scan_parquet(source)
        .pipe(run_qry)
        .pipe(_inspect if profile else lambda lf: lf)
        .collect()
        .map_rows(lambda x: FunctionInfo(*x), return_dtype=pl.Object)  # pyright: ignore[reportAny]
        .pipe(lambda df: pc.Iter[FunctionInfo](df.to_series()))
        .collect()
        .inspect(
            lambda x: print(Text(f"Generated {x.length()} functions", style="yellow"))
        )
        .into(build_file, caller)
    )
