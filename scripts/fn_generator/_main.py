"""Generate typed SQL function wrappers from DuckDB introspection."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.exceptions import ComputeError
from pyochain import Iter
from rich import print
from rich.text import Text

from ._query import run_qry
from ._schemas import TableSchema
from ._sections import FunctionInfo, build_file


def run_pipeline(
    caller: Path, source: Path, *, profile: bool = False, regenerate: bool
) -> str:
    return (
        _try_scan(source, regenerate=regenerate)
        .pipe(run_qry)
        .pipe(_inspect if profile else lambda lf: lf)
        .collect()
        .map_rows(lambda x: FunctionInfo(*x), return_dtype=pl.Object)
        .pipe(lambda df: Iter[FunctionInfo](df.to_series()))
        .collect()
        .inspect(
            lambda x: print(Text(f"Generated {x.length()} functions", style="yellow"))
        )
        .into(build_file, caller)
    )


def _try_scan(source: Path, *, regenerate: bool) -> pl.LazyFrame:
    if source.exists() and not regenerate:
        return pl.scan_parquet(source)
    import duckdb

    conn = duckdb.connect()
    conn.install_extension("spatial")
    conn.load_extension("spatial")
    conn.install_extension("delta")
    conn.load_extension("delta")
    df = conn.table_function("duckdb_functions").pl().cast(TableSchema)
    df.write_parquet(source)
    return df.lazy()


def _inspect(lf: pl.LazyFrame) -> pl.LazyFrame:
    try:
        lf.profile()[1].with_columns(
            pl.col("end").sub(pl.col("start")).alias("duration")
        ).sort("duration", descending=True).show(10, fmt_str_lengths=100)
    except ComputeError:
        return lf
    return lf
