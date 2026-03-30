"""Generate module-level DuckDB meta table functions (duckdb_*).

These are TABLE-type functions that return DuckDB relations containing
metadata about the DuckDB instance (tables, columns, functions, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import polars as pl
import pyochain as pc
from rich import print
from rich.text import Text

from ..fn_generator._dtypes import DuckDbTypes, FuncTypes
from ..fn_generator._rules import CONVERTER, SHADOWERS


@dataclass(slots=True)
class MetaFnInfo:
    """Container for a meta table function's metadata."""

    name: str
    final_name: str
    description: pc.Option[str]
    params: pc.Seq[tuple[str, str]]
    varargs_type: pc.Option[str]

    @classmethod
    def from_row(  # noqa: PLR0913, PLR0917
        cls,
        name: str,
        final_name: str,
        description: str | None,
        params: list[str],
        py_types: list[str],
        varargs_type: str | None,
    ) -> Self:
        return cls(
            name,
            final_name,
            pc.Option(description),
            pc.Iter(params).zip(py_types, strict=True).collect(),
            pc.Option(varargs_type),
        )

    def _body_args(self) -> str:
        return (
            self.params
            .iter()
            .map_star(lambda n, _: n)
            .chain(
                self.varargs_type.map(lambda _: pc.Iter.once("*args")).unwrap_or_else(
                    pc.Iter.new
                )
            )
            .collect()
            .then(lambda a: f", {a.join(', ')}")
            .unwrap_or("")
        )

    def _args_section(self) -> str:
        return (
            self.params
            .iter()
            .map_star(lambda n, t: f"        {n} ({t} | None): Parameter")
            .chain(
                self.varargs_type.map(
                    lambda t: pc.Iter.once(f"        *args ({t}): Variable arguments")
                ).unwrap_or_else(pc.Iter.new)
            )
            .collect()
            .then(lambda docs: f"\n\n    Args:\n{docs.join(chr(10))}")
            .unwrap_or("")
        )

    def _signature(self) -> str:
        return (
            self.params
            .iter()
            .map_star(lambda n, t: f"{n}: {t} | None = None")
            .chain(
                self.varargs_type.map(
                    lambda t: pc.Iter.once(f"*args: {t}")
                ).unwrap_or_else(pc.Iter.new)
            )
            .join(", ")
        )

    def _description(self) -> str:
        return self.description.map(
            lambda d: (
                " ".join(d.split()).rstrip(".").replace('"', "").replace("\u2019", "'")
            )
        ).unwrap_or(f"SQL {self.name} table function")

    def build(self) -> str:
        return f'''
def {self.final_name}({self._signature()}) -> duckdb.DuckDBPyRelation:
    """{self._description()}.

    **SQL name**: *{self.name}*{self._args_section()}

    Returns:
        duckdb.DuckDBPyRelation
    """
    return duckdb.table_function("{self.name}"{self._body_args()})'''


def run_pipeline(caller: Path, source: Path) -> str:
    return (
        pl
        .scan_parquet(source)
        .pipe(_query)
        .collect()
        .pipe(_to_infos)
        .inspect(
            lambda x: print(
                Text(f"Generated {x.length()} meta functions", style="yellow")
            )
        )
        .into(_build_file, caller)
    )


def _build_file(fns: pc.Seq[MetaFnInfo], caller: Path) -> str:
    body = fns.iter().map(MetaFnInfo.build).join("\n\n\n")
    return f"{_header(caller)}\n\n\n{body}\n"


def _query(lf: pl.LazyFrame) -> pl.LazyFrame:
    fn_name = pl.col("function_name")
    return (
        lf
        .filter(
            pl.col("function_type").cast(pl.String).eq(FuncTypes.TABLE),
            fn_name.str.starts_with("duckdb_"),
            fn_name.eq("duckdb_table_sample").not_(),
        )
        .select(
            fn_name,
            fn_name.str.strip_prefix("duckdb_").alias("final_name"),
            "description",
            pl.col("parameters").list.eval(
                pl
                .element()
                .str.strip_chars("'\"[]")
                .str.to_lowercase()
                .pipe(
                    lambda e: (
                        pl
                        .when(e.is_in(SHADOWERS))
                        .then(pl.concat_str(e, pl.lit("_arg")))
                        .otherwise(e)
                    )
                )
            ),
            pl
            .col("parameter_types")
            .list.eval(
                pl
                .element()
                .fill_null(DuckDbTypes.ANY)
                .cast(pl.String)
                .replace_strict(CONVERTER, default="object", return_dtype=pl.String)
            )
            .alias("py_types"),
            pl
            .col("varargs")
            .cast(pl.String)
            .replace_strict(CONVERTER, default="object", return_dtype=pl.String)
            .alias("varargs_type"),
        )
        .unique(subset=fn_name, keep="first")
        .sort(fn_name)
    )


def _to_infos(df: pl.DataFrame) -> pc.Seq[MetaFnInfo]:
    return (
        df
        .map_rows(lambda x: MetaFnInfo.from_row(*x), return_dtype=pl.Object)  # pyright: ignore[reportAny]
        .pipe(lambda df: pc.Iter[MetaFnInfo](df.to_series()))
        .collect()
    )


def _header(caller: Path) -> str:
    return f'''"""DuckDB meta information table functions.

Functions are extracted from DuckDB duckdb_functions() introspection.

This file is AUTO-GENERATED by `{caller.as_posix()}`. DO NOT EDIT MANUALLY.
"""

from __future__ import annotations

import duckdb'''
