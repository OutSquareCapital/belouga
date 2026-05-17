"""Generate module-level DuckDB meta table functions (duckdb_*).

These are TABLE-type functions that return DuckDB relations containing
metadata about the DuckDB instance (tables, columns, functions, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import polars as pl
from pyochain import Iter, Option, Seq, option
from rich import print
from rich.text import Text

from ..fn_generator._dtypes import DuckDbTypes, FuncTypes
from ..fn_generator._main import _try_scan  # pyright: ignore[reportPrivateUsage]
from ..fn_generator._query import _to_param_names  # pyright: ignore[reportPrivateUsage]
from ..fn_generator._rules import CONVERTER


@dataclass(slots=True)
class MetaFnInfo:
    """Container for a meta table function's metadata."""

    name: str
    final_name: str
    description: Option[str]
    params: Seq[tuple[str, str]]
    varargs_type: Option[str]

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
            option(description),
            Iter(params).zip(py_types, strict=True).collect(),
            option(varargs_type),
        )

    def build(self) -> str:
        return f'''
def {self.final_name}({self._signature()}) -> LazyFrame:
    """{self._description()}.

    **SQL name**: *{self.name}*{self._args_section()}

    Returns:
        LazyFrame
    """
    return LazyFrame(duckdb.table_function("{self.name}"{self._body_args()}))'''

    def _signature(self) -> str:
        return (
            self.params
            .iter()
            .map_star(lambda n, t: f"{n}: {t} | None = None")
            .chain(
                self.varargs_type.map(
                    lambda t: Iter.once(f"*args: {t}")
                ).unwrap_or_else(Iter.new)
            )
            .join(", ")
        )

    def _description(self) -> str:
        return self.description.map(
            lambda d: (
                " ".join(d.split()).rstrip(".").replace('"', "").replace("\u2019", "'")
            )
        ).unwrap_or(f"SQL {self.name} table function")

    def _args_section(self) -> str:
        return (
            self.params
            .iter()
            .map_star(lambda n, t: f"        {n} ({t} | None): Parameter")
            .chain(
                self.varargs_type.map(
                    lambda t: Iter.once(f"        *args ({t}): Variable arguments")
                ).unwrap_or_else(Iter.new)
            )
            .collect()
            .then(lambda docs: f"\n\n    Args:\n{docs.join(chr(10))}")
            .unwrap_or("")
        )

    def _body_args(self) -> str:
        return (
            self.params
            .iter()
            .map_star(lambda n, _: n)
            .chain(
                self.varargs_type.map(lambda _: Iter.once("*args")).unwrap_or_else(
                    Iter.new
                )
            )
            .collect()
            .then(lambda a: f", {a.join(', ')}")
            .unwrap_or("")
        )


def run_pipeline(caller: Path, source: Path) -> str:
    return (
        _try_scan(source, regenerate=False)
        .pipe(_query)
        .collect()
        .map_rows(lambda x: MetaFnInfo.from_row(*x), return_dtype=pl.Object)  # pyright: ignore[reportAny]
        .pipe(lambda df: Iter[MetaFnInfo](df.to_series()))
        .collect()
        .inspect(
            lambda x: print(
                Text(f"Generated {x.length()} meta functions", style="yellow")
            )
        )
        .into(_build_file, caller)
    )


def _build_file(fns: Seq[MetaFnInfo], caller: Path) -> str:
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
            pl.col("parameters").list.eval(_to_param_names(pl.element())),
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


def _header(caller: Path) -> str:
    return f'''"""DuckDB meta information table functions.

Functions are extracted from DuckDB duckdb_functions() introspection.

This file is AUTO-GENERATED by `{caller.as_posix()}`. DO NOT EDIT MANUALLY.
"""

from __future__ import annotations
from ._frame import LazyFrame
import duckdb
'''
