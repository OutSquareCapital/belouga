import polars as pl
from pyochain import Dict

import pql
from pql.sql._sqlglot_patch import DUCKDB_FUNCTIONS

from .fn_generator._query import (
    DuckCols,  # pyright: ignore[reportPrivateLocalImportUsage]
    _filters,  # pyright: ignore[reportPrivateUsage]
)


def check_missing_sqlglot() -> None:
    function_name = pl.col("function_name")
    alias_of = pl.col("alias_of")
    alias_root = pl.col("alias_root")
    all_aliases = pl.col("all_aliases")
    other_aliases = pl.col("other_aliases")
    known_function_names = pl.col("known_function_names")
    dk_func_keys = pl.LazyFrame(
        Dict.from_ref(DUCKDB_FUNCTIONS)
        .iter()
        .map(str.upper)
        .collect()
        .into(lambda x: pl.Series("glot_name", x))
    )

    return (
        pql.meta.functions()
        .collect()
        .lazy()
        .pipe(_filters, DuckCols())
        .select(
            function_name.str.to_uppercase(),
            pl.coalesce(alias_of, function_name).str.to_uppercase().alias("alias_root"),
        )
        .group_by(alias_root)
        .agg(function_name.unique().sort().alias("all_aliases"))
        .with_columns(all_aliases.alias("function_name"))
        .explode("function_name")
        .join(dk_func_keys, left_on=function_name, right_on="glot_name", how="anti")
        .drop("alias_root")
        .with_columns(
            all_aliases.list.set_difference(pl.concat_list(function_name)).alias(
                "other_aliases"
            )
        )
        .pipe(
            lambda lf: lf.join(
                lf.select(
                    function_name.unique()
                    .sort()
                    .implode()
                    .alias("known_function_names")
                ),
                how="cross",
            )
        )
        .select(
            function_name,
            other_aliases.list.set_intersection(known_function_names).alias(
                "absent_aliases"
            ),
            other_aliases.list.set_difference(known_function_names).alias(
                "present_aliases"
            ),
        )
        .sort(function_name)
        .filter(pl.col("absent_aliases").list.len().gt(0))
        .explode("present_aliases")
        .collect()
        .show(None)
    )
