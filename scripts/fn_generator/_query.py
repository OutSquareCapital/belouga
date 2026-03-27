import re
from collections.abc import Iterable

import polars as pl
import pyochain as pc
from sqlglot import exp

from .._utils import Pql, Typing
from ._dtypes import DuckDbTypes, FuncTypes
from ._format import to_func
from ._rules import (
    CONVERTER,
    NAMESPACE_SPECS,
    PREFIXES,
    RENAME_RULES,
    SHADOWERS,
    SPECIAL_CASES,
)
from ._schemas import DuckCols, ParamLens, ParamLists, Params, PyCols
from ._str_builder import EMPTY_STR, format_kwords


def run_qry(lf: pl.LazyFrame) -> pl.LazyFrame:
    py = PyCols()
    params = Params()
    dk = DuckCols()
    return (
        lf.pipe(_filters, dk)
        .with_columns(
            dk.parameter_types.list.eval(pl.element().fill_null(DuckDbTypes.ANY)),
            *dk.parameters.list.len().pipe(
                lambda expr_len: (
                    expr_len.alias("sig_param_count"),
                    expr_len.min().over(dk.function_name).alias("min_params_per_fn"),
                    expr_len.min()
                    .over(dk.function_name, dk.categories, dk.description)
                    .alias("min_params_per_fn_cat_desc"),
                )
            ),
        )
        .pipe(
            lambda lf: lf.join(lf.pipe(_alias_map, dk), on=dk.function_name, how="left")
        )
        .pipe(
            lambda lf: lf.join(
                lf.pipe(_py_name_map, dk, params.lens),
                on=[dk.function_name, dk.categories, dk.description],
                how="left",
            )
        )
        .pipe(lambda lf: lf.join(lf.pipe(_glot_name_map, dk), on=dk.function_name))
        .with_columns(
            dk.function_name.alias(py.sql_name.meta.output_name()),
            dk.function_name.replace_strict(
                RENAME_RULES, default=dk.function_name, return_dtype=pl.String
            ).alias(py.raw_name.meta.output_name()),
        )
        .with_columns(dk.categories.pipe(_namespace_specs, py.raw_name))
        .explode("namespace")
        .with_columns(_py_name(py.raw_name, py), py.namespace.pipe(_return_type))
        .with_row_index("sig_id")
        .explode("parameters", "parameter_types")
        .with_columns(
            pl.int_range(pl.len()).over("sig_id").alias("param_idx"),
            dk.parameters.pipe(_to_param_names),
            dk.varargs.pipe(_convert_duckdb_type_to_python)
            .pipe(_make_type_union)
            .alias("py_varargs_type"),
        )
        .group_by(py.namespace, py.name, params.idx, maintain_order=True)
        .agg(
            pl.all().exclude("parameter_types").drop_nulls().first(),
            dk.parameter_types.pipe(_into_union),
            dk.parameter_types.pipe(_convert_duckdb_type_to_python)
            .pipe(_into_union)
            .pipe(_make_type_union)
            .alias("py_types"),
        )
        .group_by(py.namespace, py.name, maintain_order=True)
        .agg(
            pl.all().exclude("param_names").first(),
            *_joined_parts(params, py.types, dk.parameter_types),
        )
        .select(
            py.namespace,
            py.name,
            pl.col("has_params").pipe(to_func, py, ParamLists(), dk),
        )
        .sort(py.namespace, py.name)
    )


def _filters(lf: pl.LazyFrame, dk: DuckCols) -> pl.LazyFrame:
    """First-step filter to remove unwanted functions."""
    return lf.select(dk.to_dict().keys()).filter(
        dk.function_type.is_in(FuncTypes.unwanted()).not_(),
        dk.parameters.list.len().eq(0).and_(dk.varargs.is_null()).not_(),  # literals
        dk.function_name.is_in(SPECIAL_CASES).not_(),
        *PREFIXES.iter().map(
            lambda prefix: dk.function_name.str.starts_with(prefix).not_()
        ),
    )


def _return_type(namespace: pl.Expr) -> pl.Expr:
    return (
        pl.when(namespace.is_not_null())
        .then(pl.lit(Typing.T))
        .otherwise(pl.lit(Typing.SELF))
        .alias("self_type")
    )


def _py_name(raw_name: pl.Expr, py: PyCols) -> pl.Expr:
    return (
        pl.concat_str(
            raw_name,
            pl.when(py.suffixes.is_not_null()).then(
                pl.concat_str(pl.lit("_"), py.suffixes)
            ),
            ignore_nulls=True,
        )
        .str.to_lowercase()
        .str.replace(
            (
                NAMESPACE_SPECS.iter()
                .flat_map(lambda spec: spec.strip_prefixes.iter().map(re.escape))
                .into(lambda p: f"^(?:{p.join('|')})")
            ),
            EMPTY_STR,
            literal=False,
        )
        .alias("py_name")
    )


def _alias_map(lf: pl.LazyFrame, dk: DuckCols) -> pl.LazyFrame:
    """Map of `function_name` to list of aliases.

    Alias root is determined by taking the first value between `function_name` and `alias_of` that is not null.
    Then all other `function_name`s that share the same `alias_root` are considered aliases of each other.
    """
    return (
        lf.select(
            dk.function_name,
            dk.alias_of,
            pl.coalesce(dk.alias_of, dk.function_name).alias("alias_root"),
        )
        .pipe(
            lambda lf: lf.join(
                lf.group_by("alias_root").agg(
                    dk.function_name.sort().alias("alias_group")
                ),
                on="alias_root",
                how="left",
            )
        )
        .select(
            dk.function_name,
            pl.col("alias_group")
            .list.set_difference(pl.concat_list(dk.function_name))
            .alias("aliases"),
        )
    )


def _glot_name_map(lf: pl.LazyFrame, dk: DuckCols) -> pl.LazyFrame:
    glot_name = pl.col("glot_name")
    return (
        lf.select(
            dk.function_name,
            pl.coalesce(dk.alias_of, dk.function_name).alias("alias_root"),
        )
        .unique()
        .join(
            pc.Iter(exp.FUNCTION_BY_NAME.items())
            .map_star(lambda sql_name, fn: (sql_name.lower(), fn.__name__))
            .into(
                pl.LazyFrame,
                schema={"function_name": pl.String, "glot_name": pl.String},
            ),
            on="function_name",
            how="left",
        )
        .select(
            dk.function_name,
            pl.when(pl.col("alias_root").is_not_null())
            .then(glot_name.drop_nulls().first().over("alias_root"))
            .otherwise(glot_name)
            .alias("glot_name"),
        )
    )


def _into_union(expr: pl.Expr) -> pl.Expr:
    return expr.filter(expr.ne(EMPTY_STR)).unique().sort().str.join(" | ")


def _joined_parts(
    params: Params, py_union: pl.Expr, params_union: pl.Expr
) -> Iterable[pl.Expr]:
    cond = params.idx.ge(params.lens.min_params_per_fn)

    return (
        params.names.alias("param_names_list"),
        params.names.len().alias("has_params"),
        format_kwords(
            "{param_name}: {py_type}{union}",
            param_name=params.names,
            py_type=py_union,
            union=pl.when(cond).then(pl.lit(" | None = None")),
            ignore_nulls=True,
        ).alias("param_sig_list"),
        format_kwords(
            "            {param_name} ({py_type}{union}): `{dk_type}` expression",
            param_name=params.names,
            py_type=py_union,
            union=pl.when(cond).then(pl.lit(" | None")),
            dk_type=params_union,
            ignore_nulls=True,
        ).alias("param_doc_list"),
    )


def _make_type_union(py_type: pl.Expr) -> pl.Expr:
    into_expr_col = pl.lit(Pql.INTO_EXPR_COLUMN)
    return (
        pl.when(py_type.eq(EMPTY_STR))
        .then(into_expr_col)
        .when(py_type.str.contains(_token_pattern(Pql.INTO_EXPR)))
        .then(pl.lit(Pql.INTO_EXPR))
        .when(py_type.str.contains(_token_pattern(Pql.INTO_EXPR_COLUMN)))
        .then(py_type)
        .otherwise(
            format_kwords(
                "{self_type} | {py_type}", self_type=into_expr_col, py_type=py_type
            )
        )
        .map_elements(_simplify_generated_union, return_dtype=pl.String)
    )


def _token_pattern(token: str) -> str:
    return rf"(^|\s\|\s){re.escape(token)}($|\s\|\s)"


def _simplify_generated_union(type_hint: str) -> str:
    tokens = (
        pc.Iter(type_hint.split("|"))
        .map(str.strip)
        .filter(lambda type_name: type_name != "")
        .collect()
    )
    if tokens.any(lambda type_name: type_name == Pql.INTO_EXPR):
        return Pql.INTO_EXPR.value

    has_into_expr_column = tokens.any(
        lambda type_name: type_name == Pql.INTO_EXPR_COLUMN
    )
    return (
        tokens.iter()
        .filter(
            lambda type_name: (
                type_name != "str" if has_into_expr_column else type_name != ""
            )
        )
        .join(" | ")
    )


def _namespace_specs(cats: pl.Expr, fn_name: pl.Expr) -> pl.Expr:
    return (
        NAMESPACE_SPECS.iter()
        .map(
            lambda spec: pl.when(
                spec.prefixes.iter()
                .map(fn_name.str.starts_with)
                .into(pl.any_horizontal)
                .or_(fn_name.is_in(spec.explicit_names))
            ).then(pl.lit(spec.name))
        )
        .into(pl.coalesce)
        .pipe(
            lambda prefix_ns: pl.when(prefix_ns.is_not_null()).then(
                pl.concat_list(prefix_ns)
            )
        )
        .otherwise(
            NAMESPACE_SPECS.iter()
            .map(
                lambda spec: pl.when(
                    spec.categories.iter()
                    .map(lambda c: c.value)
                    .into(lambda x: cats.list.set_intersection(x.collect(tuple)))
                    .list.len()
                    .gt(0)
                ).then(pl.lit(spec.name))
            )
            .into(pl.concat_list)
        )
        .list.drop_nulls()
        .alias("namespace")
    )


def _convert_duckdb_type_to_python(param_type: pl.Expr) -> pl.Expr:
    return param_type.replace_strict(CONVERTER, return_dtype=pl.String)


def _to_param_names(params: pl.Expr) -> pl.Expr:
    return (
        params.str.strip_chars("'\"[]")
        .str.replace(r"\(.*$", EMPTY_STR)
        .str.replace_all(r"\.\.\.", EMPTY_STR)
        .str.to_lowercase()
        .pipe(
            lambda expr: (
                pl.when(expr.is_in(SHADOWERS))
                .then(pl.concat_str(expr, pl.lit("_arg")))
                .otherwise(expr)
            )
        )
        .alias("param_names")
    )


def _py_name_map(lf: pl.LazyFrame, dk: DuckCols, p_lens: ParamLens) -> pl.LazyFrame:
    return lf.filter(
        dk.description.n_unique()
        .over(dk.function_name, dk.categories)
        .gt(1)
        .and_(p_lens.min_params_per_fn_cat_desc.gt(p_lens.min_params_per_fn))
        .and_(p_lens.sig_param_count.eq(p_lens.min_params_per_fn_cat_desc))
    ).select(
        dk.function_name,
        dk.categories,
        dk.description,
        pl.when(p_lens.min_params_per_fn_cat_desc.eq(p_lens.min_params_per_fn))
        .then(dk.parameters)
        .otherwise(
            dk.parameters.list.slice(
                p_lens.min_params_per_fn,
                p_lens.sig_param_count.sub(p_lens.min_params_per_fn),
            )
        )
        .list.join("_")
        .alias("py_suffixes"),
    )
