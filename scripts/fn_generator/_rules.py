import builtins
import keyword
from dataclasses import dataclass, field

import polars as pl
import pyochain as pc
from sqlglot.parsers.duckdb import DuckDBParser

from .._utils import Builtins, Pql, Typing
from ._dtypes import Categories, DuckDbTypes

CONVERTER = pc.Iter(DuckDbTypes).map(lambda t: (t, t.into_py())).collect(dict)
"""DuckDB type -> Python type hint mapping."""

DK_FUNC_KEYS = pl.LazyFrame({"glot_name": tuple(DuckDBParser.FUNCTIONS)})  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
"""DuckDBParser.FUNCTIONS keys as a single-column LazyFrame."""

SHADOWERS = (
    Pql
    .into_iter()
    .chain(Typing, Builtins)
    .map(lambda s: s.value)
    .chain(dir(builtins), keyword.kwlist)
    .insert("l")
    .collect(pc.Set)
)
"""Names that should be renamed to avoid shadowing."""

RENAME_RULES = pc.Dict.from_ref({
    "list": "implode",
    "json": "json_parse",
    "map": "to_map",
    "kurtosis": "kurtosis_samp",
    "isnan": "is_nan",
    "isinf": "is_inf",
    "isfinite": "is_finite",
    "bool_and": "all",
    "bool_or": "any",
    "round": "round_from_zero",  # allow to expose round as parametrizable method
    "entropy": "entropy_shannon",  # allow to use entropy for polars aligned version
})
"""Explicit SQL function name -> generated Python method name mapping."""


SPECIAL_CASES = pc.Set({
    # "raw" operators
    "+",
    "-",
    "/",
    "*",
    "//",
    "%",
    "**",
    "&",
    "|",
    "^",
    "~",
    "&&",
    "||",
    "@",
    "^@",
    "@>",
    "<@",
    "<->",
    "<=>",
    "<<",
    ">>",
    "->>",
    "~~",
    "!~~",
    "~~*",
    "!~~*",
    "~~~",
    "!__postfix",
    "!",
    "…",
    # Primary operators, we want to handle them manually
    "add",
    "subtract",
    "multiply",
    "divide",
    "alias",  # conflicts with duckdb alias method
    # Need arg swapping
    "log",  # Need to swap argument order to take self.inner() as value and not as base
    "date_trunc",  # Need to swap argument order to take self.inner() as timestamp and not as precision
    "datetrunc",  # alias of date_trunc, same issue
    # Need to transform the expr input in a lambda in all cases, better to handle it manually
    "array_filter",
    "list_filter",
    "filter",
    # Generic functions that cause too much conflicts with other names
    "greatest",  # Has 5 categories, same behavior across thoses, no namespace needed
    "least",  # Has 5 categories, same behavior across thoses, no namespace needed
    "concat",  # too much conflict with list_concat, array_concat, etc..
    # sqlglot issues
    "xor",  # Actual match casing logic gives it `XOR` when really it should be `BitwiseXor`
    # overrides
    "quantile",  # Allow to make quantile a parametrizable method
})
"""Function to exclude by name, either because they require special handling or because they conflict with existing names."""
PREFIXES = pc.Set((
    "__",  # Internal functions
    "current_",  # Utility fns
    "has_",  # Utility fns
    "pg_",  # Postgres fns
    "icu_",  # timestamp extension
))
"""Functions to exclude by prefixes."""


def _rule[T](*args: T) -> pc.Seq[T]:
    return pc.Seq(args)


@dataclass(slots=True)
class NamespaceSpec:
    name: str
    doc: str
    prefixes: pc.Seq[str]
    strip_prefixes: pc.Seq[str]
    categories: pc.Seq[Categories] = field(default_factory=pc.Seq[Categories].new)
    explicit_names: pc.Seq[str] = field(default_factory=pc.Seq[str].new)


NAMESPACE_SPECS = pc.Seq((
    NamespaceSpec(
        name="ListFns",
        doc="Mixin providing auto-generated DuckDB list functions as methods.",
        prefixes=pc.Seq(("list_",)),
        categories=_rule(Categories.LIST),
        strip_prefixes=_rule("list_", "array_"),
    ),
    NamespaceSpec(
        name="StructFns",
        doc="Mixin providing auto-generated DuckDB struct functions as methods.",
        prefixes=pc.Seq(("struct_",)),
        categories=_rule(Categories.STRUCT),
        strip_prefixes=pc.Seq(("struct_",)),
    ),
    NamespaceSpec(
        name="RegexFns",
        doc="Mixin providing auto-generated DuckDB regex functions as methods.",
        prefixes=pc.Seq(("regexp_",)),
        categories=pc.Seq((Categories.REGEX,)),
        strip_prefixes=_rule("regexp_", "str_", "string_"),
    ),
    NamespaceSpec(
        name="StringFns",
        doc="Mixin providing auto-generated DuckDB string functions as methods.",
        prefixes=_rule("string_", "str_"),
        categories=_rule(Categories.STRING, Categories.TEXT_SIMILARITY),
        strip_prefixes=_rule("string_", "str_"),
        explicit_names=_rule("strftime", "strptime"),
    ),
    NamespaceSpec(
        name="DateTimeFns",
        doc="Mixin providing auto-generated DuckDB datetime functions as methods.",
        prefixes=_rule("date", "epoch", "iso", "time", "day", "month", "week", "year"),
        categories=_rule(Categories.TIMESTAMP, Categories.DATE),
        strip_prefixes=_rule("date_", "date"),
        explicit_names=_rule(
            "microsecond",
            "nanosecond",
            "millisecond",
            "second",
            "minute",
            "hour",
            "quarter",
            "decade",
            "century",
            "millennium",
            "era",
            "julian",
            "last_day",
            "to_timestamp",
            "to_microseconds",
            "to_milliseconds",
            "to_seconds",
            "to_minutes",
            "to_hours",
            "to_days",
            "to_weeks",
            "to_months",
            "to_quarters",
            "to_years",
            "to_decades",
            "to_centuries",
            "to_millennia",
            "make_date",
            "make_date_month_day",
            "make_time",
            "make_timestamp",
            "make_timestamp_ms",
            "make_timestamp_ns",
            "make_timestamptz",
            "normalized_interval",
        ),
    ),
    NamespaceSpec(
        name="ArrayFns",
        doc="Mixin providing auto-generated DuckDB array functions as methods.",
        prefixes=pc.Seq(("array_",)),
        categories=pc.Seq((Categories.ARRAY,)),
        strip_prefixes=pc.Seq(("array_",)),
    ),
    NamespaceSpec(
        name="JsonFns",
        doc="Mixin providing auto-generated DuckDB JSON functions as methods.",
        prefixes=pc.Seq(("json_",)),
        strip_prefixes=pc.Seq(("json_",)),
    ),
    NamespaceSpec(
        name="MapFns",
        doc="Mixin providing auto-generated DuckDB map functions as methods.",
        prefixes=pc.Seq(("map_",)),
        strip_prefixes=pc.Seq(("map_",)),
    ),
    NamespaceSpec(
        name="EnumFns",
        doc="Mixin providing auto-generated DuckDB enum functions as methods.",
        prefixes=pc.Seq(("enum_",)),
        strip_prefixes=pc.Seq(("enum_",)),
    ),
    NamespaceSpec(
        name="GeoSpatialFns",
        doc="Mixin providing auto-generated DuckDB geospatial functions as methods.",
        categories=pc.Seq((Categories.GEOMETRY,)),
        prefixes=pc.Seq(("st_", "ST_")),
        strip_prefixes=pc.Seq(("st_", "ST_")),
    ),
))
"""Namespace metadata and function prefixes."""
