from dataclasses import asdict, dataclass, field
from typing import final

import polars as pl
from pyochain import Dict, Iter

from ._dtypes import CATEGORY_TYPES, DTYPES, FUNC_TYPES, SchemaName, Stability


def schema(cls: type[object]) -> pl.Schema:
    """Simple decorator for creating Polars Schema from class attributes.

    Args:
        cls (type[object]): The class to create a schema from.

    Returns:
        pl.Schema: A Polars Schema created from the class attributes.
    """

    def _is_polars_dtype(_k: str, v: object) -> bool:
        return isinstance(v, (type, pl.DataType)) and (
            isinstance(v, pl.DataType) or issubclass(v, pl.DataType)
        )

    return Iter(cls.__dict__.items()).filter_star(_is_polars_dtype).collect(pl.Schema)


@final
@schema
class TableSchema:
    """Schema for DuckDB functions table."""

    database_name = pl.String
    database_oid = pl.UInt8
    schema_name = pl.Enum(SchemaName)
    function_name = pl.String
    alias_of = pl.String()
    function_type = FUNC_TYPES
    description = pl.String()
    """Only present for scalar and aggregate functions."""
    comment = pl.String()
    tags = pl.List(pl.Struct({"key": pl.String(), "value": pl.String()}))
    return_type = DTYPES
    parameters = pl.List(pl.String)
    parameter_types = pl.List(DTYPES)
    varargs = DTYPES
    macro_definition = pl.String()
    has_side_effects = pl.Boolean()
    internal = pl.Boolean
    function_oid = pl.UInt16
    examples = pl.List(pl.String)
    stability = pl.Enum(Stability)
    categories = pl.List(CATEGORY_TYPES)


@dataclass(slots=True)
class ParamLens:
    sig_param_count: pl.Expr = field(default=pl.col("sig_param_count"))
    min_params_per_fn: pl.Expr = field(default=pl.col("min_params_per_fn"))
    min_params_per_fn_cat_desc: pl.Expr = field(
        default=pl.col("min_params_per_fn_cat_desc")
    )


@dataclass(slots=True)
class PyCols:
    sql_name: pl.Expr = field(default=pl.col("sql_name"))
    raw_name: pl.Expr = field(default=pl.col("raw_py_name"))
    glot_name: pl.Expr = field(default=pl.col("glot_name"))
    expr_builder: pl.Expr = field(default=pl.col("expr_builder"))
    expr_name: pl.Expr = field(default=pl.col("expr_name"))
    namespace: pl.Expr = field(default=pl.col("namespace"))
    name: pl.Expr = field(default=pl.col("py_name"))
    types: pl.Expr = field(default=pl.col("py_types"))
    self_type: pl.Expr = field(default=pl.col("self_type"))
    suffixes: pl.Expr = field(default=pl.col("py_suffixes"))
    aliases: pl.Expr = field(default=pl.col("aliases"))
    varargs_type: pl.Expr = field(default=pl.col("py_varargs_type"))


@dataclass(slots=True)
class Params:
    names: pl.Expr = field(default=pl.col("param_names"))
    idx: pl.Expr = field(default=pl.col("param_idx"))
    lens: ParamLens = field(default_factory=ParamLens)


@dataclass(slots=True)
class ParamLists:
    signatures: pl.Expr = field(default=pl.col("param_sig_list"))
    docs: pl.Expr = field(default=pl.col("param_doc_list"))
    names: pl.Expr = field(default=pl.col("param_names_list"))


@dataclass(slots=True)
class DuckCols:
    function_name: pl.Expr = field(default=pl.col("function_name"))
    function_type: pl.Expr = field(default=pl.col("function_type"))
    description: pl.Expr = field(default=pl.col("description"))
    categories: pl.Expr = field(default=pl.col("categories"))
    examples: pl.Expr = field(default=pl.col("examples"))
    varargs: pl.Expr = field(default=pl.col("varargs"))
    alias_of: pl.Expr = field(default=pl.col("alias_of"))
    parameters: pl.Expr = field(default=pl.col("parameters"))
    parameter_types: pl.Expr = field(default=pl.col("parameter_types"))

    def to_dict(self) -> Dict[str, pl.Expr]:
        return Dict.from_ref(asdict(self))
