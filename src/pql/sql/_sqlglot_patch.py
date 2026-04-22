from collections.abc import Callable

import pyochain as pc
from sqlglot import Dialect, exp, parser
from sqlglot.dialects.dialect import build_regexp_extract
from sqlglot.dialects.duckdb import DuckDB
from sqlglot.parsers.duckdb import DuckDBParser

type FuncRegistery = dict[str, Callable[..., exp.Expr]]
type BindedFn = Callable[[list[exp.Expr]], exp.Expr]


def _regexp_extract(expr: type[exp.Expr]) -> BindedFn:
    return _bind_dialect(build_regexp_extract(expr))


def _extract_json_with_path(expr: type[exp.Expr]) -> BindedFn:
    return _bind_dialect(parser.build_extract_json_with_path(expr))


def _bind_dialect(
    builder: Callable[[list[exp.Expr], Dialect], exp.Expr],
) -> BindedFn:
    dialect = DuckDB()

    def f(args: list[exp.Expr]) -> exp.Expr:
        return builder(args, dialect)

    return f


def _object_insert(source: exp.Expr, field: exp.Expr) -> exp.ObjectInsert:
    match field:
        case exp.PropertyEQ(this=exp.Expr() as key, expression=exp.Expr() as value):
            return exp.ObjectInsert(this=source, key=key, value=value)
        case exp.Alias(this=exp.Expr() as value) as alias:
            key = exp.to_identifier(alias.output_name)
            return exp.ObjectInsert(this=source, key=key, value=value)
        case _:
            key = exp.to_identifier(field.output_name)
            return exp.ObjectInsert(this=source, key=key, value=field.unalias())


def _struct_insert(args: list[exp.Expr]) -> exp.Expr:
    match args:
        case []:
            return exp.Struct(expressions=[])
        case [exp.Expr() as source, *fields]:
            return pc.Iter(fields).fold(source, _object_insert)
        case _:
            return exp.Struct(expressions=args)


_build_hex = _bind_dialect(parser.build_hex)
_PATCHED_FROM_GLOBAL: FuncRegistery = {
    "HEX": _build_hex,
    "TO_HEX": _build_hex,
    "LOG": _bind_dialect(parser.build_logarithm),
    "CONCAT": _bind_dialect(
        lambda args, dialect: exp.Concat(
            expressions=args,
            safe=not dialect.STRICT_STRING_CONCAT,
            coalesce=dialect.CONCAT_COALESCE,
        )
    ),
}
_PATCHED_FROM_DUCKDB: FuncRegistery = {
    "JSON_EXTRACT_PATH": _extract_json_with_path(exp.JSONExtract),
    "JSON_EXTRACT_STRING": _extract_json_with_path(exp.JSONExtractScalar),
    "LIST_CONCAT": _bind_dialect(parser.build_array_concat),
    "REGEXP_EXTRACT": _regexp_extract(exp.RegexpExtract),
    "REGEXP_EXTRACT_ALL": _regexp_extract(exp.RegexpExtractAll),
}
_MISSING_FROM_GLOT: FuncRegistery = {
    "ARBITRARY": exp.First.from_arg_list,
    "ARRAY_APPLY": exp.Transform.from_arg_list,
    "ARRAY_HAS_ANY": exp.ArrayOverlaps.from_arg_list,
    "ARRAY_INDEXOF": exp.ArrayPosition.from_arg_list,
    "ARRAY_REDUCE": exp.Reduce.from_arg_list,
    "ARRAY_TRANSFORM": exp.Transform.from_arg_list,
    "BASE64": exp.ToBase64.from_arg_list,
    "BIN": exp.ToBinary.from_arg_list,
    "FROM_HEX": exp.Unhex.from_arg_list,
    "LIST_APPLY": exp.Transform.from_arg_list,
    "LIST_CAT": _bind_dialect(parser.build_array_concat),
    "LIST_DISTINCT": exp.ArrayDistinct.from_arg_list,
    "LIST_INDEXOF": exp.ArrayPosition.from_arg_list,
    "LIST_INTERSECT": lambda args: exp.ArrayIntersect(expressions=args),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    "LIST_PACK": lambda args: exp.Array(expressions=args),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    "LIST_POSITION": exp.ArrayPosition.from_arg_list,
    "LIST_REDUCE": exp.Reduce.from_arg_list,
    "LIST_SLICE": exp.ArraySlice.from_arg_list,
    "MEAN": exp.Avg.from_arg_list,
    "ORD": exp.Unicode.from_arg_list,
    "POSITION": exp.StrPosition.from_arg_list,
    "REGEXP_SPLIT_TO_ARRAY": exp.RegexpSplit.from_arg_list,
    "STRUCT_INSERT": _struct_insert,
    "SUFFIX": exp.EndsWith.from_arg_list,
}
DUCKDB_FUNCTIONS: FuncRegistery = {
    **DuckDBParser.FUNCTIONS,  # pyright: ignore[reportUnknownMemberType]$
    **_PATCHED_FROM_GLOBAL,
    **_PATCHED_FROM_DUCKDB,
    **_MISSING_FROM_GLOT,
}
DuckDBParser.FUNCTIONS = DUCKDB_FUNCTIONS
