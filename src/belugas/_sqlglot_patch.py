from __future__ import annotations

from collections.abc import Callable

from pyochain import Iter
from sqlglot import Dialect, exp, parser
from sqlglot.dialects.dialect import build_formatted_time, build_regexp_extract
from sqlglot.dialects.duckdb import DuckDB
from sqlglot.generators.duckdb import DuckDBGenerator
from sqlglot.parsers.duckdb import DuckDBParser

type FuncRegistery = dict[str, Callable[..., exp.Expr]]
type BindedFn = Callable[[list[exp.Expr]], exp.Expr]


_DUCKDB_DATATYPE_SQL = DuckDBGenerator.TRANSFORMS[exp.DataType]


def _patched_datatype_sql(self: DuckDBGenerator, expression: exp.DataType) -> str:
    """Render LIST dtypes with bracket syntax for DuckDB CAST compatibility.

    Root cause:
        DuckDB modern versions REJECT the ``CAST(x AS LIST(T))`` syntax in CAST contexts:
        ``Parser Error: Expected a constant as type modifier``.

        However, DuckDB ACCEPTS both ``T[]`` (dynamic array) and ``T[N]`` (fixed-size array)
        syntax for all array/list types.

        SQLGlot's DuckDBGenerator emits ``LIST(T)`` for ``exp.DType.LIST`` by default,
        causing belugas casts to fail at parse time.

    Solution:
        Patch the DuckDB generator to emit ``T[]`` (bracket syntax) instead of ``LIST(T)``.
        This is semantically equivalent but compatible with modern DuckDB parser.

    Concrete example:
        Before (fails in DuckDB):
            CAST("col" AS LIST(USMALLINT))
            ✗ Parser Error: Expected a constant as type modifier

        After (works):
            CAST("col" AS USMALLINT[])
            ✓ Success

    Related parsing behavior:
        When DuckDB/sqlglot parse ``T[]`` syntax, it produces ``DType.ARRAY`` (not ``LIST``).
        So from_sql() in datatypes.py normalizes ``ARRAY without values`` back to ``List``.

    Returns:
        str: DuckDB-compatible SQL for the provided data type expression.
    """
    if expression.is_type(exp.DType.LIST):
        return f"{self.expressions(expression, flat=True)}[]"

    return _DUCKDB_DATATYPE_SQL(self, expression)


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
            return Iter(fields).fold(source, _object_insert)
        case _:
            return exp.Struct(expressions=args)


_build_hex = _bind_dialect(parser.build_hex)


def _patched_from_global() -> FuncRegistery:
    r"""Those functions need to be patched because they expect a `dialect` argument that is not passed.

    ```shell
    FAILED tests/test_exprs.py::test_uint_only_simple - TypeError: build_logarithm() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_to_titlecase - TypeError: __call__() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_zfill[4] - TypeError: __call__() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_zfill[5] - TypeError: __call__() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_zfill[10] - TypeError: __call__() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_encode[hex] - TypeError: build_hex() missing required argument 'dialect' (pos 2)
    ```

    Returns:
        FuncRegistery: The patched functions.
    """
    return {
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


def _patched_from_duckdb() -> FuncRegistery:
    r"""Those functions need to be patched because they expect a `dialect` argument that is not passed.

    ```shell
    FAILED tests/test_str_namespace.py::test_to_titlecase - TypeError: build_regexp_extract.<locals>._builder() missing 1 required positional argument: 'dialect'
    FAILED tests/test_str_namespace.py::test_extract_all - TypeError: build_regexp_extract.<locals>._builder() missing 1 required positional argument: 'dialect'
    FAILED tests/test_str_namespace.py::test_extract - TypeError: build_regexp_extract.<locals>._builder() missing 1 required positional argument: 'dialect'
    FAILED tests/test_str_namespace.py::test_find - TypeError: build_regexp_extract.<locals>._builder() missing 1 required positional argument: 'dialect'
    FAILED tests/test_str_namespace.py::test_json_path_match[json_path0] - TypeError: __call__() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_json_path_match[json_path1] - TypeError: __call__() missing required argument 'dialect' (pos 2)
    FAILED tests/test_str_namespace.py::test_count_matches[False-a] - TypeError: build_regexp_extract.<locals>._builder() missing 1 required positional argument: 'dialect'
    FAILED tests/test_str_namespace.py::test_count_matches[False-\\d+] - TypeError: build_regexp_extract.<locals>._builder() missing 1 required positional argument: 'dialect'
    ```

    Returns:
        FuncRegistery: The patched functions.
    """
    return {
        "JSON_EXTRACT_PATH": _extract_json_with_path(exp.JSONExtract),
        "JSON_EXTRACT_STRING": _extract_json_with_path(exp.JSONExtractScalar),
        "LIST_CONCAT": _bind_dialect(parser.build_array_concat),
        "REGEXP_EXTRACT": _regexp_extract(exp.RegexpExtract),
        "REGEXP_EXTRACT_ALL": _regexp_extract(exp.RegexpExtractAll),
        "STRFTIME": _bind_dialect(build_formatted_time(exp.TimeToStr)),
        "STRPTIME": _bind_dialect(build_formatted_time(exp.StrToTime)),
    }


def _missing_from_glot() -> FuncRegistery:
    return {
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
        "LIST_INDEXOF": exp.ArrayPosition.from_arg_list,
        "LIST_INTERSECT": lambda args: exp.ArrayIntersect(expressions=args),
        "LIST_PACK": lambda args: exp.Array(expressions=args),
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


DUCKDB_FUNCTIONS: FuncRegistery = DuckDBParser.FUNCTIONS | {
    **_patched_from_global(),
    **_patched_from_duckdb(),
    **_missing_from_glot(),
}

DuckDBParser.FUNCTIONS = DUCKDB_FUNCTIONS

DuckDBGenerator.TRANSFORMS |= {
    exp.DataType: _patched_datatype_sql,
}


def _add_to_meta(
    *exprs: type[exp.Expr], dtype: exp.DType
) -> dict[type[exp.Expr], dict[str, exp.DType]]:
    return Iter(exprs).map(lambda e: (e, {"returns": dtype})).collect(dict)


DuckDB.EXPRESSION_METADATA |= {
    # Ranking/window functions are not typed in DuckDB's metadata, so Window(this=...) can
    # propagate UNKNOWN through downstream schema inference.
    **_add_to_meta(
        exp.DenseRank, exp.Ntile, exp.Rank, exp.RowNumber, dtype=exp.DType.BIGINT
    ),
    **_add_to_meta(exp.CumeDist, exp.PercentRank, dtype=exp.DType.DOUBLE),
    exp.ArrayDistinct: {"annotator": lambda self, e: self._annotate_by_args(e, "this")},
}
