from enum import StrEnum, auto

import pyochain as pc

from .._utils import Builtins, CollectionsABC, DuckDB, Pql, Pyochain

type ContainerType = CollectionsABC | Builtins | Pyochain


def _set[T](*args: T) -> pc.Set[T]:
    return pc.Set(args)


class _Arg(StrEnum):
    ALLOW_PARALLEL = auto()
    PARALLEL = auto()
    FORCE_PARALLEL = auto()
    MULTITHREADED = auto()
    MORE_EXPRS = auto()
    MORE_PREDICATES = auto()
    INTERPOLATION = auto()


type IgnoredParams = pc.Dict[Pql, pc.Dict[str, pc.Set[str]]]
type IgnoredMembers = pc.Dict[Pql, pc.Set[str]]


IGNORED_MEMBERS: IgnoredMembers = pc.Dict({
    Pql.MODULE_FUNCTIONS: _set(
        "show_versions",
        "col",
        "Utf8",
        "StringCache",
        "Series",
        "CredentialProvider",
        "CredentialProviderAWS",
        "CredentialProviderAzure",
        "CredentialProviderFunction",
        "CredentialProviderFunctionReturn",
        "CredentialProviderGCP",
        "DataFrame",
        "enable_string_cache",
        "unregister_extension_type",
        "using_string_cache",
        "get_extension_type",
        "generate_temporary_column_name",
        "QueryOptFlags",
        "SQLContext",
        "PartitionBy",
        "GPUEngine",
        "Implementation",
        "FileProviderArgs",
        "Config",
        "Categorical",
        "is_ordered_categorical",
        "disable_string_cache",
        "Categories",
    ),
    Pql.LAZY_FRAME: _set("cache", "clear", "with_columns_seq", "select_seq"),
    Pql.SELECTORS: _set("Selector", "categorical"),
})

IGNORED_PARAMS: IgnoredParams = pc.Dict({
    Pql.LAZY_FRAME: pc.Dict.from_kwargs(
        sort=_set("maintain_order", _Arg.MULTITHREADED),
        join=_set(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
        join_asof=_set(_Arg.ALLOW_PARALLEL, _Arg.FORCE_PARALLEL),
        collect=_set("backend", "kwargs"),
        quantile=_set(_Arg.INTERPOLATION),
        select=_set(_Arg.MORE_EXPRS),
        filter=_set(_Arg.MORE_PREDICATES),
        with_columns=_set(_Arg.MORE_EXPRS),
    ),
    Pql.EXPR: pc.Dict.from_kwargs(
        sort_by=_set(_Arg.MULTITHREADED), quantile=_set(_Arg.INTERPOLATION)
    ),
    Pql.EXPR_STRUCT_NAME_SPACE: pc.Dict.from_kwargs(with_fields=_set(_Arg.MORE_EXPRS)),
    Pql.EXPR_LIST_NAME_SPACE: pc.Dict.from_kwargs(eval=_set(_Arg.PARALLEL)),
    Pql.EXPR_ARR_NAME_SPACE: pc.Dict.from_kwargs(eval=_set(_Arg.PARALLEL)),
    Pql.LAZY_GROUP_BY: pc.Dict.from_kwargs(
        agg=_set("more_aggs"),
        quantile=_set(_Arg.INTERPOLATION),
    ),
    Pql.MODULE_FUNCTIONS: pc.Dict.from_kwargs(
        all_horizontal=_set(_Arg.MORE_EXPRS),
        any_horizontal=_set(_Arg.MORE_EXPRS),
        max_horizontal=_set(_Arg.MORE_EXPRS),
        mean_horizontal=_set(_Arg.MORE_EXPRS),
        min_horizontal=_set(_Arg.MORE_EXPRS),
        sum_horizontal=_set(_Arg.MORE_EXPRS),
        when=_set(_Arg.MORE_PREDICATES),
    ),
})


class Status(StrEnum):
    """Status of a method comparison."""

    MISSING = auto()
    SIGNATURE_MISMATCH = auto()
    MATCH = auto()
    EXTRA = auto()


_SEQUENCE_TYPES: pc.Set[ContainerType] = _set(
    CollectionsABC.SEQUENCE, Builtins.LIST, Builtins.TUPLE, Pyochain.SEQ, Pyochain.VEC
)
_COLLECTION_TYPES: pc.Set[ContainerType] = _set(
    CollectionsABC.COLLECTION, Builtins.SET, Builtins.FROZENSET, *_SEQUENCE_TYPES
)
_ITERABLE_TYPES: pc.Set[ContainerType] = _set(
    CollectionsABC.ITERABLE, *_COLLECTION_TYPES
)
CONTAINER_SUPERTYPES: pc.Dict[CollectionsABC, pc.Set[ContainerType]] = pc.Dict({
    CollectionsABC.ITERABLE: _ITERABLE_TYPES,
    CollectionsABC.COLLECTION: _COLLECTION_TYPES,
    CollectionsABC.SEQUENCE: _SEQUENCE_TYPES,
})

_INTO_EXPR_COL_TYPES: pc.Set[str] = _set(
    Pql.INTO_EXPR_COLUMN,
    "ColumnNameOrSelector",
    Builtins.STR,
    Pql.EXPR,
    Pql.SQLEXPR,
    Pql.EXPR_HANDLER,
)

TYPE_SUPERTYPES: pc.Dict[str, pc.Set[str]] = pc.Dict({
    Pql.INTO_EXPR: _set(
        Pql.INTO_EXPR,
        "ColumnNameOrSelector",
        Builtins.INT,
        Builtins.FLOAT,
        Builtins.BOOL,
        Builtins.BYTES,
        Builtins.BYTEARRAY,
        Builtins.MEMORYVIEW,
        DuckDB.EXPRESSION,
        Pql.PYTHON_LITERAL,
        *_INTO_EXPR_COL_TYPES,
    ),
    Pql.INTO_EXPR_COLUMN: _INTO_EXPR_COL_TYPES,
    "ColumnNameOrSelector": _set("ColumnNameOrSelector", *_INTO_EXPR_COL_TYPES),
})
