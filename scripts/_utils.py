from __future__ import annotations

import collections.abc as collections_abc
import datetime
import decimal
import typing as ty
from dataclasses import dataclass
from enum import StrEnum, auto

import duckdb
from pyochain import Dict, Iter, Option, Seq, Set, Vec, option

import belugas as bl
from belugas import _core as bl_core, namespaces as bl_nm  # noqa: PLC2701


class KwordEnum(StrEnum):
    """Enum to easily manipulate python constructs as text, e.g. for code generation."""

    @classmethod
    def module(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def into_iter(cls) -> Iter[KwordEnum]:
        return Iter(cls)

    def of_type(self, *dtypes: str, has_ellipsis: bool = False) -> str:
        type_union = Iter(dtypes).join(" | ")
        if has_ellipsis:
            type_union = f"{type_union}, ..."
        return f"{self.value}[{type_union}]"

    def into_union(self, *args: str) -> str:
        return Iter(args).insert(self).join(" | ")


def get_attr(obj: object, name: str) -> Option[object]:
    """Safe getattr returning Option.

    Args:
        obj (object): The object to get the attribute from.
        name (str): The name of the attribute to get.

    Returns:
        Option[object]: An Option containing the attribute value if it exists, or None if it does not.
    """
    return option(getattr(obj, name, None))


@dataclass(slots=True)
class From[T: KwordEnum]:
    module: type[T]

    def import_(self, *names: T) -> str:
        return f"from {self.module.module()} import {Iter(names).map(lambda n: n.value).join(', ')}"


@dataclass(slots=True)
class Import[T: KwordEnum]:
    module: type[T]

    @ty.override
    def __str__(self) -> str:
        return f"import {self.module.module()}"

    def as_(self, alias: str) -> str:
        return f"import {self.module.module()} as {alias}"


class Dunders(KwordEnum):
    INIT = object.__init__.__name__
    CALL = object.__call__.__name__
    DEPRECATED = "__deprecated__"
    DOC = "__doc__"
    AND = "__and__"
    OR = "__or__"
    INVERT = "__invert__"


class Pql(KwordEnum):
    SELECTORS = auto()
    EXPR = bl.Expr.__name__
    INTO_EXPR = "IntoExpr"
    BLOB_LITERAL = "BlobLiteral"
    INTO_EXPR_COLUMN = "IntoExprColumn"
    PYTHON_LITERAL = "PythonLiteral"
    TRY_ITER = auto()
    INTO_DUCKDB = auto()
    INTO_DUCKDB_MAPPING = auto()
    CORE_HANDLER = bl_core.CoreHandler.__name__
    EXPR_HANDLER = bl_core.ExprHandler.__name__
    LAZY_FRAME = bl.LazyFrame.__name__
    LAZY_GROUP_BY = "LazyGroupBy"
    EXPR_STR_NAME_SPACE = bl_nm.ExprStringNameSpace.__name__
    EXPR_LIST_NAME_SPACE = bl_nm.ExprListNameSpace.__name__
    EXPR_STRUCT_NAME_SPACE = bl_nm.ExprStructNameSpace.__name__
    EXPR_NAME_NAME_SPACE = bl_nm.ExprNameNameSpace.__name__
    EXPR_ARR_NAME_SPACE = bl_nm.ExprArrayNameSpace.__name__
    EXPR_DT_NAME_SPACE = bl_nm.ExprDateTimeNameSpace.__name__
    MODULE_FUNCTIONS = "ModuleFunctions"
    SEQ_LITERAL = "SeqLiteral"
    DATA_TYPE = bl.DataType.__name__
    SCHEMA = "Schema"


class Pyochain(KwordEnum):
    OPTION = Option.__name__
    SEQ = Seq.__name__
    ITER = Iter.__name__
    VEC = Vec.__name__
    DICT = Dict.__name__
    SET = Set.__name__


class Builtins(KwordEnum):
    NONE = "None"
    LIST = list.__name__
    SET = set.__name__
    FROZENSET = frozenset.__name__
    TUPLE = tuple.__name__
    DICT = dict.__name__
    PROPERTY = property.__name__
    SELF = auto()
    CLS = auto()
    STR = str.__name__
    BOOL = bool.__name__
    INT = int.__name__
    FLOAT = float.__name__
    BYTES = bytes.__name__
    BYTEARRAY = bytearray.__name__
    MEMORYVIEW = memoryview.__name__


class DateTime(KwordEnum):
    TIME = datetime.time.__name__
    DATE = datetime.date.__name__
    DATETIME = datetime.datetime.__name__
    TIMEDELTA = datetime.timedelta.__name__


class DuckDB(KwordEnum):
    SQLTYPES = auto()
    EXPLAIN_TYPE = duckdb.ExplainType.__name__
    RENDER_MODE = duckdb.RenderMode.__name__
    EXPRESSION = duckdb.Expression.__name__
    RELATION = duckdb.DuckDBPyRelation.__name__


class Decimal(KwordEnum):
    DECIMAL = decimal.Decimal.__name__


class Typing(KwordEnum):
    TYPE_CHECKING = "TYPE_CHECKING"
    T = "T"
    ANY = ty.Any.__name__
    LITERAL = ty.Literal.__name__
    SELF = ty.Self.__name__
    UNION = ty.Union.__name__  # pyright: ignore[reportDeprecated]
    SUPPORTS_INT = ty.SupportsInt.__name__
    OVERLOAD = ty.overload.__name__
    CLASSVAR = ty.ClassVar.__name__


class CollectionsABC(KwordEnum):
    @classmethod
    @ty.override
    def module(cls) -> str:
        return "collections.abc"

    ITERABLE = collections_abc.Iterable.__name__
    SEQUENCE = collections_abc.Sequence.__name__
    COLLECTION = collections_abc.Collection.__name__
    CALLABLE = collections_abc.Callable.__name__
    MAPPING = collections_abc.Mapping.__name__


def set_pl_config() -> None:
    from polars.config import Config

    _ = (
        Config()
        .set_tbl_formatting("ASCII_MARKDOWN", rounded_corners=True)
        .set_tbl_hide_column_data_types(True)
        .set_tbl_hide_dataframe_shape(True)
        .set_fmt_str_lengths(1000)
        .set_tbl_rows(-1)
        .set_tbl_cols(-1)
    )
