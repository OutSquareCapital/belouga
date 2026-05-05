from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, override

import duckdb
from pygments import token
from pygments.lexers.sql import SqlLexer  # pyright: ignore[reportMissingTypeStubs]
from pyochain import Dict, Iter, Set, Some, Vec
from rich.console import Console
from rich.syntax import Syntax

from . import meta

if TYPE_CHECKING:
    from pygments.token import (
        _TokenType as TokenType,  # pyright: ignore[reportPrivateUsage]
    )
    from sqlglot import exp

    from ._frame import LazyFrame
    from .typing import Themes

CONSOLE = Console()
DUCK_PYGMENT_MAP = Dict.from_ref({
    duckdb.token_type.identifier: token.Name,
    duckdb.token_type.keyword: token.Keyword,
    duckdb.token_type.string_const: token.String,
    duckdb.token_type.numeric_const: token.Number,
    duckdb.token_type.comment: token.Comment,
    duckdb.token_type.operator: token.Operator,
})


type ProcessedToken = tuple[int, TokenType, str]


class DuckDbSqlLexer(SqlLexer):
    @override
    def get_tokens_unprocessed(self, text: str) -> Iter[ProcessedToken]:  # pyright: ignore[reportIncompatibleMethodOverride]
        duck_tokens = Dict(duckdb.tokenize(text))
        process = partial(self._process, duck_tokens)
        return Iter(super().get_tokens_unprocessed(text)).map_star(process)

    def _process(  # noqa: PLR6301
        self,
        duck_tokens: Dict[int, duckdb.token_type],
        pos: int,
        tokentype: TokenType,
        token_text: str,
    ) -> ProcessedToken:
        match duck_tokens.get_item(pos):
            case Some(duckdb.token_type.identifier) if token_text in FUNCTIONS:
                return (pos, token.Name.Function, token_text)
            case Some(duckdb.token_type.keyword) if token_text in DTYPES:
                return (pos, token.Name.Builtin, token_text)
            case Some(duck_type):
                return (
                    pos,
                    DUCK_PYGMENT_MAP.get_item(duck_type).unwrap_or(tokentype),
                    token_text,
                )
            case _:
                return (pos, tokentype, token_text)


def _get_dtypes() -> Set[str]:
    return (
        meta
        .types()
        .pipe(_get_names, "type_name")
        .union(meta.types().pipe(_get_names, "logical_type"))
    )


def _get_functions() -> Set[str]:
    return meta.functions().pipe(_get_names, "function_name")


def _get_names(lf: LazyFrame, col_name: str) -> Set[str]:
    return lf.select(col_name).fetch_all().iter().flatten().collect(Set)


DTYPES = _get_dtypes()
FUNCTIONS = _get_functions()
SYNTAX = partial(Syntax, lexer=DuckDbSqlLexer(), background_color="default")


@dataclass(slots=True)
class ParsedQuery:
    query: exp.Query

    def sql(self, *, pretty: bool = False) -> str:
        return self.query.sql(dialect="duckdb", pretty=pretty)

    def show(self, theme: Themes = "github-dark", *, pretty: bool = True) -> None:
        return CONSOLE.print(SYNTAX(self.sql(pretty=pretty), theme=theme))

    def tokenize(self) -> Vec[tuple[int, duckdb.token_type]]:
        return Vec.from_ref(duckdb.tokenize(self.sql()))
