from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Self, override

import duckdb
import pyochain as pc
import sqlparse
from pygments import token
from pygments.lexers.sql import SqlLexer  # pyright: ignore[reportMissingTypeStubs]
from rich.console import Console
from rich.syntax import Syntax
from sqlparse.lexer import Lexer

from . import meta

if TYPE_CHECKING:
    from pygments.token import (
        _TokenType as TokenType,  # pyright: ignore[reportPrivateUsage]  # noqa: PLC2701
    )

    from ._frame import LazyFrame
    from ._typing import Themes

CONSOLE = Console()
DUCK_PYGMENT_MAP = pc.Dict.from_ref({
    duckdb.token_type.identifier: token.Name,
    duckdb.token_type.keyword: token.Keyword,
    duckdb.token_type.string_const: token.String,
    duckdb.token_type.numeric_const: token.Number,
    duckdb.token_type.comment: token.Comment,
    duckdb.token_type.operator: token.Operator,
})


def _get_names(lf: LazyFrame, col_name: str) -> pc.Set[str]:

    return lf.select(col_name).fetch_all().iter().flatten().collect(pc.Set)


type ProcessedToken = tuple[int, TokenType, str]


class DuckDbSqlLexer(SqlLexer):
    @override
    def get_tokens_unprocessed(self, text: str) -> pc.Iter[ProcessedToken]:  # pyright: ignore[reportIncompatibleMethodOverride]
        process = partial(self._process, pc.Dict(duckdb.tokenize(text)))
        return pc.Iter(super().get_tokens_unprocessed(text)).map_star(process)

    def _process(  # noqa: PLR6301
        self,
        duck_tokens: pc.Dict[int, duckdb.token_type],
        pos: int,
        tokentype: TokenType,
        token_text: str,
    ) -> ProcessedToken:
        match duck_tokens.get_item(pos):
            case pc.Some(duckdb.token_type.identifier) if token_text in FUNCTIONS:
                return (pos, token.Name.Function, token_text)
            case pc.Some(duckdb.token_type.keyword) if token_text in DTYPES:
                return (pos, token.Name.Builtin, token_text)
            case pc.Some(duck_type):
                return (
                    pos,
                    DUCK_PYGMENT_MAP.get_item(duck_type).unwrap_or(tokentype),
                    token_text,
                )
            case _:
                return (pos, tokentype, token_text)


def _get_kwords() -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
    from sqlparse.tokens import Keyword

    from .sql import col, lit, when

    name = col("keyword_name")

    return (
        meta
        .keywords()
        .select(
            when(col("keyword_category").is_in(lit("reserved"), lit("unreserved")))
            .then(name.str.upper())
            .otherwise(name)
        )
        .fetch_all()
        .iter()
        .flatten()
        .map(lambda x: (x, Keyword))  # pyright: ignore[reportAny]
        .collect(dict)
    )


DTYPES = (
    meta
    .types()
    .pipe(_get_names, "type_name")
    .union(meta.types().pipe(_get_names, "logical_type"))
)
FUNCTIONS = meta.functions().pipe(_get_names, "function_name")

SYNTAX = partial(Syntax, lexer=DuckDbSqlLexer(), background_color="default")

Lexer.get_default_instance().add_keywords(_get_kwords())  # pyright: ignore[reportUnknownMemberType]
FORMATTER = partial(
    sqlparse.format,
    indent_width=4,
    reindent=True,
    keyword_case="upper",
    use_space_around_operators=True,
)


@dataclass(slots=True)
class ParsedQuery:
    raw: str

    def show(self, theme: Themes = "github-dark") -> None:
        return CONSOLE.print(SYNTAX(self.raw, theme=theme))

    def prettify(self) -> Self:
        return self.__class__(FORMATTER(self.raw))

    def tokenize(self) -> pc.Vec[tuple[int, duckdb.token_type]]:
        return pc.Vec.from_ref(duckdb.tokenize(self.raw))
