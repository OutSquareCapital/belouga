from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import Field, dataclass, fields
from functools import partial
from typing import TYPE_CHECKING, Any, override

import duckdb
from pygments import token
from pygments.lexers.sql import SqlLexer  # pyright: ignore[reportMissingTypeStubs]
from pyochain import Dict, Iter, Seq, Set, Some, Vec
from rich.console import Console
from rich.syntax import Syntax
from sqlglot import exp

from . import meta
from .typing import RichRenderable

if TYPE_CHECKING:
    from pygments.token import (
        _TokenType as TokenType,  # pyright: ignore[reportPrivateUsage]
    )
    from rich.console import RenderableType

    from ._frame import LazyFrame
    from ._plan.nodes import BaseNode
    from .typing import Themes

# TODO: reduce code duplication
# TODO: handle tree without rich if not available (hence why deduplication is critical)
# TODO: handle syntax highlighting if pygments is not available. Also make this lazy. Eventually see if we can handle this with duckdb/rich/sqlglot directly to avoid 2 table queries at the import of this module
# TODO: consolidate the method `LazyFrame::sql_query` in `LazyFrame::show` so we can directly see the tree with 3 options: 1) sql (pretty or not), 2) belugas IR, 3) sqlglot AST

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
        process = partial(_process_token, duck_tokens)
        return Iter(super().get_tokens_unprocessed(text)).map_star(process)


def _process_token(
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
            tken = DUCK_PYGMENT_MAP.get_item(duck_type).unwrap_or(tokentype)
            return (pos, tken, token_text)
        case _:
            return (pos, tokentype, token_text)


def _get_dtypes() -> Set[str]:
    lf_for = meta.types().pipe
    return lf_for(_get_names, "type_name").union(lf_for(_get_names, "logical_type"))


def _get_functions() -> Set[str]:
    return meta.functions().pipe(_get_names, "function_name")


def _get_names(lf: LazyFrame, col_name: str) -> Set[str]:
    return lf.select(col_name).fetch_all().iter().flatten().collect(Set)


DTYPES = _get_dtypes()
FUNCTIONS = _get_functions()
SYNTAX = partial(Syntax, lexer=DuckDbSqlLexer(), background_color="default")


@dataclass(slots=True)
class ParsedQuery:
    query: exp.Selectable

    def show(self, theme: Themes = "github-dark", *, pretty: bool = True) -> None:
        return CONSOLE.print(SYNTAX(self.sql(pretty=pretty), theme=theme))

    def tokenize(self) -> Vec[tuple[int, duckdb.token_type]]:
        return Vec.from_ref(duckdb.tokenize(self.sql()))

    def sql(self, *, pretty: bool = False) -> str:
        return self.query.sql(dialect="duckdb", pretty=pretty)


def node_tree(node: BaseNode) -> RenderableType:
    from rich.pretty import Pretty
    from rich.text import Text
    from rich.tree import Tree

    from ._plan.nodes import BaseNode

    def _attach(branch: Tree, value: object) -> None:
        match value:
            case BaseNode():
                _ = branch.add(value.__rich__())
            case exp.Expr():
                _ = branch.add(expr_tree(value))
            case RichRenderable():
                _ = branch.add(value.__rich__())
            case Mapping():
                if not value:
                    _ = branch.add(Pretty(value, expand_all=True))
                    return

                def _add_map_item(key: object, item: object) -> None:
                    item_branch = branch.add(Text(repr(key), style="bright_black"))
                    _attach(item_branch, item)

                _ = Iter(value.items()).for_each_star(_add_map_item)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
            case Iterable() as items if not isinstance(items, str | bytes | bytearray):
                if not items:
                    _ = branch.add(Pretty(items, expand_all=True))
                    return

                def _add_iter_item(index: int, item: object) -> None:
                    item_branch = branch.add(Text(f"[{index}]", style="bright_black"))
                    _attach(item_branch, item)

                _ = Iter(items).enumerate().for_each_star(_add_iter_item)
            case _:
                _ = branch.add(Pretty(value, expand_all=True))

    def _add_to_tree(field: Field[Any]) -> None:  # pyright: ignore[reportExplicitAny]
        name = field.name
        value = _node_field_value(node, name)
        match value:
            case BaseNode():
                branch = tree.add(Text("↑", style="bold bright_black"))
                _attach(branch, value)
            case _:
                branch = tree.add(Text(f"{name}", style="bold cyan"))
                _attach(branch, value)

    name = node.__class__.__name__
    header = Text(f" {name} ", style="bold white on dark_green")
    tree = Tree(header)
    node_fields = fields(node)
    (
        Iter(node_fields)
        .filter(lambda field: field.name != "inner")
        .chain(Iter(node_fields).filter(lambda field: field.name == "inner"))
        .for_each(_add_to_tree)
    )

    return tree


def expr_tree(node: exp.Expr) -> RenderableType:
    from rich.pretty import Pretty
    from rich.text import Text
    from rich.tree import Tree

    elem_style = "bright_black"

    def _expr_header(value: exp.Expr) -> Text:
        name = value.__class__.__name__
        match value:
            case exp.Selectable() | exp.From():
                return Text(f" {name.upper()} ", style="bold white on dark_green")
            case modifier if modifier.key in exp.QUERY_MODIFIERS:
                return Text(f" {name.upper()} ", style="bold white on dark_blue")
            case _:
                return Text(name, style="bold magenta")

    def _handle_mapping(branch: Tree, items: Mapping[Any, object]) -> None:  # pyright: ignore[reportExplicitAny]

        def _add_map_item(key: object, item: object) -> None:
            item_branch = branch.add(Text(repr(key), style=elem_style))
            _attach(item_branch, item)

        if not items:
            return

        Iter(items.items()).for_each_star(_add_map_item)

    def _handle_iterable(branch: Tree, items: Iterable[Any]) -> None:  # pyright: ignore[reportExplicitAny]

        def _add_iter_item(index: int, item: object) -> None:
            item_branch = branch.add(Text(f"[{index}]", style=elem_style))
            _attach(item_branch, item)

        if not items:
            _ = branch.add(Pretty(items, expand_all=True))
            return

        Iter(items).enumerate().for_each_star(_add_iter_item)

    def _attach(branch: Tree, value: object) -> None:
        match value:
            case exp.Expr():
                _ = branch.add(expr_tree(value))
            case Mapping():
                _handle_mapping(branch, value)  # pyright: ignore[reportUnknownArgumentType]
            case Iterable() if not isinstance(value, str | bytes | bytearray):
                _handle_iterable(branch, value)
            case _:
                _ = branch.add(Pretty(value, expand_all=True))

    def _add_arg(name: str, value: object) -> None:
        match value:
            case None | []:
                return
            case _:
                branch = tree.add(Text(name, style="bold cyan"))
                _attach(branch, value)

    tree = Tree(_expr_header(node))
    Iter(node.args.items()).for_each_star(_add_arg)
    return tree


def node_structure(node: object, level: int = 0) -> str:
    from ._plan.nodes import BaseNode

    indent = "\n" + ("  " * (level + 1))
    delim = f",{indent}"

    def _is_nested_node(node: BaseNode, name: str) -> bool:
        value: object = getattr(node, name)  # pyright: ignore[reportAny]
        return isinstance(value, BaseNode)

    match node:
        case BaseNode():
            node_fields = Seq(fields(node))
            is_leaf = node_fields.all(
                lambda field: not _is_nested_node(node, field.name)
            )

            if is_leaf:
                indent = ""
                delim = ", "

            items = (
                node_fields
                .iter()
                .filter(lambda field: field.name != "inner")
                .chain(node_fields.iter().filter(lambda field: field.name == "inner"))
                .map(lambda field: _field_to_s(node, field.name, level + 1))
                .join(delim)
            )
            return f"{node.__class__.__name__}({indent}{items})"
        case _:
            return repr(node)


def _field_to_s(node: BaseNode, name: str, level: int) -> str:
    value = _node_field_value(node, name)
    return f"{name}={node_structure(value, level)}"


def _node_field_value(node: BaseNode, name: str) -> object:
    return getattr(node, name)  # pyright: ignore[reportAny]
