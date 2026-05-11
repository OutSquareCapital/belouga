from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import Field, fields
from typing import TYPE_CHECKING, Any

from pyochain import Iter
from sqlglot import exp

from .typing import RichRenderable

if TYPE_CHECKING:
    from rich.console import RenderableType

    from ._plan.nodes import BaseNode

# TODO: reduce code duplication
# TODO: handle tree without rich (hence why deduplication is critical)
# TODO: migrate the SQL rendering logic here and consolidate all "show" related logic in this module


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


def _node_field_value(node: BaseNode, name: str) -> object:
    return getattr(node, name)  # pyright: ignore[reportAny]


def expr_tree(node: exp.Expr) -> RenderableType:
    from rich.pretty import Pretty
    from rich.text import Text
    from rich.tree import Tree

    def _handle_mapping(branch: Tree, items: Mapping[Any, object]) -> None:  # pyright: ignore[reportExplicitAny]

        def _add_map_item(key: object, item: object) -> None:
            item_branch = branch.add(Text(repr(key), style="bright_black"))
            _attach(item_branch, item)

        if not items:
            return

        _ = Iter(items.items()).for_each_star(_add_map_item)

    def _handle_iterable(branch: Tree, items: Iterable[Any]) -> None:  # pyright: ignore[reportExplicitAny]

        def _add_iter_item(index: int, item: object) -> None:
            item_branch = branch.add(Text(f"[{index}]", style="bright_black"))
            _attach(item_branch, item)

        if not items:
            _ = branch.add(Pretty(items, expand_all=True))
            return

        _ = Iter(items).enumerate().for_each_star(_add_iter_item)

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

    name = node.__class__.__name__
    tree = Tree(Text(name, style="bold magenta"))
    _ = Iter(node.args.items()).for_each_star(_add_arg)
    return tree
