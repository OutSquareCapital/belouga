import ast
import re
from typing import override

from pyochain import NONE, Dict, Iter, Option, Seq, Some

from .._utils import Builtins, CollectionsABC, Pql, Pyochain, Typing
from ._rules import CONTAINER_SUPERTYPES, TYPE_SUPERTYPES, ContainerType

GENERIC_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

SELF_PATTERN = re.compile(r"\b(Self|Expr|LazyFrame)\b")


def _slice_to_expr(slice_node: ast.expr) -> ast.expr:
    match slice_node:
        case ast.Tuple() | ast.Name() | ast.Subscript() | ast.BinOp() | ast.Attribute():
            return slice_node
        case _:
            return ast.Name(id="Any", ctx=ast.Load())


class _AliasExpander(ast.NodeTransformer):
    _single_param_aliases: Dict[str, str] = Dict.from_kwargs(
        TryIter="Iterable[{arg}] | {arg}",
        TrySeq="Sequence[{arg}] | {arg}",
    )

    @override
    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        rewritten = self.generic_visit(node)
        match rewritten:
            case ast.Subscript(value=ast.Name(id=alias_name), slice=slice_node):
                return (
                    self._single_param_aliases
                    .get_item(alias_name)
                    .map(
                        lambda template: ast.copy_location(
                            ast.parse(
                                template.format(
                                    arg=ast.unparse(_slice_to_expr(slice_node))
                                ),
                                mode="eval",
                            ).body,
                            rewritten,
                        )
                    )
                    .unwrap_or(rewritten)
                )
            case _:
                return rewritten


class _GenericCanonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._mapping: Dict[str, str] = Dict.new()

    @override
    def visit_Name(self, node: ast.Name) -> ast.AST:
        match node.id:
            case name if name not in {Typing.SELF, Pql.EXPR, Pql.LAZY_FRAME} and bool(
                GENERIC_SYMBOL_PATTERN.match(name)
            ):
                canonical_name = self._mapping.setdefault(
                    name, f"__GENERIC_{self._mapping.length()}__"
                )
                return ast.copy_location(
                    ast.Name(id=canonical_name, ctx=node.ctx), node
                )
            case _:
                return node


class _UnionCanonicalizer(ast.NodeTransformer):
    @override
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        match self.generic_visit(node):
            case ast.BinOp(op=ast.BitOr()):

                def _build_union_expr(parts: Seq[str]) -> ast.expr:
                    def _union_expr(left: ast.expr, right: ast.expr) -> ast.expr:
                        return ast.BinOp(left=left, op=ast.BitOr(), right=right)

                    def _build_union(
                        acc: Option[ast.expr], expr: ast.expr
                    ) -> Option[ast.expr]:
                        return acc.map(lambda left: _union_expr(left, expr)).or_(
                            Some(expr)
                        )

                    return (
                        parts
                        .iter()
                        .map(lambda part: ast.parse(part, mode="eval").body)
                        .fold(NONE, _build_union)
                        .unwrap_or(ast.Constant(value=None))
                    )

                def _union_members(node: ast.expr) -> Iter[ast.expr]:
                    match node:
                        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                            return _union_members(left).chain(_union_members(right))
                        case _:
                            return Iter.once(node)

                members_as_text = _union_members(node).map(ast.unparse).collect()
                has_float = members_as_text.any(lambda text: text == Builtins.FLOAT)

                return (
                    members_as_text
                    .iter()
                    .filter(lambda text: not (has_float and text == Builtins.INT))
                    .unique()
                    .sort()
                    .into(_build_union_expr)
                )

            case _ as rewritten:
                return rewritten


def annotations_compatible(reference_ann: str, target_ann: str) -> bool:
    normalized_reference = normalize_annotation(reference_ann)
    normalized_target = normalize_annotation(target_ann)
    match (
        normalized_reference in {Typing.ANY, normalized_target}
        or normalized_target == Typing.ANY
    ):
        case True:
            return True
        case False:
            return _annotation_accepts(
                _normalize_expr(ast.parse(normalized_target, mode="eval")),
                _normalize_expr(ast.parse(normalized_reference, mode="eval")),
            )


def normalize_annotation(annotation: str) -> str:

    return extract_last_name(
        SELF_PATTERN.sub(
            "__SELF__",
            extract_last_name(
                ast.unparse(_normalize_expr(ast.parse(annotation, mode="eval")))
            ),
        )
    )


def _normalize_expr(parsed: ast.Expression) -> ast.expr:
    return ast.fix_missing_locations(  # pyright: ignore[reportAny]
        _UnionCanonicalizer().visit(  # pyright: ignore[reportAny]
            _GenericCanonicalizer().visit(_AliasExpander().visit(parsed))  # pyright: ignore[reportAny]
        )
    ).body


def _generic_base_accepts(
    target_base: CollectionsABC,
    target_args: Seq[ast.expr],
    reference_base: ContainerType,
    reference_args: Seq[ast.expr],
) -> bool:
    return (
        _collection_item_type(target_base, target_args)
        .and_then(
            lambda target_item: _collection_item_type(
                reference_base, reference_args
            ).map(
                lambda reference_item: (
                    CONTAINER_SUPERTYPES
                    .get_item(target_base)
                    .map(lambda accepted: accepted.contains(reference_base))
                    .unwrap_or(default=False)
                    and _annotation_accepts(target_item, reference_item)
                )
            )
        )
        .unwrap_or(
            target_base == reference_base
            and target_args.length() == reference_args.length()
            and target_args
            .iter()
            .zip(reference_args)
            .map_star(_annotation_accepts)
            .all(bool)
        )
    )


def _annotation_accepts(target: ast.expr, reference: ast.expr) -> bool:
    target_members = _union_members(target).collect()
    return _union_members(reference).all(
        lambda reference_member: target_members.any(
            lambda target_member: _member_accepts(target_member, reference_member)
        )
    )


def _union_members(reference: ast.expr) -> Iter[ast.expr]:
    match reference:
        case ast.BinOp(left=left, op=ast.BitOr(), right=right):
            return _union_members(left).chain(_union_members(right))
        case _:
            return Iter.once(reference)


def _member_accepts(target: ast.expr, reference: ast.expr) -> bool:
    if ast.unparse(target) == ast.unparse(reference):
        return True
    match (target, reference):
        case (ast.Subscript(), ast.Subscript()):
            return _generic_accepts(target, reference)
        case _:
            return (
                _type_name(target)
                .and_then(
                    lambda target_name: _type_name(reference).map(
                        lambda reference_name: _type_accepts(
                            target_name, reference_name
                        )
                    )
                )
                .unwrap_or(default=False)
            )


def _generic_accepts(target: ast.Subscript, reference: ast.Subscript) -> bool:
    return (
        _type_name(target.value)
        .and_then(
            lambda target_base: _type_name(reference.value).map(
                lambda reference_base: _generic_base_accepts(
                    target_base,  # pyright: ignore[reportArgumentType]
                    _into_seq_args(target.slice),
                    reference_base,  # pyright: ignore[reportArgumentType]
                    _into_seq_args(reference.slice),
                )
            )
        )
        .unwrap_or(default=False)
    )


def _into_seq_args(target: ast.expr) -> Seq[ast.expr]:
    match target:
        case ast.Tuple():
            return Seq(target.elts)
        case _:
            return Seq((target,))


def _collection_item_type(base: str, args: Seq[ast.expr]) -> Option[ast.expr]:
    match (base, args.length()):
        case (
            CollectionsABC.ITERABLE
            | CollectionsABC.COLLECTION
            | CollectionsABC.SEQUENCE
            | Builtins.LIST
            | Builtins.SET
            | Builtins.FROZENSET
            | Pyochain.SEQ
            | Pyochain.VEC,
            1,
        ):
            return Some(args.first())
        case (Builtins.TUPLE, 1):
            return Some(args.first())
        case (Builtins.TUPLE, 2):
            match args[1]:
                case ast.Constant() as constant if constant.value is Ellipsis:
                    return Some(args.first())
                case _:
                    return NONE
        case _:
            return NONE


def _type_name(node: ast.expr) -> Option[str]:
    match node:
        case ast.Name(id=name):
            return Some(name)
        case ast.Attribute(attr=name):
            return Some(name)
        case ast.Constant(value=None):
            return Some(Builtins.NONE)
        case _:
            return NONE


def _type_accepts(target_name: str, reference_name: str) -> bool:
    return target_name in {reference_name, Typing.ANY} or TYPE_SUPERTYPES.get_item(
        target_name
    ).map(lambda accepted: accepted.contains(reference_name)).unwrap_or(default=False)


def extract_last_name(annotation: str) -> str:
    if "[" in annotation:
        base_type = annotation.split("[", maxsplit=1)[0]
        generic_part = annotation[len(base_type) :]
        return extract_last_name(base_type) + generic_part

    return annotation.rsplit(".", maxsplit=1)[-1]
