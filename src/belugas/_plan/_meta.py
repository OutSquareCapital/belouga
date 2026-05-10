from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from pyochain import Iter, Null, Option, Seq, Set, Some
from pyochain.traits import Pipeable
from sqlglot import exp

from ..utils import TryIter, try_iter

if TYPE_CHECKING:
    from .._expr import Cols, Expr
    from ..typing import IntoExpr, Schema


class Marker(StrEnum):
    """Column name markers for special expression types."""

    LITERAL = auto()
    LEN = auto()
    TEMP = "__bl_temp__"

    def to_expr(self) -> Expr:
        from .._funcs import col

        return col(self.value)


class Tables:
    SRC: exp.Table = exp.to_table("src")
    LHS: exp.Table = exp.to_table("lhs")
    RHS: exp.Table = exp.to_table("rhs")
    STATS: exp.Table = exp.to_table("stats")
    EXPLODE_SRC: exp.Table = exp.to_table("_explode_src")


def _has_window_ancestor(node: exp.Expr) -> bool:
    def _ancestor_is_window(ancestor: exp.Expr | None) -> bool:
        match ancestor:
            case exp.Window():
                return True
            case exp.Distinct() | exp.Filter() | exp.IgnoreNulls() | exp.WithinGroup():
                return (
                    Option(ancestor.parent)
                    .map(_ancestor_is_window)
                    .unwrap_or(default=False)
                )
            case _:
                return False

    return _ancestor_is_window(node.parent)


def lookup_type(inner: exp.Expr, schema: Schema) -> exp.DataType:
    node = inner.unalias()
    match node, node.args.get("to"):
        case exp.Cast() | exp.TryCast(), exp.DataType() as to:
            return to
        case _:
            return (
                Option(node.find(exp.Column))
                .and_then(lambda c: schema.get_item(c.output_name))
                .unwrap_or_else(exp.DType.UNKNOWN.into_expr)
            )


@dataclass(slots=True, init=False)
class ResolvedExpr(Pipeable):
    """A fully resolved expression ready for SQL emission."""

    expr: Expr
    name: str
    has_projection_distinct: bool
    is_pure_reducer: bool

    def __init__(self, expr: Expr, name: str) -> None:
        self.name = name
        inner = expr.inner

        def _is_projection_distinct(node: exp.Expr) -> bool:
            return node.find_ancestor(exp.AggFunc, exp.List, exp.Window) is None

        def _classify(
            acc: tuple[bool, bool, bool], node: exp.Expr
        ) -> tuple[bool, bool, bool]:
            has_distinct, has_bare_agg, has_bare_col = acc
            match node:
                case exp.Distinct() if _is_projection_distinct(node):
                    return (True, has_bare_agg, has_bare_col)
                case exp.AggFunc() | exp.List() if not _has_window_ancestor(node):
                    return (has_distinct, True, has_bare_col)
                case exp.Column() if _is_projection_distinct(node):
                    return (has_distinct, has_bare_agg, True)
                case _:
                    return acc

        has_distinct, has_bare_agg, has_bare_col = Iter(inner.walk(bfs=False)).fold(
            (False, False, False), _classify
        )
        self.has_projection_distinct = has_distinct
        self.is_pure_reducer = has_bare_agg and not has_bare_col

        if has_distinct:

            def _strip(node: exp.Expr) -> exp.Expr:
                match node:
                    case exp.Distinct(expressions=[exp.Expr() as expr]):
                        return expr
                    case _:
                        return node

            self.expr = inner.transform(_strip).pipe(expr.__class__)
        else:
            self.expr = expr

    def as_aliased(self, *, broadcast_agg: bool) -> Expr:
        def _broadcast_reducers(expr: Expr) -> Expr:
            def _window_agg(node: exp.Expr) -> exp.Expr:
                match node:
                    case exp.AggFunc() | exp.List() if not _has_window_ancestor(node):
                        return expr.__class__(node, expr.aliaser).window().inner
                    case _:
                        return node

            return expr.inner.transform(_window_agg).pipe(expr.__class__)

        return self.expr.pipe(
            lambda e: _broadcast_reducers(e) if broadcast_agg else e
        ).alias(self.name)


def resolve_all(
    schema: Schema,
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    named_exprs: dict[str, IntoExpr],
) -> Seq[ResolvedExpr]:

    def _alias_named_expr(name: str, val: IntoExpr) -> IntoExpr:
        from .._expr import Expr

        match val:
            case Expr():
                return val.alias(name)
            case _:
                return Expr.new(val, as_col=True).alias(name)

    return (
        try_iter(exprs)
        .chain(more_exprs)
        .chain(Iter(named_exprs.items()).map_star(_alias_named_expr))
        .flat_map(lambda val: _resolve(val, schema))
        .collect()
    )


def _resolve(val: IntoExpr, schema: Schema) -> Iter[ResolvedExpr]:  # noqa: PLR0912
    from .._expr import Expr, MultiAliasMapper

    def _get_inner_node(inner_node: exp.Star | list[exp.Expr]) -> Cols:
        match inner_node:
            case exp.Star():
                return schema.keys()
            case list() as cols:
                return Iter(cols).map(lambda c: c.name).collect()

    match val:
        case Expr():
            match val.aliaser:
                case MultiAliasMapper() as meta:
                    base_names = meta.resolver(schema)
                    match val.inner, meta.alias_name:
                        case exp.Alias() as inner, Some(alias_fn):
                            output_names = Seq((alias_fn(inner.output_name),))
                        case exp.Alias() as inner, Null():
                            output_names = Seq((inner.output_name,))
                        case _, Some(alias_fn):
                            output_names = base_names.iter().map(alias_fn).collect()
                        case _:
                            output_names = base_names
                    return _expand_columns(val, base_names, output_names)

                case _:
                    match val.inner.find(exp.Columns):
                        case None:
                            match val.inner:
                                case exp.Star() as star:
                                    excepts: list[exp.Expr] | None = star.args.get(
                                        "except_"
                                    )
                                    match excepts:
                                        case None:
                                            base_names = schema.keys()
                                        case list():
                                            excluded = (
                                                Iter(excepts)
                                                .map(lambda c: c.name)
                                                .collect(Set)
                                            )
                                            base_names = (
                                                schema
                                                .iter()
                                                .filter(lambda n: n not in excluded)
                                                .collect()
                                            )
                                    return _expand_columns(val, base_names, base_names)
                                case _:
                                    name = extract_root_name(val.inner)
                                    return ResolvedExpr(val, name).into(Iter.once)
                        case _ as columns_node:
                            base_names = _get_inner_node(columns_node.this)  # pyright: ignore[reportAny]
                            return _expand_columns(val, base_names, base_names)
        case _:
            return (
                Expr
                .new(val, as_col=True)
                .pipe(lambda e: ResolvedExpr(e, e.inner.output_name))
                .into(Iter.once)
            )


def _expand_columns(
    expr: Expr, base_names: Cols, output_names: Cols
) -> Iter[ResolvedExpr]:
    def _resolved(src: Expr, col_name: str, name: str) -> ResolvedExpr:
        target = exp.column(col_name)

        def _replacer(node: exp.Expr) -> exp.Expr:
            match node:
                case exp.Star() | exp.Columns():
                    return target
                case _:
                    return node

        return (
            src.inner.transform(_replacer).pipe(src.__class__).pipe(ResolvedExpr, name)
        )

    match expr.inner:
        case exp.Alias():
            unaliased = expr.inner.unalias().pipe(expr.__class__)
            alias = output_names.first()
            return base_names.iter().map(
                lambda col_name: _resolved(unaliased, col_name, alias)
            )
        case _:
            return (
                base_names
                .iter()
                .zip(output_names)
                .map_star(lambda name, output: _resolved(expr, name, output))
            )


def extract_root_name(node: exp.Expr) -> str:  # noqa: PLR0911
    match node:
        case exp.Alias() | exp.Column():
            return node.output_name
        case exp.Literal() | exp.Boolean() | exp.Null():
            return Marker.LITERAL
        case exp.Case():
            return _case_root_name(node)
        case exp.Anonymous() | exp.AnonymousAggFunc() | exp.Distinct() | exp.List():
            match node.expressions:
                case [exp.Expr() as first_arg, *_]:
                    return extract_root_name(first_arg)
                case _:
                    return Marker.LITERAL
        case exp.Func():
            return _func_root_name(node)
        case exp.Window():
            return _window_root_name(node)
        case exp.Expr():
            match node.this:  # pyright: ignore[reportAny]
                case exp.Expr() as inner:
                    return extract_root_name(inner)
                case _:  # pyright: ignore[reportAny]
                    return Marker.LITERAL


def _case_root_name(node: exp.Case) -> str:
    match node.args.get("ifs", []):
        case [exp.If() as first_if, *_]:
            match first_if.args.get("true"):
                case exp.Expr() as then_val:
                    return extract_root_name(then_val)
                case _:
                    return Marker.LITERAL
        case _:  # pyright: ignore[reportAny]
            return Marker.LITERAL


def _func_root_name(node: exp.Func) -> str:
    match node.this:  # pyright: ignore[reportAny]
        case exp.Expr() as inner:
            name = extract_root_name(inner)
            match name:
                case Marker.LITERAL:
                    return _root_col_name(node)
                case _:
                    return name
        case _:  # pyright: ignore[reportAny]
            return _root_col_name(node)


def _window_root_name(node: exp.Window) -> str:
    name = extract_root_name(node.this)  # pyright: ignore[reportAny]
    if name in {Marker.LITERAL, Marker.TEMP}:
        return _root_col_name(node)
    return name


def _root_col_name(node: exp.Expr) -> str:
    return (
        find_all(node, exp.Column)
        .map(lambda c: c.output_name)
        .find(lambda name: name != Marker.TEMP)
        .unwrap_or(Marker.LITERAL)
    )


def find_all[T: exp.Expr](expr: exp.Expr, *exprs: type[T], bfs: bool = True) -> Iter[T]:
    return Iter(expr.find_all(*exprs, bfs=bfs))
