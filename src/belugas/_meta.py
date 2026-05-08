from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, Self, override

from pyochain import NONE, Dict, Iter, NoneOption as Null, Option, Seq, Set, Some
from pyochain.traits import Pipeable
from sqlglot import exp

from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from pyochain.traits import PyoIterable

    from ._expr import Expr
    from .selectors import Cols, Resolver
    from .typing import IntoExpr, Schema

type Aliaser = Callable[[str], str]
"""Alias function type, used for generating deferred column aliases."""


class Marker(StrEnum):
    """Column name markers for special expression types."""

    LITERAL = auto()
    LEN = auto()
    TEMP = "__bl_temp__"

    def to_expr(self) -> Expr:
        from ._funcs import col

        return col(self.value)


def _into_windowed(cols: PyoIterable[ResolvedExpr]) -> exp.Expr:
    from ._funcs import row_number

    source = exp.to_table("src")
    if cols.any(lambda p: p.is_windowed(Marker.TEMP)):
        row_nb = row_number().window().sub(1).alias(Marker.TEMP).inner
        return exp.select(row_nb, exp.Star()).from_(source).subquery("src")
    return source


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


def _is_projection_distinct(node: exp.Expr) -> bool:
    return node.find_ancestor(exp.AggFunc, exp.List, exp.Window) is None


def _resolve_exploded(expr: Expr, *, is_distinct: bool) -> Expr:
    if is_distinct:
        return expr.implode().list.distinct()
    return expr.implode()


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
        _find_all(node, exp.Column)
        .map(lambda c: c.output_name)
        .find(lambda name: name != Marker.TEMP)
        .unwrap_or(Marker.LITERAL)
    )


@dataclass(slots=True)
class ExprMeta:
    """Metadata for expressions, used for tracking properties that affect query generation."""

    def unalias(self) -> Self:
        return self


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: Resolver
    alias_name: Option[Aliaser] = field(default_factory=lambda: NONE)

    def get_output_names(self, base_names: Cols, expr: Expr) -> Cols:
        match expr.inner, self.alias_name:
            case exp.Alias() as inner, Some(alias_fn):
                return Iter.once(inner.output_name).map(alias_fn).collect()
            case exp.Alias() as inner, Null():
                return Seq((inner.output_name,))
            case _, Some(alias_fn):
                return base_names.iter().map(alias_fn).collect()
            case _:
                return base_names

    def with_alias_mapper(self, mapper: Aliaser) -> Self:
        def _get_mapper() -> Aliaser:
            match self.alias_name:
                case Some(current):
                    return lambda name: mapper(current(name))
                case _:
                    return mapper

        return self.__class__(self.resolver, Some(_get_mapper()))

    @override
    def unalias(self) -> Self:
        return self.__class__(self.resolver, NONE)

    def into_resolved(self, expr: Expr, schema: Schema) -> Iter[ResolvedExpr]:
        base_names = self.resolver(schema)
        output_names = self.get_output_names(base_names, expr)

        def _resolved(expr: Expr, col_name: str, name: str) -> ResolvedExpr:
            target = exp.column(col_name)

            def _replacer(node: exp.Expr) -> exp.Expr:
                match node:
                    case exp.Star() | exp.Columns():
                        return target
                    case _:
                        return node

            return (
                expr.inner
                .transform(_replacer)
                .pipe(expr.__class__)
                .pipe(ResolvedExpr, name)
            )

        match expr.inner:
            case exp.Alias():
                expr = expr.inner.unalias().pipe(expr.__class__)
                alias = output_names.first()
                return base_names.iter().map(lambda name: _resolved(expr, name, alias))

            case _:
                return (
                    base_names
                    .iter()
                    .zip(output_names)
                    .map_star(lambda name, output: _resolved(expr, name, output))
                )


def _find_all[T: exp.Expr](
    expr: exp.Expr, *exprs: type[T], bfs: bool = True
) -> Iter[T]:
    return Iter(expr.find_all(*exprs, bfs=bfs))


@dataclass(slots=True, init=False)
class ResolvedExpr(Pipeable):
    """A fully resolved expression ready for SQL emission."""

    expr: Expr
    name: str
    has_projection_distinct: bool
    is_pure_reducer: bool
    is_multi: bool

    def __init__(self, expr: Expr, name: str) -> None:
        self.name = name
        inner = expr.inner
        self.has_projection_distinct = inner.pipe(_find_all, exp.Distinct).any(
            _is_projection_distinct
        )

        search = partial(_find_all, inner)
        self.is_pure_reducer = search(exp.AggFunc, exp.List).any(
            lambda node: not _has_window_ancestor(node)
        ) and not search(exp.Column).any(_is_projection_distinct)
        self.is_multi = isinstance(inner, exp.Columns) or inner.is_star
        if self.has_projection_distinct:

            def _strip(node: exp.Expr) -> exp.Expr:
                match node:
                    case exp.Distinct(expressions=[exp.Expr() as expr]) if (
                        _is_projection_distinct(node)
                    ):
                        return expr
                    case _:
                        return node

            self.expr = inner.transform(_strip).pipe(expr.__class__)
        else:
            self.expr = expr

    def maybe_alias(self, expr: Expr) -> Expr:
        return expr if self.is_multi or not self.name else expr.alias(self.name)

    def implode_or_scalar(self) -> Expr:
        if self.is_pure_reducer:
            expr = self.expr
        else:
            expr = self.expr.pipe(
                _resolve_exploded, is_distinct=self.has_projection_distinct
            )
        return expr.pipe(self.maybe_alias)

    def as_aliased(self, *, broadcast_agg: bool) -> Expr:
        def _broadcast_reducers(expr: Expr) -> Expr:
            def _window_agg(node: exp.Expr) -> exp.Expr:
                match node:
                    case exp.AggFunc() | exp.List() if not _has_window_ancestor(node):
                        return expr.__class__(node, expr.meta).window().inner
                    case _:
                        return node

            return expr.inner.transform(_window_agg).pipe(expr.__class__)

        return self.expr.pipe(
            lambda e: _broadcast_reducers(e) if broadcast_agg else e
        ).pipe(self.maybe_alias)

    def is_windowed(self, marker: Marker) -> bool:
        is_temp = self.expr.inner.pipe(_find_all, exp.Column).any(
            lambda col: col.parts[-1].name == marker
        )
        return self.name != marker and is_temp


@dataclass(slots=True, init=False)
class ExprPlan:
    schema: Schema
    projections: Seq[ResolvedExpr]

    def __init__(
        self,
        schema: Schema,
        exprs: TryIter[IntoExpr],
        more_exprs: Iterable[IntoExpr],
        named_exprs: dict[str, IntoExpr],
    ) -> None:

        def _alias_named_expr(name: str, val: IntoExpr) -> IntoExpr:
            from ._expr import Expr

            match val:
                case Expr():
                    return val.alias(name)
                case _:
                    return Expr.new(val, as_col=True).alias(name)

        def _resolve(val: IntoExpr) -> Iter[ResolvedExpr]:
            from ._expr import Expr
            from .selectors import Selector

            match val:
                case Selector():
                    return val.meta.into_resolved(val, schema)
                case Expr():
                    name = extract_root_name(val.inner)
                    return ResolvedExpr(val, name).into(Iter.once)
                case _:
                    return (
                        Expr
                        .new(val, as_col=True)
                        .pipe(lambda e: ResolvedExpr(e, e.inner.output_name))
                        .into(Iter.once)
                    )

        self.schema = schema
        self.projections = (
            try_iter(exprs)
            .chain(more_exprs)
            .chain(Iter(named_exprs.items()).map_star(_alias_named_expr))
            .flat_map(_resolve)
            .collect()
        )

    def _should_broadcast_agg(self, *, include_source_cols: bool) -> bool:
        return include_source_cols or not self.projections.all(
            lambda resolved: resolved.is_pure_reducer
        )

    def select_ctx(self) -> Option[exp.Select]:
        def _non_empty_slct(source: exp.Expr) -> exp.Select:
            if self.projections.all(
                lambda resolved: resolved.has_projection_distinct
            ):
                return (
                    self.aliased_sql(broadcast_agg=False).from_(source).distinct()
                )
            return self.aliased_sql(
                broadcast_agg=self._should_broadcast_agg(
                    include_source_cols=False
                )
            ).from_(source)

        return self.projections.then(
            lambda _projs: _projs.into(_into_windowed).pipe(_non_empty_slct)
        )

    def with_columns_ctx(self) -> exp.Select:
        def _resolved(updates: Dict[str, Expr]) -> Iter[exp.Expr]:
            update_iter = updates.items().iter()
            if not updates.any(lambda name: name in self.schema):
                return update_iter.map_star(lambda _name, expr: expr.inner).insert(
                    exp.Star()
                )
            return (
                self.schema
                .iter()
                .map(
                    lambda name: updates.get_item(name).map_or(
                        exp.column(name), lambda expr: expr.inner
                    )
                )
                .chain(
                    update_iter.filter_star(
                        lambda name, _expr: name not in self.schema
                    ).map_star(lambda _name, expr: expr.inner)
                )
            )

        broadcast_agg = self._should_broadcast_agg(include_source_cols=True)
        updates = (
            self.projections
            .iter()
            .filter(lambda proj: not proj.is_multi)
            .map(
                lambda proj: (
                    proj.name,
                    proj.as_aliased(broadcast_agg=broadcast_agg),
                )
            )
            .collect(Dict)
        )
        return exp.select(*updates.into(_resolved)).from_(
            self.projections.into(_into_windowed)
        )

    def with_fields_ctx(self, expr: Expr) -> Expr:
        return (
            self.projections
            .iter()
            .map(lambda proj: proj.as_aliased(broadcast_agg=False))
            .into(lambda args: expr.struct.insert(*args))
        )

    def group_by_all_ctx(self) -> exp.Select:
        return self.aliased_sql(broadcast_agg=False).from_("src").group_by("ALL")

    def aliased_sql(self, *, broadcast_agg: bool) -> exp.Select:
        def _into_expr(resolved: ResolvedExpr) -> exp.Expr:
            return resolved.as_aliased(broadcast_agg=broadcast_agg).inner

        return exp.select(*self.projections.iter().map(_into_expr))

    def agg_ctx(self, keys: PyoIterable[exp.Expr]) -> exp.Select:
        def _lower_projection(proj: ResolvedExpr) -> Iter[exp.Expr]:
            def _excluded(star: exp.Star) -> Set[str]:
                return (
                    Option(star.args.get("except_"))
                    .map(Iter[exp.Expr])
                    .unwrap_or_else(Iter[exp.Expr].new)
                    .map(lambda e: e.name)
                    .collect(Set)
                )

            def _into_glot(name: str) -> exp.Expr:
                from ._funcs import col

                return col(name).pipe(ResolvedExpr, name).implode_or_scalar().inner

            match proj.expr.inner:
                case exp.Star() as star:
                    excluded = _excluded(star)
                    return (
                        self.schema
                        .iter()
                        .filter(lambda name: name not in excluded)
                        .map(_into_glot)
                    )
                case exp.Explode(this=exp.Expr() as inner):
                    return (
                        proj.expr
                        .__class__(inner)
                        .pipe(
                            _resolve_exploded, is_distinct=proj.has_projection_distinct
                        )
                        .list.flatten()
                        .alias(proj.name)
                        .inner.pipe(Iter.once)
                    )
                case _:
                    return proj.implode_or_scalar().inner.pipe(Iter.once)

        exprs = keys.iter().chain(self.projections.iter().flat_map(_lower_projection))
        return exp.select(*exprs).from_("src")
