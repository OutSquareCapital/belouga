from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, Self, override

import pyochain as pc
from pyochain.traits import Pipeable
from sqlglot import exp

from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT
    from pyochain.traits import PyoIterable

    from ._expr import Expr
    from .selectors import Cols, Resolver
    from .typing import IntoExpr

type Aliaser = Callable[[str], str]
"""Alias function type, used for generating deferred column aliases."""


class Marker(StrEnum):
    """Column name markers for special expression types."""

    ELEMENT = auto()
    LITERAL = auto()
    LEN = auto()
    TEMP = "__pql_temp__"

    def to_expr(self) -> Expr:
        from ._funcs import col

        return col(self.value)

    @classmethod
    def replace_col(cls, expr: Expr, column_name: str) -> Expr:
        target = exp.column(column_name)

        def _replacer(node: exp.Expr) -> exp.Expr:
            match node:
                case exp.Star() | exp.Columns():
                    return target
                case _:
                    return node

        return expr.__class__(expr.inner.transform(_replacer))  # pyright: ignore[reportUnknownMemberType, reportAny]

    @classmethod
    def drop_marker(cls, result: IntoFrameT, cols: Collection[str]) -> IntoFrameT:
        import narwhals as nw

        match cls.TEMP in cols:
            case True:
                return nw.from_native(result).drop(cls.TEMP).to_native()
            case False:
                return result

    @classmethod
    def windowed(cls, source: exp.Expr, cols: PyoIterable[ResolvedExpr]) -> exp.Expr:
        from ._funcs import row_number

        match cols.any(lambda p: p.is_windowed(cls.TEMP)):
            case True:
                return (
                    row_number()
                    .window()
                    .sub(1)
                    .alias(cls.TEMP)
                    .inner.pipe(lambda row_nb: exp.select(row_nb, exp.Star()))
                    .from_(source)
                    .subquery("src")
                )
            case False:
                return source


def _broadcast_reducers(expr: Expr) -> Expr:
    def _window_agg(node: exp.Expr) -> exp.Expr:
        match node:
            case exp.AggFunc() | exp.List() if not _has_window_ancestor(node):
                return expr.__class__(node, expr.meta).window().inner
            case _:
                return node

    return expr.__class__(expr.inner.transform(_window_agg))  # pyright: ignore[reportUnknownMemberType, reportAny]


def _has_window_ancestor(node: exp.Expr) -> bool:
    def _ancestor_is_window(ancestor: exp.Expr | None) -> bool:
        match ancestor:
            case exp.Window():
                return True
            case exp.Distinct() | exp.Filter() | exp.IgnoreNulls() | exp.WithinGroup():
                return (
                    pc.Option
                    .if_some(ancestor.parent)
                    .map(_ancestor_is_window)
                    .unwrap_or(default=False)
                )
            case _:
                return False

    return _ancestor_is_window(node.parent)


def _is_projection_distinct(node: exp.Expr) -> bool:
    return node.find_ancestor(exp.AggFunc, exp.List, exp.Window) is None


def _resolve_exploded(expr: Expr, *, is_distinct: bool) -> Expr:
    match is_distinct:
        case True:
            return expr.implode().list.distinct()
        case False:
            return expr.implode()


def _extract_root_name(node: exp.Expr) -> str:  # noqa: C901, PLR0911, PLR0912
    match node:
        case exp.Alias() | exp.Column():
            return node.output_name
        case exp.Literal() | exp.Boolean() | exp.Null():
            return Marker.LITERAL
        case exp.Case():
            match node.args.get("ifs", []):
                case [exp.If() as first_if, *_]:
                    match first_if.args.get("true"):
                        case exp.Expr() as then_val:
                            name = _extract_root_name(then_val)
                            match name:
                                case Marker.LITERAL:
                                    match node.args.get("default"):
                                        case exp.Expr() as default_val:
                                            return _extract_root_name(default_val)
                                        case _:
                                            return name
                                case _:
                                    return name
                        case _:
                            return Marker.LITERAL
                case _:  # pyright: ignore[reportAny]
                    return Marker.LITERAL
        case exp.Anonymous() | exp.AnonymousAggFunc() | exp.Distinct() | exp.List():
            match node.expressions:
                case [exp.Expr() as first_arg, *_]:
                    return _extract_root_name(first_arg)
                case _:
                    return Marker.LITERAL
        case exp.Func():
            match node.this:  # pyright: ignore[reportAny]
                case exp.Expr() as inner:
                    name = _extract_root_name(inner)
                    match name:
                        case Marker.LITERAL:
                            return _root_col_name(node)
                        case _:
                            return name
                case _:  # pyright: ignore[reportAny]
                    return _root_col_name(node)
        case exp.Window():
            name = _extract_root_name(node.this)  # pyright: ignore[reportAny]
            match name in {Marker.LITERAL, Marker.TEMP}:
                case True:
                    return _root_col_name(node)
                case False:
                    return name
        case _:
            match node.this:  # pyright: ignore[reportAny]
                case exp.Expr() as inner:
                    return _extract_root_name(inner)
                case _:  # pyright: ignore[reportAny]
                    return Marker.LITERAL


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

    alias_name: pc.Option[Aliaser] = field(default_factory=lambda: pc.NONE)

    def into_resolved(self, expr: Expr, _cols: Cols) -> pc.Iter[ResolvedExpr]:
        output_name = (_extract_root_name(expr.inner),)
        name = pc.Seq(output_name).into(self.get_output_names, expr).first()
        return ResolvedExpr(expr, name).into(pc.Iter.once)

    def get_output_names(self, base_names: Cols, expr: Expr) -> Cols:
        match expr.inner, self.alias_name:
            case exp.Alias() as inner, pc.Some(alias_fn):
                return pc.Iter.once(inner.output_name).map(alias_fn).collect()
            case exp.Alias() as inner, pc.NONE:
                return pc.Seq((inner.output_name,))
            case _, pc.Some(alias_fn):
                return base_names.iter().map(alias_fn).collect()
            case _:
                return base_names

    def with_alias_mapper(self, mapper: Aliaser) -> Self:
        def _get_mapper() -> Aliaser:
            match self.alias_name:
                case pc.Some(current):
                    return lambda name: mapper(current(name))
                case _:
                    return mapper

        return replace(self, alias_name=pc.Some(_get_mapper()))

    def unalias(self) -> Self:
        return replace(self, alias_name=pc.NONE)


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: Resolver = field(kw_only=True)

    @override
    def into_resolved(self, expr: Expr, cols: Cols) -> pc.Iter[ResolvedExpr]:
        base_names = self.resolver(cols)
        output_names = self.get_output_names(base_names, expr)

        def _resolved(expr: Expr, col_name: str, name: str) -> ResolvedExpr:
            return ResolvedExpr(Marker.replace_col(expr, col_name), name)

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
) -> pc.Iter[T]:
    return pc.Iter(expr.find_all(*exprs, bfs=bfs))


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
        match self.has_projection_distinct:
            case True:

                def _strip(node: exp.Expr) -> exp.Expr:
                    match node:
                        case exp.Distinct(expressions=[exp.Expr() as expr]) if (
                            _is_projection_distinct(node)
                        ):
                            return expr
                        case _:
                            return node

                self.expr = expr.__class__(inner.transform(_strip))  # pyright: ignore[reportUnknownMemberType, reportAny]
            case False:
                self.expr = expr

    def maybe_alias(self, expr: Expr) -> Expr:
        return expr if self.is_multi or not self.name else expr.alias(self.name)

    def implode_or_scalar(self) -> Expr:
        match self.is_pure_reducer:
            case True:
                expr = self.expr
            case False:
                expr = self.expr.pipe(
                    _resolve_exploded, is_distinct=self.has_projection_distinct
                )
        return expr.pipe(self.maybe_alias)

    def as_aliased(self, *, broadcast_agg: bool) -> Expr:
        return self.expr.pipe(
            lambda e: _broadcast_reducers(e) if broadcast_agg else e
        ).pipe(self.maybe_alias)

    def is_windowed(self, marker: Marker) -> bool:
        def _check_temp(col: exp.Column) -> bool:
            return (
                pc.Option
                .if_some(col.parts[-1])
                .map(lambda part: part.name == marker)
                .unwrap_or(default=False)
            )

        is_temp = self.expr.inner.pipe(_find_all, exp.Column).any(_check_temp)
        return self.name != marker and is_temp


@dataclass(slots=True, init=False)
class ExprPlan:
    cols: Cols
    projections: pc.Seq[ResolvedExpr]

    def __init__(
        self,
        cols: Cols,
        exprs: TryIter[IntoExpr],
        more_exprs: Iterable[IntoExpr],
        named_exprs: dict[str, IntoExpr],
    ) -> None:

        def _alias_named_expr(name: str, val: IntoExpr) -> IntoExpr:
            from ._expr import Expr

            match val:
                case Expr() as expr:
                    return expr.alias(name)
                case _:
                    return Expr.new(val, as_col=True).alias(name)

        def _resolve(val: IntoExpr) -> pc.Iter[ResolvedExpr]:
            from ._expr import Expr

            match val:
                case Expr() as expr:
                    return expr.meta.into_resolved(expr, cols)
                case _:
                    return (
                        Expr
                        .new(val, as_col=True)
                        .pipe(lambda e: ResolvedExpr(e, e.inner.output_name))
                        .into(pc.Iter.once)
                    )

        self.cols = cols
        self.projections = (
            try_iter(exprs)
            .chain(more_exprs)
            .chain(pc.Iter(named_exprs.items()).map_star(_alias_named_expr))
            .flat_map(_resolve)
            .collect()
        )

    def select_ctx(self) -> pc.Option[exp.Select]:
        def _non_empty_slct(projs: pc.Seq[ResolvedExpr], lf: exp.Expr) -> exp.Select:
            match projs.all(lambda r: r.has_projection_distinct):
                case True:
                    return (
                        self
                        .aliased_sql(broadcast_agg=False)
                        .into(lambda exprs: exp.select(*exprs).from_(lf))
                        .distinct()
                    )
                case False:
                    broadcast = projs.all(lambda r: r.is_pure_reducer)
                    return (
                        self
                        .aliased_sql(broadcast_agg=not broadcast)
                        .into(lambda exprs: exp.select(*exprs))
                        .from_(lf)
                    )

        return self.projections.then(
            lambda projs: _non_empty_slct(
                projs, exp.to_table("src").pipe(Marker.windowed, projs)
            )
        )

    def with_columns_ctx(self) -> exp.Select:
        def _resolve() -> pc.Iter[exp.Expr]:
            def _into_update(proj: ResolvedExpr) -> pc.Option[tuple[str, Expr]]:
                match proj.is_multi:
                    case True:
                        return pc.NONE
                    case False:
                        expr = _broadcast_reducers(proj.expr)
                        return pc.Some((proj.name, expr))

            def _resolved(updates: pc.Dict[str, Expr]) -> pc.Iter[exp.Expr]:

                match updates.any(lambda name: name in self.cols):
                    case False:
                        return (
                            updates
                            .items()
                            .iter()
                            .map_star(lambda name, e: e.alias(name).inner)
                            .insert(exp.Star())
                        )
                    case True:
                        return (
                            self.cols
                            .iter()
                            .map(
                                lambda name: updates.get_item(name).map_or(
                                    exp.column(name), lambda c: c.alias(name).inner
                                )
                            )
                            .chain(
                                updates
                                .items()
                                .iter()
                                .filter_star(lambda name, _expr: name not in self.cols)
                                .map_star(lambda name, e: e.alias(name).inner)
                            )
                        )

            return (
                self.projections
                .iter()
                .filter_map(_into_update)
                .collect(pc.Dict)
                .into(_resolved)
            )

        source = exp.to_table("src").pipe(Marker.windowed, self.projections)
        return exp.select(*_resolve()).from_(source)

    def with_fields_ctx(self, expr: Expr) -> Expr:
        return (
            self.projections
            .iter()
            .map(lambda proj: proj.as_aliased(broadcast_agg=False))
            .into(lambda args: expr.struct.insert(*args))
        )

    def group_by_all_ctx(self) -> exp.Select:
        return (
            self
            .aliased_sql(broadcast_agg=False)
            .into(lambda exprs: exp.select(*exprs))
            .from_("src")
            .group_by("ALL")
        )

    def aliased_sql(self, *, broadcast_agg: bool) -> pc.Iter[exp.Expr]:
        def _into_expr(resolved: ResolvedExpr) -> exp.Expr:
            return resolved.as_aliased(broadcast_agg=broadcast_agg).inner

        return self.projections.iter().map(_into_expr)

    def agg_ctx(self, keys: PyoIterable[exp.Expr]) -> exp.Select:
        def _lower_projection(proj: ResolvedExpr) -> pc.Iter[exp.Expr]:
            def _excluded(star: exp.Star) -> pc.Set[str]:
                return (
                    pc
                    .Option(star.args.get("except_"))
                    .map(pc.Iter[exp.Expr])
                    .unwrap_or_else(pc.Iter[exp.Expr].new)
                    .map(lambda e: e.name)
                    .collect(pc.Set)
                )

            def _into_glot(name: str) -> exp.Expr:
                from ._funcs import col

                return col(name).pipe(ResolvedExpr, name).implode_or_scalar().inner

            match proj.expr.inner:
                case exp.Star() as star:
                    excluded = _excluded(star)
                    return (
                        self.cols
                        .iter()
                        .filter(lambda name: name not in excluded)
                        .map(_into_glot)
                    )
                case exp.Explode(this=exp.Expr() as inner):
                    return pc.Iter.once(
                        proj.expr
                        .__class__(inner)
                        .pipe(
                            _resolve_exploded, is_distinct=proj.has_projection_distinct
                        )
                        .list.flatten()
                        .alias(proj.name)
                        .inner
                    )
                case _:
                    return pc.Iter.once(proj.implode_or_scalar().inner)

        plan = self.projections.iter().flat_map(_lower_projection)

        return (
            keys.iter().chain(plan).into(lambda exprs: exp.select(*exprs)).from_("src")
        )
