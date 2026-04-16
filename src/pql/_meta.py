from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, Self, override

import pyochain as pc
from pyochain.traits import Pipeable
from sqlglot import exp

from . import sql
from .sql import SqlExpr
from .sql.utils import TryIter, try_iter

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT
    from pyochain.traits import PyoIterable

    from .selectors import Cols, Resolver
    from .sql.typing import IntoExpr

type Aliaser = Callable[[str], str]
"""Alias function type, used for generating deferred column aliases."""


class Marker(StrEnum):
    """Column name markers for special expression types."""

    ELEMENT = auto()
    LIT = "literal"
    LEN = "len"
    EMPTY = "__pql_empty__"
    """Marker for empty `LazyFrames`.

    DuckDB doesn't allow empty `DuckDBPyRelation`, so we need to create an empty column that is cleaned up afterwards if we want to convert to another type of empty frame."""
    TEMP = "__pql_temp__"

    IDX = "__pql_idx__"

    def to_expr(self) -> SqlExpr:
        return sql.col(self.value)

    @classmethod
    def replace_col(cls, template: SqlExpr, column_name: str) -> SqlExpr:
        target = exp.column(column_name)

        def _replacer(node: exp.Expr) -> exp.Expr:
            match node:
                case exp.Star() | exp.Columns():
                    return target
                case _:
                    return node

        return SqlExpr(template.inner().transform(_replacer))  # pyright: ignore[reportUnknownMemberType, reportAny]

    @classmethod
    def drop_marker(cls, result: IntoFrameT, cols: Collection[str]) -> IntoFrameT:
        import narwhals as nw

        match cls.EMPTY in cols:
            case True:
                return nw.from_native(result).drop(cls.EMPTY).to_native()
            case False:
                return result

    @classmethod
    def windowed(cls, source: exp.Expr, cols: PyoIterable[ResolvedExpr]) -> exp.Expr:
        match cols.any(lambda p: p.is_windowed(cls.TEMP)):
            case True:
                row_nb = sql.row_number().over().sub(1).alias(cls.TEMP).inner()
                return exp.select(row_nb, "*").from_(source).subquery("src")
            case False:
                return source


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


def _broadcast_reducers(expr: SqlExpr) -> SqlExpr:
    def _window_agg(node: exp.Expr) -> exp.Expr:
        match node:
            case exp.AggFunc() | exp.List() if not _has_window_ancestor(node):
                return SqlExpr(node).over().inner()
            case _:
                return node

    return SqlExpr(expr.inner().transform(_window_agg))  # pyright: ignore[reportUnknownMemberType, reportAny]


def _resolve_exploded(expr: SqlExpr, *, is_distinct: bool) -> SqlExpr:
    match is_distinct:
        case True:
            return expr.implode().list.distinct()
        case False:
            return expr.implode()


@dataclass(slots=True)
class ExprMeta(ABC):
    """Metadata for expressions, used for tracking properties that affect query generation."""

    alias_name: pc.Option[Aliaser] = field(default_factory=lambda: pc.NONE)

    @abstractmethod
    def into_resolved(self, template: SqlExpr, cols: Cols) -> pc.Iter[ResolvedExpr]: ...

    def get_output_names(self, base_names: Cols, template: SqlExpr) -> Cols:
        match template.inner(), self.alias_name:
            case exp.Alias() as expr, pc.Some(alias_fn):
                return pc.Iter.once(expr.output_name).map(alias_fn).collect()
            case exp.Alias() as expr, pc.NONE:
                return pc.Seq((expr.output_name,))
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
class SingleMeta(ExprMeta):
    root_name: str = field(kw_only=True)

    @override
    def into_resolved(self, template: SqlExpr, cols: Cols) -> pc.Iter[ResolvedExpr]:
        name = pc.Seq((self.root_name,)).into(self.get_output_names, template).first()
        return ResolvedExpr(template, name).into(pc.Iter.once)


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: Resolver = field(kw_only=True)

    @override
    def into_resolved(self, template: SqlExpr, cols: Cols) -> pc.Iter[ResolvedExpr]:
        base_names = self.resolver(cols)
        output_names = self.get_output_names(base_names, template)

        def _resolved(expr: SqlExpr, col_name: str, name: str) -> ResolvedExpr:
            return ResolvedExpr(Marker.replace_col(expr, col_name), name)

        match template.inner():
            case exp.Alias():
                expr = template.inner().unalias().pipe(SqlExpr)
                alias = output_names.first()
                return base_names.iter().map(lambda name: _resolved(expr, name, alias))

            case _:
                return (
                    base_names
                    .iter()
                    .zip(output_names)
                    .map_star(lambda name, output: _resolved(template, name, output))
                )


def _find_all[T: exp.Expr](expr: exp.Expr, *exprs: type[T]) -> pc.Iter[T]:
    return pc.Iter(expr.find_all(*exprs))


@dataclass(slots=True, init=False)
class ResolvedExpr(Pipeable):
    """A fully resolved expression ready for SQL emission."""

    expr: SqlExpr
    name: str
    has_projection_distinct: bool
    is_pure_reducer: bool
    is_multi: bool

    def __init__(self, expr: SqlExpr, name: str) -> None:
        self.name = name
        inner = expr.inner()
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

                self.expr = SqlExpr(inner.transform(_strip))  # pyright: ignore[reportUnknownMemberType, reportAny]
            case False:
                self.expr = expr

    def maybe_alias(self, expr: SqlExpr) -> SqlExpr:
        return expr if self.is_multi or not self.name else expr.alias(self.name)

    def implode_or_scalar(self) -> SqlExpr:
        match self.is_pure_reducer:
            case True:
                expr = self.expr
            case False:
                expr = self.expr.pipe(
                    _resolve_exploded, is_distinct=self.has_projection_distinct
                )
        return expr.pipe(self.maybe_alias)

    def as_aliased(self, *, broadcast_agg: bool) -> SqlExpr:
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

        is_temp = self.expr.inner().pipe(_find_all, exp.Column).any(_check_temp)
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
        from ._expr import Expr

        def _alias_named_expr(name: str, val: IntoExpr) -> IntoExpr:
            match val:
                case Expr() as expr:
                    return expr.alias(name)
                case _:
                    return SqlExpr.new(val, as_col=True).alias(name)

        def _resolve(val: IntoExpr) -> pc.Iter[ResolvedExpr]:
            match val:
                case Expr() as expr:
                    return expr.meta.into_resolved(expr.inner(), cols)
                case _:
                    return (
                        SqlExpr
                        .new(val, as_col=True)
                        .pipe(lambda e: ResolvedExpr(e, e.inner().output_name))
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

    def aliased_sql(self, *, broadcast_agg: bool) -> pc.Iter[exp.Expr]:
        def _into_expr(resolved: ResolvedExpr) -> exp.Expr:
            return resolved.as_aliased(broadcast_agg=broadcast_agg).inner()

        return self.projections.iter().map(_into_expr)

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
            def _into_update(proj: ResolvedExpr) -> pc.Option[tuple[str, SqlExpr]]:
                match proj.is_multi:
                    case True:
                        return pc.NONE
                    case False:
                        expr = _broadcast_reducers(proj.expr)
                        return pc.Some((proj.name, expr))

            def _resolved(updates: pc.Dict[str, SqlExpr]) -> pc.Iter[exp.Expr]:
                match updates.any(lambda name: name in self.cols):
                    case False:
                        return (
                            updates
                            .items()
                            .iter()
                            .map_star(lambda name, e: e.alias(name).inner())
                            .insert(exp.Star())
                        )
                    case True:
                        return (
                            self.cols
                            .iter()
                            .map(
                                lambda name: updates.get_item(name).map_or(
                                    sql.col(name).inner(),
                                    lambda c: c.alias(name).inner(),
                                )
                            )
                            .chain(
                                updates
                                .items()
                                .iter()
                                .filter_star(lambda name, _expr: name not in self.cols)
                                .map_star(lambda name, e: e.alias(name).inner())
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

    def with_fields_ctx(self, expr: SqlExpr) -> SqlExpr:
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
                return (
                    sql.col(name).pipe(ResolvedExpr, name).implode_or_scalar().inner()
                )

            match proj.expr.inner():
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
                        SqlExpr(inner)
                        .pipe(
                            _resolve_exploded, is_distinct=proj.has_projection_distinct
                        )
                        .list.flatten()
                        .alias(proj.name)
                        .inner()
                    )
                case _:
                    return pc.Iter.once(proj.implode_or_scalar().inner())

        plan = self.projections.iter().flat_map(_lower_projection)

        return (
            keys.iter().chain(plan).into(lambda exprs: exp.select(*exprs)).from_("src")
        )
