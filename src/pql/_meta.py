from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Self, override

import pyochain as pc
from sqlglot import exp

from . import sql
from .sql import ScanSource, SqlExpr
from .sql.utils import TryIter, try_iter

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation, Expression
    from narwhals.typing import IntoFrameT
    from pyochain.traits import PyoIterable

    from .selectors import Cols, Resolver
    from .sql.typing import IntoExpr


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
    def windowed(
        cls, lf: DuckDBPyRelation, cols: PyoIterable[ResolvedExpr]
    ) -> DuckDBPyRelation:
        match cols.any(lambda p: p.is_windowed(cls.TEMP)):
            case True:
                row_nb = sql.row_number().over().sub(1).alias(cls.TEMP).into_duckdb()
                return lf.select(row_nb, sql.all().into_duckdb())
            case False:
                return lf


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


def _is_projection_distinct(node: exp.Distinct) -> bool:
    return pc.Option(node.find_ancestor(exp.AggFunc, exp.List, exp.Window)).is_none()


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

    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)

    @abstractmethod
    def into_resolved(
        self, template: SqlExpr, cols: Cols, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]: ...

    def get_output_names(self, base_names: Cols, forced_name: pc.Option[str]) -> Cols:
        match forced_name, self.alias_name:
            case pc.Some(name), _:
                return pc.Seq((name,))
            case _, pc.Some(alias_fn):
                return base_names.iter().map(alias_fn).collect()
            case _:
                return base_names

    def with_alias_mapper(self, mapper: Callable[[str], str]) -> Self:
        def _get_mapper() -> Callable[[str], str]:
            match self.alias_name:
                case pc.Some(current):
                    return lambda name: mapper(current(name))
                case _:
                    return mapper

        return replace(self, alias_name=pc.Some(_get_mapper()))

    def clear_alias(self) -> Self:
        return replace(self, alias_name=pc.NONE)


@dataclass(slots=True)
class SingleMeta(ExprMeta):
    root_name: str = field(kw_only=True)

    @override
    def into_resolved(
        self, template: SqlExpr, cols: Cols, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:
        match template.inner():
            case exp.Alias():
                base_name = template.get_name()
            case _:
                base_name = self.root_name

        output_name = self.get_output_names(
            pc.Seq((base_name,)), alias_override
        ).first()
        return ResolvedExpr(template, output_name).into_iter()


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: Resolver = field(kw_only=True)
    preserve_native: bool = field(default=False, kw_only=True)

    @override
    def into_resolved(
        self, template: SqlExpr, cols: Cols, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:

        def _get_builder() -> NamesBuilder:
            base_names = self.resolver(cols)
            output_names = self.get_output_names(base_names, alias_override)
            return NamesBuilder(base_names, output_names, template)

        match alias_override.is_none(), self._is_multi(template):
            case True, True:
                return ResolvedExpr(template, template.get_name()).into_iter()
            case True, _:
                return _get_builder().overriden()
            case False, _:
                return _get_builder().not_overriden()

    def _is_multi(self, template: SqlExpr) -> bool:
        return (
            self.preserve_native
            and self.alias_name.is_none()
            and (isinstance(template.inner(), exp.Columns) or template.inner().is_star)
        )


class NamesBuilder(NamedTuple):
    base: Cols
    output: Cols
    template: SqlExpr

    def _to_resolved(self, name: str, output: str) -> ResolvedExpr:
        return ResolvedExpr(Marker.replace_col(self.template, name), output)

    def overriden(self) -> pc.Iter[ResolvedExpr]:
        return self.base.iter().zip(self.output).map_star(self._to_resolved)

    def not_overriden(self) -> pc.Iter[ResolvedExpr]:
        res = partial(self._to_resolved, output=self.output.first())
        return self.base.iter().map(res)


def _find_all(expr: exp.Expr, *exprs: type[exp.Expr]) -> pc.Iter[exp.Expr]:
    return pc.Iter(expr.find_all(*exprs))


class ResolvedExpr(NamedTuple):
    """A fully resolved expression ready for SQL emission."""

    expr: SqlExpr
    name: str

    @classmethod
    def from_named(cls, val: IntoExpr, alias_override: pc.Option[str]) -> Self:
        resolved = SqlExpr.new(val, as_col=True)
        output_name = alias_override.unwrap_or(resolved.get_name())
        return cls(resolved, output_name)

    def is_multi(self) -> bool:
        return isinstance(self.expr.inner(), exp.Columns) or self.expr.inner().is_star

    def has_projection_distinct(self) -> bool:
        return pc.Iter(self.expr.inner().find_all(exp.Distinct)).any(
            _is_projection_distinct
        )

    def lowered_expr(self) -> SqlExpr:
        def _strip(node: exp.Expr) -> exp.Expr:
            match node:
                case exp.Distinct(expressions=[exp.Expr() as inner]) if (
                    _is_projection_distinct(node)
                ):
                    return inner
                case _:
                    return node

        return SqlExpr(self.expr.inner().transform(_strip))  # pyright: ignore[reportUnknownMemberType, reportAny]

    def is_pure_reducer(self) -> bool:
        search = partial(_find_all, self.lowered_expr().inner())

        return search(exp.AggFunc, exp.List).any(
            lambda node: not _has_window_ancestor(node)
        ) and not search(exp.Column).any(
            lambda col: col.find_ancestor(exp.AggFunc, exp.List, exp.Window) is None
        )

    def implode_or_scalar(self) -> SqlExpr:
        expr = self.lowered_expr()
        match (self.is_pure_reducer(), self.is_multi()):
            case (True, True):
                return expr
            case (False, True):
                return expr.pipe(
                    _resolve_exploded, is_distinct=self.has_projection_distinct()
                )
            case (True, False):
                return expr.alias(self.name)
            case (False, False):
                return expr.pipe(
                    _resolve_exploded, is_distinct=self.has_projection_distinct()
                ).alias(self.name)

    def as_aliased(self, *, broadcast_agg: bool) -> SqlExpr:
        return (
            self
            .lowered_expr()
            .pipe(lambda e: _broadcast_reducers(e) if broadcast_agg else e)
            .pipe(lambda e: e if self.is_multi() else e.alias(self.name))
        )

    def into_iter(self) -> pc.Iter[Self]:
        return pc.Iter.once(self)

    def is_windowed(self, marker: Marker) -> bool:
        def _check_temp(col: exp.Column) -> bool:
            return (
                pc.Option
                .if_some(col.parts[-1])
                .map(lambda part: part.name == marker)
                .unwrap_or(default=False)
            )

        return self.name != marker and self.expr.inner().pipe(
            lambda e: pc.Iter(e.find_all(exp.Column))
        ).any(_check_temp)


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

        def _resolve(
            val: IntoExpr, alias_override: pc.Option[str] = pc.NONE
        ) -> pc.Iter[ResolvedExpr]:
            from ._expr import Expr

            match val:
                case Expr() as expr:
                    return expr.meta.into_resolved(expr.inner(), cols, alias_override)
                case _:
                    return ResolvedExpr.from_named(val, alias_override).into_iter()

        self.cols = cols
        self.projections = (
            try_iter(exprs)
            .chain(more_exprs)
            .flat_map(_resolve)
            .chain(
                pc
                .Iter(named_exprs.items())
                .map_star(lambda k, v: _resolve(v, pc.Some(k)))
                .flatten()
            )
            .collect()
        )

    def aliased_sql(self, *, broadcast_agg: bool) -> pc.Iter[Expression]:
        def _into_expr(resolved: ResolvedExpr) -> Expression:
            return resolved.as_aliased(broadcast_agg=broadcast_agg).into_duckdb()

        return self.projections.iter().map(_into_expr)

    def select_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:
        def _non_empty_slct(
            projs: pc.Seq[ResolvedExpr], lf: DuckDBPyRelation
        ) -> DuckDBPyRelation:
            match projs.all(ResolvedExpr.has_projection_distinct):
                case True:
                    return (
                        self
                        .aliased_sql(broadcast_agg=False)
                        .into(lambda exprs: lf.select(*exprs))
                        .distinct()
                    )
                case False:
                    match projs.all(lambda r: r.is_pure_reducer()):
                        case True:
                            return self.aliased_sql(broadcast_agg=False).into(
                                lf.aggregate
                            )
                        case False:
                            return self.aliased_sql(broadcast_agg=True).into(
                                lambda exprs: lf.select(*exprs)
                            )

        return self.projections.then(
            lambda projs: _non_empty_slct(projs, Marker.windowed(lf, projs))
        ).unwrap_or_else(lambda: ScanSource.from_none().relation)

    def with_columns_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:

        def _resolve() -> pc.Iter[Expression]:
            def _into_update(proj: ResolvedExpr) -> pc.Option[tuple[str, SqlExpr]]:
                match proj.is_multi():
                    case True:
                        return pc.NONE
                    case False:
                        expr = _broadcast_reducers(proj.expr)
                        return pc.Some((proj.name, expr))

            def _resolved(updates: pc.Dict[str, SqlExpr]) -> pc.Iter[SqlExpr]:
                match updates.any(lambda name: name in self.cols):
                    case False:
                        return (
                            updates
                            .items()
                            .iter()
                            .map_star(lambda name, e: e.alias(name))
                            .insert(sql.all())
                        )
                    case True:
                        return (
                            self.cols
                            .iter()
                            .map(
                                lambda name: updates.get_item(name).map_or(
                                    sql.col(name), lambda c: c.alias(name)
                                )
                            )
                            .chain(
                                updates
                                .items()
                                .iter()
                                .filter_star(lambda name, _expr: name not in self.cols)
                                .map_star(lambda name, e: e.alias(name))
                            )
                        )

            return (
                self.projections
                .iter()
                .filter_map(_into_update)
                .collect(pc.Dict)
                .into(_resolved)
                .map(lambda c: c.into_duckdb())
            )

        return Marker.windowed(lf, self.projections).select(*_resolve())

    def with_fields_context(self, expr: SqlExpr) -> SqlExpr:
        return (
            self.projections
            .iter()
            .map(lambda proj: proj.as_aliased(broadcast_agg=False))
            .into(lambda args: expr.struct.insert(*args))
        )

    def group_by_all_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:
        return self.aliased_sql(broadcast_agg=False).into(lf.aggregate, "ALL")

    def agg_context(
        self,
        keys: PyoIterable[Expression],
        aggregator: Callable[[pc.Iter[Expression]], DuckDBPyRelation],
    ) -> DuckDBPyRelation:
        def _lower_projection(proj: ResolvedExpr) -> pc.Iter[Expression]:
            def _excluded(star: exp.Star) -> pc.Set[str]:
                return (
                    pc
                    .Option(star.args.get("except_"))
                    .map(pc.Iter[exp.Expr])
                    .unwrap_or_else(pc.Iter[exp.Expr].new)
                    .filter_map(lambda e: SqlExpr(e).root_column_name())
                    .collect(pc.Set)
                )

            def _into_duck(name: str) -> Expression:
                return (
                    sql
                    .col(name)
                    .pipe(ResolvedExpr, name)
                    .implode_or_scalar()
                    .into_duckdb()
                )

            match proj.expr.inner():
                case exp.Star() as star:
                    excluded = _excluded(star)
                    return (
                        self.cols
                        .iter()
                        .filter(lambda name: name not in excluded)
                        .map(_into_duck)
                    )
                case exp.Explode(this=exp.Expr() as inner):
                    return pc.Iter.once(
                        SqlExpr(inner)
                        .pipe(
                            _resolve_exploded,
                            is_distinct=proj.has_projection_distinct(),
                        )
                        .list.flatten()
                        .alias(proj.name)
                        .into_duckdb()
                    )
                case _:
                    return pc.Iter.once(proj.implode_or_scalar().into_duckdb())

        plan = self.projections.iter().flat_map(_lower_projection)

        return keys.iter().chain(plan).into(aggregator)
