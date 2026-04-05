from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum, auto
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
    from pyochain.traits import PyoCollection, PyoIterable

    from .selectors import Selector
    from .sql.typing import IntoExpr, IntoExprColumn

type Cols = PyoCollection[str]


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


class ExprKind(IntEnum):
    ROW = auto()
    SCALAR = auto()
    WINDOW = auto()
    UNIQUE = auto()

    def resolve_exploded(self, expr: SqlExpr) -> SqlExpr:
        match self:
            case self.SCALAR:
                return expr
            case self.UNIQUE:
                return expr.implode().list.distinct()
            case _:
                return expr.implode()


@dataclass(slots=True)
class ExprMeta(ABC):
    """Metadata for expressions, used for tracking properties that affect query generation."""

    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)
    kind: ExprKind = ExprKind.ROW

    @abstractmethod
    def into_resolved(
        self, template: SqlExpr, cols: Cols, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]: ...

    def get_output_names(self, base_names: Cols, forced_name: pc.Option[str]) -> Cols:
        match forced_name:
            case pc.Some(name):
                return pc.Seq((name,))
            case _:
                match self.alias_name:
                    case pc.Some(alias_fn):
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
        output_names = self.get_output_names(pc.Seq((self.root_name,)), alias_override)
        return ResolvedExpr(template, output_names.first(), self.kind).into_iter()


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: Resolver = field(kw_only=True)
    preserve_native: bool = field(default=False, kw_only=True)

    @override
    def into_resolved(
        self, template: SqlExpr, cols: Cols, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:
        resolved_fn = partial(ResolvedExpr, kind=self.kind)

        def _get_builder() -> NamesBuilder:
            base_names = self.resolver(cols)
            output_names = self.get_output_names(base_names, alias_override)
            return NamesBuilder(base_names, output_names, template, resolved_fn)

        is_multi = (
            self.preserve_native
            and self.alias_name.is_none()
            and (isinstance(template.inner(), exp.Columns) or template.inner().is_star)
        )
        match (alias_override.is_none(), is_multi):
            case (True, True):
                return resolved_fn(template, template.get_name()).into_iter()
            case (True, _):
                return _get_builder().overriden()
            case (False, _):
                return _get_builder().not_overriden()


class NamesBuilder(NamedTuple):
    base: Cols
    output: Cols
    template: SqlExpr
    fn: partial[ResolvedExpr]

    def _to_resolved(self, name: str, output: str) -> ResolvedExpr:
        return self.fn(Marker.replace_col(self.template, name), output)

    def overriden(self) -> pc.Iter[ResolvedExpr]:
        return self.base.iter().zip(self.output).map_star(self._to_resolved)

    def not_overriden(self) -> pc.Iter[ResolvedExpr]:
        res = partial(self._to_resolved, output=self.output.first())
        return self.base.iter().map(res)


class ResolvedExpr(NamedTuple):
    """A fully resolved expression ready for SQL emission."""

    expr: SqlExpr
    name: str
    kind: ExprKind

    @classmethod
    def from_named(cls, val: IntoExpr, alias_override: pc.Option[str]) -> Self:
        resolved = sql.into_expr(val, as_col=True)
        output_name = alias_override.unwrap_or(resolved.get_name())
        return cls(resolved, output_name, kind=ExprKind.ROW)

    def is_multi(self) -> bool:
        return isinstance(self.expr.inner(), exp.Columns) or self.expr.inner().is_star

    def implode_or_scalar(self) -> SqlExpr:
        expr = self.kind.resolve_exploded(self.expr)
        return expr if self.is_multi() else expr.alias(self.name)

    def as_aliased(self) -> SqlExpr:
        return self.expr if self.is_multi() else self.expr.alias(self.name)

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

        return self.name != marker and pc.Iter(
            self.expr.inner().find_all(exp.Column)
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

    def aliased_sql(self) -> pc.Iter[Expression]:
        return (
            self.projections
            .iter()
            .map(ResolvedExpr.as_aliased)
            .map(lambda e: e.into_duckdb())
        )

    def select_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:
        def _non_empty_slct(
            projs: pc.Seq[ResolvedExpr], lf: DuckDBPyRelation
        ) -> DuckDBPyRelation:
            match projs.all(lambda r: r.kind == ExprKind.UNIQUE):
                case True:
                    return self.aliased_sql().into(
                        lambda exprs: lf.select(*exprs).distinct()
                    )
                case False:
                    match projs.all(lambda r: r.kind == ExprKind.SCALAR):
                        case True:
                            return self.aliased_sql().into(lf.aggregate)
                        case False:
                            return self.aliased_sql().into(
                                lambda exprs: lf.select(*exprs)
                            )

        return self.projections.then(
            lambda projs: _non_empty_slct(projs, Marker.windowed(lf, projs))
        ).unwrap_or_else(lambda: ScanSource.from_none().relation)

    def with_columns_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:

        def _resolve() -> pc.Iter[Expression]:
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
                .map(lambda r: (r.name, r.expr))
                .collect(pc.Dict)
                .into(_resolved)
                .map(lambda c: c.into_duckdb())
            )

        def _get_lf(lf: DuckDBPyRelation) -> DuckDBPyRelation:
            match self.projections.any(lambda r: r.kind == ExprKind.SCALAR):
                case True:
                    return _resolve().into(lf.aggregate)
                case False:
                    return _resolve().into(lambda exprs: lf.select(*exprs))

        return _get_lf(Marker.windowed(lf, self.projections))

    def with_fields_context(self, expr: SqlExpr) -> SqlExpr:
        return (
            self.projections
            .iter()
            .map(ResolvedExpr.as_aliased)
            .into(lambda args: expr.struct.insert(*args))
        )

    def group_by_all_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:
        return self.aliased_sql().into(lf.aggregate, "ALL")

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
                    ResolvedExpr(sql.col(name), name, proj.kind)
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
                        proj.kind
                        .resolve_exploded(SqlExpr(inner))
                        .list.flatten()
                        .alias(proj.name)
                        .into_duckdb()
                    )
                case _:
                    return pc.Iter.once(proj.implode_or_scalar().into_duckdb())

        plan = self.projections.iter().flat_map(_lower_projection)

        return keys.iter().chain(plan).into(aggregator)


@dataclass(slots=True, repr=False)
class Resolver:
    _fn: Callable[[Cols], Cols]

    @override
    def __repr__(self) -> str:
        fn = self._fn.__name__.replace("_", " ").title()
        return f"{self.__class__.__name__}({fn})"

    def __call__(self, cols: Cols) -> Cols:
        return self._fn(cols)

    def into_selector(self) -> Selector:
        from .selectors import Selector

        return Selector(sql.all(), MultiMeta(resolver=self))

    @classmethod
    def all_columns(cls) -> Self:
        def _all_columns(cols: Cols) -> Cols:
            return cols

        return cls(_all_columns)

    @classmethod
    def fixed(cls, names: Cols) -> Self:
        def _fixed(_: Cols) -> Cols:
            return names

        return cls(_fixed)

    @classmethod
    def all_fn(cls, exclude: pc.Option[TryIter[IntoExprColumn]]) -> Self:
        return exclude.map(
            lambda exc: (
                try_iter(exc)
                .map(lambda value: sql.into_expr(value, as_col=True))
                .filter_map(SqlExpr.root_column_name)
                .collect(pc.Set)
                .into(cls.exclude)
            )
        ).unwrap_or(cls.all_columns())

    @classmethod
    def exclude(cls, excluded: Cols) -> Self:
        def _exclude(cols: Cols) -> Cols:
            return cols.iter().filter(lambda n: n not in excluded).collect()

        return cls(_exclude)

    @classmethod
    def agg_expr(cls, cols: pc.Option[pc.Seq[str]]) -> Self:
        return cols.map(cls.fixed).unwrap_or(cls.all_columns())

    @classmethod
    def ordered_name(cls, names: Iterable[str]) -> Self:
        def _ordered(cols: Cols) -> Cols:
            return pc.Iter(names).filter(lambda name: name in cols).collect()

        return cls(_ordered)

    @classmethod
    def name(cls, predicate: Callable[[str], bool]) -> Self:
        def _name(cols: Cols) -> Cols:
            return cols.iter().filter(predicate).collect()

        return cls(_name)

    def difference(self, right_fn: Self) -> Self:
        def _difference(cols: Cols) -> Cols:
            right = right_fn(cols)
            return self(cols).iter().filter(lambda n: n not in right).collect()

        return self.__class__(_difference)

    def complement(self) -> Self:
        def _complement(cols: Cols) -> Cols:
            excluded = self(cols)
            return cols.iter().filter(lambda n: n not in excluded).collect()

        return self.__class__(_complement)

    def intersection(self, right: Self) -> Self:
        def _intersection(cols: Cols) -> Cols:
            right_set = right(cols)
            return self(cols).iter().filter(lambda n: n in right_set).collect()

        return self.__class__(_intersection)

    def union(self, right: Self) -> Self:
        def _union(cols: Cols) -> Cols:
            selected = self(cols).iter().chain(right(cols)).collect(pc.Set)
            return cols.iter().filter(lambda n: n in selected).collect()

        return self.__class__(_union)
