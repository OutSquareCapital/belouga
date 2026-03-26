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
from ._schema import Schema
from .sql import SqlExpr
from .sql.utils import TryIter, try_chain, try_iter

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation, Expression
    from narwhals.typing import IntoFrameT
    from pyochain.traits import PyoCollection, PyoIterable, PyoKeysView

    from ._datatypes import DataType
    from .sql.typing import IntoExpr, IntoExprColumn


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
    def empty_frame(cls) -> DuckDBPyRelation:
        return sql.into_relation({cls.EMPTY: ()})

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


type ResolverFn = Callable[[Schema], PyoCollection[str]]


class ExprKind(IntEnum):
    ROW = auto()
    SCALAR = auto()
    WINDOW = auto()
    UNIQUE = auto()


@dataclass(slots=True)
class ExprMeta(ABC):
    """Metadata for expressions, used for tracking properties that affect query generation."""

    alias_name: pc.Option[Callable[[str], str]] = field(default_factory=lambda: pc.NONE)
    kind: ExprKind = ExprKind.ROW

    @abstractmethod
    def into_resolved(
        self, template: SqlExpr, schema: Schema, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]: ...

    def get_output_names(
        self, base_names: PyoCollection[str], forced_name: pc.Option[str]
    ) -> PyoCollection[str]:
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
        self, template: SqlExpr, schema: Schema, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:
        output_names = self.get_output_names(pc.Seq((self.root_name,)), alias_override)
        return ResolvedExpr(template, output_names.first(), self.kind).into_iter()


@dataclass(slots=True)
class MultiMeta(ExprMeta):
    resolver: ResolverFn = field(kw_only=True)
    preserve_native: bool = field(default=False, kw_only=True)

    @override
    def into_resolved(
        self, template: SqlExpr, schema: Schema, alias_override: pc.Option[str]
    ) -> pc.Iter[ResolvedExpr]:
        resolved_fn = partial(ResolvedExpr, kind=self.kind)

        def _get_builder() -> NamesBuilder:
            base_names = self.resolver(schema)
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
    base: PyoCollection[str]
    output: PyoCollection[str]
    template: SqlExpr
    fn: partial[ResolvedExpr]

    def _to_resolved(self, name: str, output: str) -> ResolvedExpr:
        return self.fn(Marker.replace_col(self.template, name), output)

    def overriden(self) -> pc.Iter[ResolvedExpr]:
        return self.base.iter().zip(self.output).map_star(self._to_resolved)

    def not_overriden(self) -> pc.Iter[ResolvedExpr]:
        _res = partial(self._to_resolved, output=self.output.first())
        return self.base.iter().map(_res)


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
        match self.kind:
            case ExprKind.SCALAR:
                expr = self.expr
            case ExprKind.UNIQUE:
                expr = self.expr.implode().list.distinct()
            case _:
                expr = self.expr.implode()
        return expr if self.is_multi() else expr.alias(self.name)

    def as_aliased(self) -> SqlExpr:
        return self.expr if self.is_multi() else self.expr.alias(self.name)

    def into_iter(self) -> pc.Iter[Self]:
        return pc.Iter.once(self)

    def is_windowed(self, marker: Marker) -> bool:
        def _check_temp(col: exp.Column) -> bool:
            return (
                pc.Option.if_some(col.parts[-1])
                .map(lambda part: part.name == marker)
                .unwrap_or(default=False)
            )

        return self.name != marker and pc.Iter(
            self.expr.inner().find_all(exp.Column)
        ).any(_check_temp)


@dataclass(slots=True, init=False)
class ExprPlan:
    cols: PyoKeysView[str]
    projections: pc.Seq[ResolvedExpr]

    def __init__(
        self,
        schema: Schema,
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
                    return expr.meta.into_resolved(expr.inner(), schema, alias_override)
                case _:
                    return ResolvedExpr.from_named(val, alias_override).into_iter()

        self.cols = schema.keys()
        expr_map = (
            pc.Iter(named_exprs.items())
            .map_star(lambda k, v: _resolve(v, pc.Some(k)))
            .flatten()
            .collect()
        )
        self.projections = (
            try_chain(exprs, more_exprs).flat_map(_resolve).chain(expr_map).collect()
        )

    def aliased_sql(self) -> pc.Iter[Expression]:
        return (
            self.projections.iter()
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
        ).unwrap_or_else(Marker.empty_frame)

    def with_columns_context(self, lf: DuckDBPyRelation) -> DuckDBPyRelation:

        def _resolve() -> pc.Iter[Expression]:
            def _resolved(updates: pc.Dict[str, SqlExpr]) -> pc.Iter[SqlExpr]:
                match updates.any(lambda name: name in self.cols):
                    case False:
                        return (
                            updates.items()
                            .iter()
                            .map_star(lambda name, e: e.alias(name))
                            .insert(sql.all())
                        )
                    case True:
                        return (
                            self.cols.iter()
                            .map(
                                lambda name: updates.get_item(name).map_or(
                                    sql.col(name), lambda c: c.alias(name)
                                )
                            )
                            .chain(
                                updates.items()
                                .iter()
                                .filter_star(lambda name, _expr: name not in self.cols)
                                .map_star(lambda name, e: e.alias(name))
                            )
                        )

            return (
                self.projections.iter()
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
            self.projections.iter()
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
                    pc.Option(star.args.get("except_"))
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
                        self.cols.iter()
                        .filter(lambda name: name not in excluded)
                        .map(_into_duck)
                    )
                case exp.Explode(this=exp.Expr() as inner):
                    match proj.kind:
                        case ExprKind.SCALAR:
                            expr = SqlExpr(inner)
                        case ExprKind.UNIQUE:
                            expr = SqlExpr(inner).implode().list.distinct()
                        case _:
                            expr = SqlExpr(inner).implode()
                    return pc.Iter.once(
                        expr.list.flatten().alias(proj.name).into_duckdb()
                    )
                case _:
                    return pc.Iter.once(proj.implode_or_scalar().into_duckdb())

        plan = self.projections.iter().flat_map(_lower_projection)

        return keys.iter().chain(plan).into(aggregator)


class Resolver:
    """Namespace class for resolver functions used in multi expressions."""

    @staticmethod
    def all_columns() -> ResolverFn:
        return Schema.keys

    @staticmethod
    def all_fn(exclude: pc.Option[TryIter[IntoExprColumn]]) -> ResolverFn:
        return exclude.map(
            lambda exc: (
                try_iter(exc)
                .map(lambda value: sql.into_expr(value, as_col=True))
                .filter_map(SqlExpr.root_column_name)
                .collect(pc.Set)
                .into(Resolver.exclude)
            )
        ).unwrap_or(Resolver.all_columns())

    @staticmethod
    def exclude(excluded: pc.Set[str]) -> ResolverFn:
        return lambda schema: (
            schema.iter().filter(lambda n: n not in excluded).collect()
        )

    @staticmethod
    def agg_expr(cols: pc.Option[pc.Seq[str]]) -> ResolverFn:
        return cols.map(Resolver.fixed).unwrap_or(Resolver.all_columns())

    @staticmethod
    def fixed(names: pc.Seq[str]) -> ResolverFn:
        return lambda _schema: names

    @staticmethod
    def ordered_name(names: pc.Seq[str]) -> ResolverFn:
        return lambda schema: names.iter().filter(lambda name: name in schema).collect()

    @staticmethod
    def dtype(*on: type[DataType]) -> ResolverFn:
        return lambda schema: (
            schema.items()
            .iter()
            .filter_star(lambda _, dtype: isinstance(dtype, on))
            .map_star(lambda name, _: name)
            .collect()
        )

    @staticmethod
    def name(predicate: Callable[[str], bool]) -> ResolverFn:
        return lambda schema: schema.iter().filter(predicate).collect()

    @staticmethod
    def difference(left: ResolverFn, right_fn: ResolverFn) -> ResolverFn:
        def _fn(schema: Schema) -> PyoCollection[str]:
            right = right_fn(schema)
            return left(schema).iter().filter(lambda n: n not in right).collect()

        return _fn

    @staticmethod
    def complement(resolver: ResolverFn) -> ResolverFn:
        def _fn(schema: Schema) -> pc.Seq[str]:
            excluded = resolver(schema)
            return schema.iter().filter(lambda n: n not in excluded).collect()

        return _fn

    @staticmethod
    def intersection(left: ResolverFn, right: ResolverFn) -> ResolverFn:
        def _fn(schema: Schema) -> pc.Seq[str]:
            right_set = right(schema)
            return left(schema).iter().filter(lambda n: n in right_set).collect()

        return _fn

    @staticmethod
    def union(left: ResolverFn, right: ResolverFn) -> ResolverFn:
        def _fn(schema: Schema) -> pc.Seq[str]:
            selected = left(schema).iter().chain(right(schema)).collect(pc.Set)
            return schema.iter().filter(lambda n: n in selected).collect()

        return _fn
