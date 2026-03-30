"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, SupportsInt

import pyochain as pc
from duckdb import DuckDBPyRelation, Expression
from sqlglot import exp

from . import sql
from ._funcs import col
from ._joins import JoinBuilder, JoinKeys
from ._meta import ExprPlan, Marker
from ._schema import Schema
from .sql import SqlExpr
from .sql.utils import TryIter, TrySeq, check_by_arg, try_chain, try_iter, try_seq

if TYPE_CHECKING:
    import polars as pl
    from _duckdb import ExplainType
    from _duckdb._enums import (  # pyright: ignore[reportMissingModuleSource]
        ExplainTypeLiteral,
        RenderModeLiteral,
    )
    from pyochain.traits import PyoIterable

    from ._datatypes import DataType
    from ._expr import Expr
    from ._groupby import LazyGroupBy
    from ._parser import ParsedQuery
    from ._typing import (
        AsofJoinStrategy,
        GroupByClause,
        JoinStrategy,
        PivotAgg,
        UniqueKeepStrategy,
    )
    from .sql.typing import (
        FillNullStrategy,
        IntoExpr,
        IntoExprColumn,
        IntoRel,
        Orientation,
        ParquetCompression,
        PythonLiteral,
    )

MAX_I64 = 9_223_372_036_854_775_807
PIVOT_AGG: dict[PivotAgg, Callable[[SqlExpr], SqlExpr]] = {
    "min": SqlExpr.min,
    "max": SqlExpr.max,
    "first": SqlExpr.first,
    "last": SqlExpr.last,
    "sum": SqlExpr.sum,
    "mean": SqlExpr.mean,
    "median": SqlExpr.median,
    "len": SqlExpr.count,
    "count": SqlExpr.count,
}


@dataclass(slots=True, init=False, repr=False)
class LazyFrame(sql.CoreHandler[DuckDBPyRelation]):
    """LazyFrame providing Polars-like API over DuckDB relations."""

    _inner: DuckDBPyRelation
    _cached_schema: pc.Option[Schema]

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._cached_schema = pc.NONE
        self._inner = sql.into_relation(data, orient)

    def _from_sql_expr(self, expr: exp.Expr, **kwargs: IntoRel) -> Self:
        qry = sql.from_query(expr.sql(dialect="duckdb"), **kwargs)
        return self.__class__(qry)

    def _iter_slct(self, func: Callable[[str], SqlExpr]) -> Self:
        return self.select(self.columns.iter().map(func))

    def _iter_agg(self, func: Callable[[SqlExpr], SqlExpr]) -> Self:
        return self._new(
            self.columns
            .iter()
            .map(lambda c: func(sql.col(c)).alias(c).into_duckdb())
            .into(lambda exprs: self.inner().aggregate(exprs))
        )

    def lazy(self) -> pl.LazyFrame:
        """Get a Polars LazyFrame."""
        return self.inner().pl(lazy=True).pipe(Marker.drop_marker, self.columns)

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame."""
        return self.inner().pl().pipe(Marker.drop_marker, self.columns)

    def select(
        self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr, **named_exprs: IntoExpr
    ) -> Self:
        """Select columns or expressions."""
        return self._new(
            self.schema.into(ExprPlan, exprs, more_exprs, named_exprs).select_context(
                self.inner()
            )
        )

    def with_columns(
        self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr, **named_exprs: IntoExpr
    ) -> Self:
        """Add or replace columns."""
        return self._new(
            self.schema.into(
                ExprPlan, exprs, more_exprs, named_exprs
            ).with_columns_context(self.inner())
        )

    def filter(
        self,
        predicates: TryIter[IntoExprColumn],
        *more_predicates: IntoExprColumn,
        **constraints: IntoExpr,
    ) -> Self:
        """Filter rows based on predicates and equality constraints."""

        def _constraint(k: str, val: IntoExpr) -> SqlExpr:
            return sql.col(k).eq(sql.into_expr(val))

        expr = (
            try_chain(predicates, more_predicates)
            .map(lambda value: sql.into_expr(value, as_col=True))
            .chain(pc.Iter(constraints.items()).map_star(_constraint))
            .into(sql.reduce, SqlExpr.and_)
            .into_duckdb()
        )
        return self._new(self.inner().filter(expr))

    def group_by(
        self,
        keys: TryIter[IntoExpr] = None,
        *more_keys: IntoExpr,
        drop_null_keys: bool = False,
        strategy: GroupByClause | None = None,
    ) -> LazyGroupBy:
        """Start a group by operation."""
        from ._groupby import LazyGroupBy

        key_exprs = (
            try_chain(keys, more_keys)
            .map(lambda key: sql.into_expr(key, as_col=True))
            .collect()
        )
        grouped_frame = (
            key_exprs.iter().map(lambda key: key.is_not_null()).into(self.filter)
            if drop_null_keys
            else self
        )

        def _group_strat(strat: pc.Option[str]) -> pc.Option[str]:
            match strat:
                case pc.Some(s):
                    return pc.Some(f"{s} ({key_exprs.iter().map(str).join(', ')})")
                case _:
                    return strat

        return LazyGroupBy(
            grouped_frame,
            key_exprs,
            group_expr=pc.Option(strategy).into(_group_strat),
        )

    def group_by_all(
        self,
        exprs: TryIter[IntoExpr] = None,
        *more_exprs: IntoExpr,
        **named_exprs: IntoExpr,
    ) -> Self:
        """Aggregate with GROUP BY ALL — DuckDB auto-detects grouping keys."""
        return self._new(
            self.schema.into(
                ExprPlan, exprs, more_exprs, named_exprs
            ).group_by_all_context(self.inner())
        )

    def sort(
        self,
        by: TryIter[IntoExpr],
        *more_by: IntoExpr,
        descending: TrySeq[bool] = False,
        nulls_last: TrySeq[bool] = False,
    ) -> Self:
        """Sort by columns."""
        return self._new(
            try_chain(by, more_by)
            .map(lambda v: sql.into_expr(v, as_col=True))
            .collect()
            .into(
                lambda sort_exprs: sort_exprs.iter().zip(
                    check_by_arg(sort_exprs, "descending", arg=descending).unwrap(),
                    check_by_arg(sort_exprs, "nulls_last", arg=nulls_last).unwrap(),
                )
            )
            .map_star(
                lambda expr, desc, nls: expr.set_order(
                    desc=desc, nulls_last=nls
                ).into_duckdb()
            )
            .into(lambda x: self.inner().sort(*x))
        )

    def limit(self, n: int) -> Self:
        """Limit the number of rows."""
        return self._new(self.inner().limit(n))

    def head(self, n: int = 5) -> Self:
        """Get the first n rows."""
        return self.limit(n)

    def slice(self, offset: int, length: int | None = None) -> Self:
        """Get a slice of rows."""

        def _with_idx_and_len() -> Self:
            return self.with_columns(
                sql.row_number().over().sub(1).alias(Marker.IDX),
                sql.lit(1).count().over().alias(Marker.LEN),
            )

        def _from_end_start(off: int) -> SqlExpr:
            return Marker.IDX.to_expr().ge(Marker.LEN.to_expr().add(off))

        def _filter_lf(
            lf_length: pc.Option[int], offset: int
        ) -> pc.Result[Self, ValueError]:
            match (lf_length, offset):
                case (pc.Some(length), _) if length < 0:
                    msg = f"negative slice lengths ({length}) are invalid for LazyFrame"
                    return pc.Err(ValueError(msg))
                case (len_val, offset) if offset >= 0:
                    rel = self.inner().limit(len_val.unwrap_or(MAX_I64), offset=offset)
                    return pc.Ok(self._new(rel))
                case (pc.Some(0), _):
                    return pc.Ok(self.limit(0))
                case (pc.Some(length), offset):
                    return pc.Ok(
                        _with_idx_and_len()
                        .filter(
                            _from_end_start(offset).and_(
                                Marker.IDX.to_expr().lt(
                                    Marker.LEN.to_expr().add(offset).add(length)
                                )
                            )
                        )
                        .drop(Marker.IDX, Marker.LEN)
                    )
                case (_, offset):
                    return pc.Ok(
                        _with_idx_and_len()
                        .filter(_from_end_start(offset))
                        .drop(Marker.IDX, Marker.LEN)
                    )

        return _filter_lf(pc.Option(length), offset).unwrap()

    def tail(self, n: int = 5) -> Self:
        """Get the last n rows."""
        match n:
            case val if val < 0:
                msg = "`n` must be greater than or equal to 0"
                raise ValueError(msg)
            case 0:
                return self.limit(0)
            case _:
                return self.slice(-n)

    def drop(
        self, columns: TryIter[IntoExprColumn] = None, *more_columns: IntoExprColumn
    ) -> Self:
        """Drop columns from the frame."""
        expr = sql.all(exclude=try_chain(columns, more_columns)).into_duckdb()
        return self._new(self.inner().select(expr))

    def drop_nulls(self, subset: TryIter[str] = None) -> Self:
        """Drop rows that contain null values."""
        return (
            pc
            .Option(subset)
            .map(try_iter)
            .unwrap_or_else(self.columns.iter)
            .map(lambda name: sql.col(name).is_not_null())
            .into(self.filter)
        )

    def explode(
        self, columns: TryIter[IntoExprColumn], *more_columns: IntoExprColumn
    ) -> Self:
        """Explode list-like columns."""
        to_explode_names = (
            self.schema
            .into(ExprPlan, columns, more_columns, {})
            .projections.iter()
            .map(lambda r: r.name)
            .collect()
        )
        to_explode = to_explode_names.iter().map(sql.col).collect()
        target = (
            to_explode.first()
            if to_explode.length() == 1
            else (
                to_explode.first().list.zip(
                    *to_explode.iter().skip(1), sql.lit(1).eq(1)
                )
            )
        )

        zipped_index = (
            to_explode_names
            .iter()
            .enumerate()
            .map_star(lambda idx, name: (name, idx + 1))
            .collect(pc.Dict)
        )
        is_single_explode = to_explode.length() == 1

        def _project_col(name: str, *, unnest: bool, replace: SqlExpr) -> SqlExpr:
            match (unnest, name in to_explode_names):
                case (True, True):
                    match is_single_explode:
                        case True:
                            return replace.alias(name)
                        case False:
                            field = zipped_index.get_item(name).unwrap()
                            return replace.struct.extract(field).alias(name)
                case (False, True):
                    return sql.lit(None).alias(name)
                case _:
                    return sql.col(name)

        def _proj(*, unnest: bool) -> pc.Iter[SqlExpr]:
            replace = sql.unnest(target) if unnest else sql.lit(None)
            return self.columns.iter().map(
                lambda name: _project_col(name, unnest=unnest, replace=replace)
            )

        cond = target.is_not_null().and_(target.len().gt(0))

        return (
            self
            .filter(cond)
            .select(_proj(unnest=True))
            .union(self.filter(cond.not_()).select(_proj(unnest=False)))
        )

    def union(self, other: Self) -> Self:
        return self._new(self.inner().union(other.inner()))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns."""
        rename_map = pc.Dict(mapping)

        return self._iter_slct(
            lambda c: sql.col(c).alias(rename_map.get_item(c).unwrap_or(c))
        )

    def sql_query(self) -> ParsedQuery:
        """Generate a `ParsedQuery` object.

        Allow to format and display prettified `SQL`.

        Returns:
            ParsedQuery
        """
        from ._parser import ParsedQuery

        return ParsedQuery(self.inner().sql_query())

    def explain(self, kind: ExplainType | ExplainTypeLiteral = "standard") -> str:
        return self.inner().explain(kind)

    def unnest(
        self, columns: TryIter[IntoExprColumn], *more_columns: IntoExprColumn
    ) -> Self:
        return self._new(
            try_chain(columns, more_columns)
            .collect()
            .into(
                lambda unnest_cols: (
                    unnest_cols
                    .iter()
                    .map(sql.unnest)
                    .insert(sql.all(exclude=unnest_cols))
                    .map(lambda expr: expr.into_duckdb())
                )
            )
            .into(lambda exprs: self.inner().select(*exprs))
        )

    def first(self) -> Self:
        """Get the first row."""
        return self.head(1)

    def last(self) -> Self:
        """Get the last row."""
        return self.tail(1)

    def count(self) -> Self:
        """Return the count of each column."""
        return self._iter_agg(SqlExpr.count)

    def describe(self) -> Self:
        """Return descriptive statistics."""
        return self._new(self.inner().describe())

    def sum(self) -> Self:
        """Aggregate the sum of each column."""
        return self._iter_agg(SqlExpr.sum)

    def mean(self) -> Self:
        """Aggregate the mean of each column."""
        return self._iter_agg(SqlExpr.mean)

    def median(self) -> Self:
        """Aggregate the median of each column."""
        return self._iter_agg(SqlExpr.median)

    def min(self) -> Self:
        """Aggregate the minimum of each column."""
        return self._iter_agg(SqlExpr.min)

    def max(self) -> Self:
        """Aggregate the maximum of each column."""
        return self._iter_agg(SqlExpr.max)

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column."""
        fn = partial(SqlExpr.std, ddof=ddof)
        return self._iter_agg(fn)

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column."""
        fn = partial(SqlExpr.var, ddof=ddof)
        return self._iter_agg(fn)

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self._iter_agg(lambda c: c.is_null().count_if())

    def quantile(self, quantile: float) -> Self:
        """Compute quantile for each column."""
        return self._iter_agg(lambda c: c.quantile_cont(quantile))

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values."""
        return self._iter_slct(
            lambda c: sql.when(sql.col(c).is_nan()).then(value).otherwise(c).alias(c)
        )

    def fill_null(
        self,
        value: IntoExpr = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        """Fill null values."""
        return self._iter_slct(
            lambda c: (
                col(c)
                .fill_null(value=value, strategy=strategy, limit=limit)
                .inner()
                .alias(c)
            )
        )

    def shift(self, n: int = 1, *, fill_value: IntoExpr = None) -> Self:
        """Shift values by n positions."""
        return self._iter_slct(
            lambda c: sql.coalesce(sql.col(c).shift(n), fill_value).alias(c)
        )

    def clone(self) -> Self:
        """Create a copy of the LazyFrame."""
        return self._new(self.inner())

    def gather_every(self, n: int, offset: int = 0) -> Self:
        """Take every nth row starting from offset."""
        expr = Marker.TEMP.to_expr()
        return (
            self
            .with_row_index(name=Marker.TEMP, order_by=self.columns)
            .filter(expr.ge(offset).and_(expr.sub(offset).mod(n).eq(0)))
            .drop(Marker.TEMP)
        )

    @property
    def columns(self) -> pc.Vec[str]:
        """Get column names."""
        return pc.Vec.from_ref(self.inner().columns)

    @property
    def width(self) -> int:
        """Get number of columns."""
        return self.columns.length()

    @property
    def schema(self) -> Schema:
        match self._cached_schema:
            case pc.Some(schma):
                return schma
            case _:
                schma = Schema.from_frame(self.inner())
                self._cached_schema = pc.Some(schma)
                return schma

    def collect_schema(self) -> Schema:
        """Collect the schema (same as schema property for lazy)."""
        return self.schema

    def join(  # noqa: PLR0913
        self,
        other: Self,
        on: TryIter[str] = None,
        how: JoinStrategy = "inner",
        *,
        left_on: TryIter[str] = None,
        right_on: TryIter[str] = None,
        suffix: str = "_right",
    ) -> Self:
        """Join with another LazyFrame."""
        join_keys = JoinKeys.from_how(
            how, try_seq(on), try_seq(left_on), try_seq(right_on)
        ).unwrap()
        builder = JoinBuilder(suffix, self.columns, join_keys.right)

        def _cols_how() -> pc.Iter[str | Expression]:
            left = builder.left.iter()
            right = other.columns.iter()
            match how:
                case "inner" | "left":
                    return (
                        left
                        .map(builder.lhs)
                        .chain(right.filter_map(builder.for_inner_left))
                        .map(lambda c: c.into_duckdb())
                    )
                case "outer":
                    return (
                        left
                        .map(builder.lhs)
                        .chain(right.map(builder.for_outer))
                        .map(lambda c: c.into_duckdb())
                    )
                case "right":
                    return (
                        left
                        .filter(lambda name: name not in join_keys.left)
                        .map(builder.lhs)
                        .chain(right.map(builder.for_right))
                        .map(lambda c: c.into_duckdb())
                    )
                case "semi" | "anti":
                    return pc.Iter.once("lhs.*")

        return self._new(
            self
            .inner()
            .set_alias("lhs")
            .join(
                other.inner().set_alias("rhs"),
                condition=join_keys.left
                .iter()
                .zip(join_keys.right)
                .map_star(builder.equals)
                .reduce(SqlExpr.and_)
                .into_duckdb(),
                how=how,
            )
            .select(*_cols_how())
            .set_alias(self.inner().alias)
        )

    def join_cross(self, other: Self, *, suffix: str = "_right") -> Self:
        """Join with another LazyFrame."""
        builder = JoinBuilder(suffix, self.columns, other.columns)
        return self._new(
            self
            .inner()
            .set_alias("lhs")
            .cross(other.inner().set_alias("rhs"))
            .select(
                *builder.left
                .iter()
                .map(builder.lhs)
                .chain(builder.right.iter().map(builder.for_outer))
                .map(lambda c: c.into_duckdb())
            )
            .set_alias(self.inner().alias)
        )

    def join_asof(  # noqa: PLR0913
        self,
        other: Self,
        *,
        left_on: str | None = None,
        right_on: str | None = None,
        on: str | None = None,
        by_left: TryIter[str] = None,
        by_right: TryIter[str] = None,
        by: TryIter[str] = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
    ) -> Self:
        """Perform an asof join."""
        on_opt = pc.Option(on)
        on_keys = JoinKeys.from_on(
            on_opt, pc.Option(left_on), pc.Option(right_on)
        ).unwrap()
        by_keys = JoinKeys.from_by(
            try_seq(by), try_seq(by_left), try_seq(by_right)
        ).unwrap()
        drop_keys = pc.SetMut(by_keys.right)
        _ = on_opt.map(lambda _: drop_keys.add(on_keys.right))
        builder = JoinBuilder(suffix, self.columns, drop_keys)

        def _get_strategy(expr: SqlExpr) -> SqlExpr:
            other = builder.rhs(on_keys.right)
            match strategy:
                case "backward":
                    return expr.ge(other)
                case "forward":
                    return expr.le(other)

        by_cond = (
            by_keys.left
            .iter()
            .zip(by_keys.right)
            .map_star(builder.equals)
            .chain(builder.lhs(on_keys.left).pipe(_get_strategy).pipe(pc.Iter.once))
            .reduce(SqlExpr.and_)
        )
        selected = (
            builder.left
            .iter()
            .map(builder.lhs)
            .chain(other.columns.iter().filter_map(builder.for_inner_left))
            .map(lambda c: c.inner())
        )
        qry = (
            exp
            .select(*selected)
            .from_("lhs")
            .join("rhs", on=by_cond.inner(), join_type="asof left")
        )

        return self._from_sql_expr(qry, lhs=self.inner(), rhs=other)

    def unique(
        self,
        subset: TryIter[str] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        order_by: TrySeq[str] = None,
    ) -> Self:
        """Drop duplicate rows from this LazyFrame."""

        def _marker(
            subset_cols: Iterable[IntoExprColumn],
        ) -> pc.Result[SqlExpr, ValueError]:
            match (
                keep,
                pc.Option(order_by).map(lambda value: try_iter(value).collect()),
            ):
                case ("none", _):
                    return pc.Ok(
                        sql.all().count().over(partition_by=pc.Some(subset_cols))
                    )
                case ("first", pc.Some(order_by_cols)):
                    return pc.Ok(
                        sql.row_number().over(
                            partition_by=pc.Some(subset_cols),
                            order_by=pc.Some(order_by_cols),
                        )
                    )
                case ("last", pc.Some(order_by_cols)):
                    return pc.Ok(
                        sql.row_number().over(
                            partition_by=pc.Some(subset_cols),
                            order_by=pc.Some(order_by_cols),
                            descending=True,
                            nulls_last=True,
                        )
                    )
                case ("first" | "last", pc.NONE):
                    msg = """`order_by` must be specified when `keep` is 'first' or 'last'
                    because LazyFrame makes no assumptions about row order."""

                    return pc.Err(ValueError(msg))
                case _:
                    return pc.Ok(
                        sql.row_number().over(partition_by=pc.Some(subset_cols))
                    )

        return (
            pc
            .Option(subset)
            .map(try_iter)
            .unwrap_or_else(self.columns.iter)
            .into(_marker)
            .map(
                lambda expr: (
                    expr
                    .alias(Marker.TEMP)
                    .pipe(self.with_columns)
                    .filter(Marker.TEMP.to_expr().eq(1))
                    .drop(Marker.TEMP)
                )
            )
            .unwrap()
        )

    def pivot(  # noqa: C901, PLR0913
        self,
        on: TryIter[str],
        on_columns: Sequence[PythonLiteral],
        index: TryIter[str] = None,
        values: TryIter[str] = None,
        aggregate_function: PivotAgg = "first",
        *,
        maintain_order: bool = False,
        separator: str = "_",
    ) -> Self:
        """Create a spreadsheet-style pivot table."""

        def _cols_not_in(cols: Iterable[str]) -> pc.Seq[str]:
            return (
                self.columns
                .iter()
                .filter(lambda c: c not in on_cols and c not in cols)
                .collect()
            )

        def _get_idx_and_vals() -> pc.Result[
            tuple[pc.Seq[str], pc.Seq[str]], ValueError
        ]:
            match (try_seq(index), try_seq(values)):
                case (pc.Some(idx), pc.Some(vals)):
                    return pc.Ok((idx, vals))
                case (pc.Some(idx), _):
                    return pc.Ok((idx, _cols_not_in(idx)))
                case (_, pc.Some(vals)):
                    return pc.Ok((_cols_not_in(vals), vals))
                case _:
                    msg = "`pivot` needs either `index` or `values` to be specified"
                    return pc.Err(ValueError(msg))

        on_cols = try_iter(on).collect(dict.fromkeys)
        idx_cols, val_cols = _get_idx_and_vals().unwrap()

        multi = val_cols.length() > 1
        agg = PIVOT_AGG[aggregate_function]

        def _aliased(col: str) -> SqlExpr:
            expr = sql.col(col).pipe(agg)
            return expr.alias(col) if multi else expr

        def _pivoted() -> Self:

            def _on_exprs(
                on_iter: pc.Seq[str],
            ) -> PyoIterable[exp.Expr] | PyoIterable[str]:
                converted = pc.Iter(on_columns).map(exp.convert).collect()
                expr = exp.In(this=on_iter.first(), expressions=converted)
                return pc.Iter.once(expr)

            def _group() -> exp.Group | None:
                group = idx_cols.then(
                    lambda cols: exp.Group(expressions=cols.into(list))
                )
                return group.unwrap() if group.is_some() else None

            def _pivot() -> exp.Expr:
                return exp.Pivot(
                    this=exp.to_table("rel"),  # pyright: ignore[reportUnknownMemberType]
                    expressions=try_iter(on).collect().into(_on_exprs),
                    using=val_cols.iter().map(_aliased).map(lambda c: c.inner()),
                    group=_group(),
                )

            def _select_ordered(cols: Iterable[str]) -> exp.Expr:
                return (
                    exp.select("*").from_(exp.Subquery(this=_pivot())).order_by(*cols)
                )

            qry = (
                try_iter(idx_cols if maintain_order else None)
                .collect()
                .then(_select_ordered)
                .unwrap_or_else(_pivot)
            )

            return self._from_sql_expr(qry, rel=self.inner())

        def _handle_multi(lf: Self) -> Self:
            match multi:
                case True:

                    def _rename_col(val_col: str) -> pc.Iter[SqlExpr]:
                        def _swap(on_val: str) -> SqlExpr:
                            in_ = f"{on_val}_{val_col}"
                            out = f"{val_col}{separator}{on_val}"
                            return sql.col(in_).alias(out)

                        return on_values.iter().map(_swap)

                    on_values = pc.Iter(on_columns).map(str).collect()
                    cols = (
                        idx_cols
                        .iter()
                        .map(sql.col)
                        .chain(val_cols.iter().flat_map(_rename_col))
                    )
                    return lf.select(*cols)
                case False:
                    return lf

        return _pivoted().pipe(_handle_multi)

    def unpivot(
        self,
        on: TryIter[str] = None,
        index: TryIter[str] = None,
        variable_name: str = "variable",
        value_name: str = "value",
        order_by: TryIter[str] = None,
    ) -> Self:
        """Unpivot from wide to long format."""
        index_cols = try_iter(index).collect(dict.fromkeys)
        unpivot_cols = (
            try_iter(on)
            .then_some()
            .unwrap_or_else(
                lambda: self.columns.iter().filter(lambda name: name not in index_cols)
            )
        )

        def _unpivot() -> exp.Pivot:
            return exp.Pivot(
                this=exp.to_table("rel"),  # pyright: ignore[reportUnknownMemberType]
                expressions=unpivot_cols,
                unpivot=True,
                into=exp.UnpivotColumns(this=variable_name, expressions=(value_name,)),
            )

        def _select() -> exp.Select:
            sub_qry = exp.Subquery(this=_unpivot())
            return exp.select(*index_cols, variable_name, value_name).from_(sub_qry)

        qry = (
            try_iter(order_by)
            .then(lambda cols: _select().order_by(*cols))
            .unwrap_or_else(_select)
        )

        return self._from_sql_expr(qry, rel=self.inner())

    def with_row_index(self, name: str, *, order_by: TrySeq[str]) -> Self:
        """Insert row index based on order_by."""
        row_nb = (
            sql
            .row_number()
            .over(order_by=pc.Some(order_by))
            .sub(1)
            .alias(name)
            .into_duckdb()
        )
        return self._new(self.inner().select(row_nb, sql.all().into_duckdb()))

    def top_k(
        self, k: int, by: TryIter[IntoExpr], *, reverse: TrySeq[bool] = False
    ) -> Self:
        """Return top k rows by column(s)."""

        def _descending() -> TrySeq[bool]:
            match reverse:
                case bool():
                    return not reverse
                case _:
                    return try_iter(reverse).map(lambda x: not x).collect()

        return self.sort(by, descending=_descending()).head(k)

    def bottom_k(
        self, k: int, by: TryIter[IntoExpr], *, reverse: TrySeq[bool] = False
    ) -> Self:
        """Return bottom k rows by column(s)."""
        return self.sort(by, descending=reverse).head(k)

    def cast(self, dtypes: Mapping[str, DataType] | DataType) -> Self:
        """Cast columns to specified dtypes."""
        match dtypes:
            case Mapping():
                dtype_map = pc.Dict(dtypes)
                return self._iter_slct(
                    lambda c: (
                        dtype_map
                        .get_item(c)
                        .map(lambda dtype: sql.col(c).cast(dtype.raw).alias(c))
                        .unwrap_or_else(lambda: sql.col(c))
                    )
                )
            case _:
                return self._iter_slct(lambda c: sql.col(c).cast(dtypes.raw).alias(c))

    def sink_parquet(
        self, path: str | Path, *, compression: ParquetCompression = "zstd"
    ) -> None:
        """Write to Parquet file."""
        self.inner().write_parquet(str(path), compression=compression)

    def sink_csv(
        self, path: str | Path, *, separator: str = ",", include_header: bool = True
    ) -> None:
        """Write to CSV file."""
        self.inner().write_csv(str(path), sep=separator, header=include_header)

    def sink_ndjson(self, path: str | Path) -> None:
        """Write to newline-delimited JSON file."""
        self.inner().pl(lazy=True).sink_ndjson(path)

    def reverse(self) -> Self:
        """Reverse the order of rows."""
        return (
            self
            .with_row_index(name=Marker.TEMP, order_by=self.columns)
            .sort(Marker.TEMP, descending=True)
            .drop(Marker.TEMP)
        )

    def drop_nans(self, subset: TryIter[str] = None) -> Self:
        """Drop rows that contain NaN values."""
        return (
            pc
            .Option(subset)
            .map(try_iter)
            .unwrap_or_else(self.columns.iter)
            .map(lambda name: sql.col(name).is_nan().not_())
            .into(self.filter)
        )

    def fetch_all(self) -> pc.Vec[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        return pc.Vec.from_ref(self.inner().fetchall())

    def show(
        self,
        max_width: SupportsInt | None = None,
        max_rows: SupportsInt | None = None,
        max_col_width: SupportsInt | None = None,
        null_value: str | None = None,
        render_mode: RenderModeLiteral | None = None,
    ) -> None:
        return self.inner().show(
            max_width=max_width,
            max_rows=max_rows,
            max_col_width=max_col_width,
            null_value=null_value,
            render_mode=render_mode,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.inner().shape
