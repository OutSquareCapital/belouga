"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

import operator as op
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, SupportsInt

import pyochain as pc
from duckdb import DuckDBPyRelation, Expression
from sqlglot import exp

from . import sql
from ._datatypes import DataType, Int64
from ._funcs import col
from ._joins import JoinBuilder, JoinKeys
from ._meta import ExprPlan, Marker
from ._schema import Schema
from .sql import SqlExpr
from .sql.utils import TryIter, TrySeq, check_by_arg, try_iter, try_seq

if TYPE_CHECKING:
    import polars as pl
    from _duckdb import ExplainType
    from _duckdb._enums import (  # pyright: ignore[reportMissingModuleSource]
        ExplainTypeLiteral,
        RenderModeLiteral,
    )
    from pyochain.traits import PyoIterable

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
    _schema: Schema

    def __init__(
        self, data: IntoRel, orient: Orientation = "col", schema: Schema | None = None
    ) -> None:
        rel = sql.into_relation(data, orient)
        self._inner = rel
        self._schema = schema or Schema.from_frame(rel)

    def _from_sql_expr(self, expr: exp.Expr, **kwargs: IntoRel) -> Self:
        qry = sql.from_query(expr.sql(dialect="duckdb"), **kwargs)
        return self.__class__(qry)

    def _iter_slct(self, func: Callable[[str], SqlExpr]) -> Self:
        return self.select(self.columns.iter().map(func))

    def _iter_agg(self, func: Callable[[SqlExpr], SqlExpr]) -> Self:
        return self.__class__(
            self.columns
            .iter()
            .map(lambda c: func(sql.col(c)).alias(c).into_duckdb())
            .into(lambda exprs: self.inner().aggregate(exprs))
        )

    def lazy(self) -> pl.LazyFrame:
        """Get a Polars LazyFrame.

        Returns:
            pl.LazyFrame: A Polars LazyFrame representing the same query.
        """
        return self.inner().pl(lazy=True).pipe(Marker.drop_marker, self.columns)

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame.

        Returns:
            pl.DataFrame: A Polars DataFrame representing the query result.
        """
        return self.inner().pl().pipe(Marker.drop_marker, self.columns)

    def select(
        self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr, **named_exprs: IntoExpr
    ) -> Self:
        """Context method to select columns or expressions.

        Args:
        exprs (TryIter[IntoExpr]): Expressions to select.
        *more_exprs (IntoExpr): Additional expressions to select.
        **named_exprs (IntoExpr): Expressions to select with aliases.

        Returns:
            Self: A new LazyFrame with the selected columns.
        """
        return self.__class__(
            self.schema.into(ExprPlan, exprs, more_exprs, named_exprs).select_context(
                self.inner()
            )
        )

    def with_columns(
        self, exprs: TryIter[IntoExpr], *more_exprs: IntoExpr, **named_exprs: IntoExpr
    ) -> Self:
        """Add or replace columns.

        Args:
            exprs (TryIter[IntoExpr]): Expressions to add or replace.
            *more_exprs (IntoExpr): Additional expressions to add or replace.
            **named_exprs (IntoExpr): Expressions to add or replace with aliases.

        Returns:
            Self: A new LazyFrame with the added or replaced columns.
        """
        return self.__class__(
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
        """Filter rows based on predicates and equality constraints.

        Args:
            predicates (TryIter[IntoExprColumn]): Predicates to filter rows.
            *more_predicates (IntoExprColumn): Additional predicates to filter rows.
            **constraints (IntoExpr): Equality constraints to filter rows.

        Returns:
            Self: A new LazyFrame with the filtered rows.
        """

        def _constraint(k: str, val: IntoExpr) -> SqlExpr:
            return sql.col(k).eq(sql.into_expr(val))

        expr = (
            try_iter(predicates)
            .chain(more_predicates)
            .map(lambda value: sql.into_expr(value, as_col=True))
            .chain(pc.Iter(constraints.items()).map_star(_constraint))
            .into(sql.reduce, SqlExpr.and_)
            .into_duckdb()
        )
        return self.__class__(self.inner().filter(expr), schema=self.schema)

    def group_by(
        self,
        keys: TryIter[IntoExpr] = None,
        *more_keys: IntoExpr,
        drop_null_keys: bool = False,
        strategy: GroupByClause | None = None,
    ) -> LazyGroupBy:
        """Start a group by operation.

        Args:
            keys (TryIter[IntoExpr]): Keys to group by.
            *more_keys (IntoExpr): Additional keys to group by.
            drop_null_keys (bool): Whether to drop rows with null keys.
            strategy (GroupByClause | None): Grouping strategy.

        Returns:
            LazyGroupBy: A LazyGroupBy object representing the group by operation.
        """
        from ._groupby import LazyGroupBy

        key_exprs = (
            try_iter(keys)
            .chain(more_keys)
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
        """Aggregate with GROUP BY ALL — DuckDB auto-detects grouping keys.

        Args:
            exprs (TryIter[IntoExpr]): Expressions to aggregate.
            *more_exprs (IntoExpr): Additional expressions to aggregate.
            **named_exprs (IntoExpr): Expressions to aggregate with aliases.

        Returns:
            Self: A new LazyFrame with the aggregated rows.
        """
        return self.__class__(
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
        """Sort by columns.

        Args:
            by (TryIter[IntoExpr]): Columns to sort by.
            *more_by (IntoExpr): Additional columns to sort by.
            descending (TrySeq[bool]): Whether to sort in descending order.
            nulls_last (TrySeq[bool]): Whether to place nulls last.

        Returns:
            Self: A new LazyFrame with the sorted rows.
        """
        lf = (
            try_iter(by)
            .chain(more_by)
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
        return self.__class__(lf, schema=self.schema)

    def limit(self, n: int) -> Self:
        """Limit the number of rows.

        Args:
            n (int): The number of rows to limit.

        Returns:
            Self: A new LazyFrame with the limited rows.
        """
        return self.__class__(self.inner().limit(n), schema=self.schema)

    def head(self, n: int = 5) -> Self:
        """Get the first n rows.

        Args:
            n (int): The number of rows to retrieve.

        Returns:
            Self: A new LazyFrame with the first n rows.
        """
        return self.limit(n)

    def slice(self, offset: int, length: int | None = None) -> Self:
        """Get a slice of rows.

        Args:
            offset (int): The starting index of the slice.
            length (int | None): The number of rows to include in the slice.

        Returns:
            Self: A new LazyFrame with the sliced rows.
        """

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
                    return pc.Ok(self.__class__(rel, schema=self.schema))
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
        """Get the last n rows.

        Args:
            n (int): The number of rows to retrieve.

        Returns:
            Self: A new LazyFrame with the last n rows.

        Raises:
            ValueError: If n is negative.
        """
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
        """Drop columns from the frame.

        Args:
            columns (TryIter[IntoExprColumn]): Columns to drop.
            *more_columns (IntoExprColumn): Additional columns to drop.

        Returns:
            Self: A new LazyFrame with the specified columns dropped.
        """
        to_drop = (
            try_iter(columns)
            .chain(more_columns)
            .map(lambda v: sql.into_expr(v, as_col=True))
            .filter_map(SqlExpr.root_column_name)
            .collect()
        )
        to_drop.iter().for_each(self._schema.__delitem__)
        expr = sql.all(exclude=to_drop).into_duckdb()
        return self.__class__(self.inner().select(expr), schema=self.schema)

    def drop_nulls(self, subset: TryIter[str] = None) -> Self:
        """Drop rows that contain null values.

        Args:
            subset (TryIter[str]): Columns to consider for null values.

        Returns:
            Self: A new LazyFrame with rows containing null values dropped.
        """
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
        """Explode list-like columns.

        Args:
            columns (TryIter[IntoExprColumn]): Columns to explode.
            *more_columns (IntoExprColumn): Additional columns to explode.

        Returns:
            Self: A new LazyFrame with the exploded columns.
        """
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
        return self.__class__(self.inner().union(other.inner()), schema=self.schema)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns.

        Args:
            mapping (Mapping[str, str]): A dictionary mapping old column names to new column names.

        Returns:
            Self: A new LazyFrame with the renamed columns.
        """
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
        return self.__class__(
            try_iter(columns)
            .chain(more_columns)
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
        """Get the first row.

        Returns:
            Self: A new LazyFrame with the first row.
        """
        return self.head(1)

    def last(self) -> Self:
        """Get the last row.

        Returns:
            Self: A new LazyFrame with the last row.
        """
        return self.tail(1)

    def count(self) -> Self:
        """Return the count of each column."""
        return self._iter_agg(SqlExpr.count)

    def describe(self) -> Self:
        """Return descriptive statistics."""
        return self.__class__(self.inner().describe())

    def sum(self) -> Self:
        """Aggregate the sum of each column.

        Returns:
            Self: A new LazyFrame with the sum of each column.
        """
        return self._iter_agg(SqlExpr.sum)

    def mean(self) -> Self:
        """Aggregate the mean of each column.

        Returns:
            Self: A new LazyFrame with the mean of each column.
        """
        return self._iter_agg(SqlExpr.mean)

    def median(self) -> Self:
        """Aggregate the median of each column.

        Returns:
            Self: A new LazyFrame with the median of each column.
        """
        return self._iter_agg(SqlExpr.median)

    def min(self) -> Self:
        """Aggregate the minimum of each column.

        Returns:
            Self: A new LazyFrame with the minimum of each column.
        """
        return self._iter_agg(SqlExpr.min)

    def max(self) -> Self:
        """Aggregate the maximum of each column.

        Returns:
            Self: A new LazyFrame with the maximum of each column.
        """
        return self._iter_agg(SqlExpr.max)

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column.

        Args:
            ddof (int): Delta Degrees of Freedom.

        Returns:
            Self: A new LazyFrame with the standard deviation of each column.
        """
        fn = partial(SqlExpr.std, ddof=ddof)
        return self._iter_agg(fn)

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column.

        Args:
            ddof (int): Delta Degrees of Freedom.

        Returns:
            Self: A new LazyFrame with the variance of each column.
        """
        fn = partial(SqlExpr.var, ddof=ddof)
        return self._iter_agg(fn)

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self._iter_agg(lambda c: c.is_null().count_if())

    def quantile(self, quantile: float) -> Self:
        """Compute quantile for each column.

        Args:
            quantile (float): The quantile to compute.

        Returns:
            Self: A new LazyFrame with the computed quantile for each column.
        """
        return self._iter_agg(lambda c: c.quantile_cont(quantile))

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values.

        Args:
            value (float | Expr | None): The value to replace NaNs with.

        Returns:
            Self: A new LazyFrame with NaNs filled.
        """
        return self._iter_slct(
            lambda c: sql.when(sql.col(c).is_nan()).then(value).otherwise(c).alias(c)
        )

    def fill_null(
        self,
        value: IntoExpr = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        """Fill null values.

        Args:
            value (IntoExpr | None): The value to replace nulls with.
            strategy (FillNullStrategy | None): The strategy to use for filling nulls.
            limit (int | None): The maximum number of nulls to fill.

        Returns:
            Self: A new LazyFrame with nulls filled.
        """
        return self._iter_slct(
            lambda c: (
                col(c)
                .fill_null(value=value, strategy=strategy, limit=limit)
                .inner()
                .alias(c)
            )
        )

    def shift(self, n: int = 1, *, fill_value: IntoExpr = None) -> Self:
        """Shift values by n positions.

        Args:
            n (int): Number of positions to shift.
            fill_value (IntoExpr | None): The value to use for filling empty positions.

        Returns:
            Self: A new LazyFrame with shifted values.
        """
        return self._iter_slct(
            lambda c: sql.coalesce(sql.col(c).shift(n), fill_value).alias(c)
        )

    def clone(self) -> Self:
        """Create a copy of the LazyFrame.

        Returns:
            Self: A new LazyFrame that is a copy of the current one.
        """
        return self.__class__(self.inner(), schema=self.schema.copy())

    def gather_every(self, n: int, offset: int = 0) -> Self:
        """Take every nth row starting from offset.

        Args:
            n (int): The step size.
            offset (int): The starting offset.

        Returns:
            Self: A new LazyFrame with every nth row starting from offset.
        """
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
        return self.schema.keys()

    @property
    def width(self) -> int:
        """Get number of columns."""
        return self.columns.length()

    @property
    def schema(self) -> Schema:
        return self._schema

    def collect_schema(self) -> Schema:
        """Collect the schema (same as schema property for lazy).

        Returns:
            Schema: The schema of the LazyFrame.
        """
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
        """Join with another LazyFrame.

        Args:
            other (Self): The other LazyFrame to join with.
            on (TryIter[str] | None): The columns to join on.
            how (JoinStrategy): The type of join to perform.
            left_on (TryIter[str] | None): The columns from the left frame to join on.
            right_on (TryIter[str] | None): The columns from the right frame to join on.
            suffix (str): The suffix to use for overlapping column names.

        Returns:
            Self: A new LazyFrame resulting from the join.
        """
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

        return self.__class__(
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
        """Join with another LazyFrame using a cross join.

        Args:
            other (Self): The other LazyFrame to join with.
            suffix (str): The suffix to use for overlapping column names.

        Returns:
            Self: A new LazyFrame resulting from the cross join.
        """
        builder = JoinBuilder(suffix, self.columns, other.columns)
        return self.__class__(
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
        """Perform an asof join.

        Args:
            other (Self): The other LazyFrame to join with.
            left_on (str | None): The column from the left frame to join on.
            right_on (str | None): The column from the right frame to join on.
            on (str | None): The column to join on.
            by_left (TryIter[str] | None): The columns from the left frame to group by.
            by_right (TryIter[str] | None): The columns from the right frame to group by.
            by (TryIter[str] | None): The columns to group by.
            strategy (AsofJoinStrategy): The strategy to use for the asof join.
            suffix (str): The suffix to use for overlapping column names.

        Returns:
            Self: A new LazyFrame resulting from the asof join.
        """
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
        """Drop duplicate rows from this LazyFrame.

        Args:
            subset (TryIter[str] | None): Subset of columns to consider for identifying duplicates.
            keep (UniqueKeepStrategy): Strategy to determine which duplicates to keep.
            order_by (TrySeq[str] | None): Columns to order by when determining which duplicates to keep.

        Returns:
            Self: A new LazyFrame with duplicates removed.
        """

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
        """Create a spreadsheet-style pivot table.

        Args:
            on (TryIter[str]): Columns to pivot on.
            on_columns (Sequence[PythonLiteral]): Values to pivot on.
            index (TryIter[str] | None): Columns to use as the index.
            values (TryIter[str] | None): Columns to use as values.
            aggregate_function (PivotAgg): Aggregation function to apply.
            maintain_order (bool): Whether to maintain the order of the index columns.
            separator (str): Separator to use for multi-level column names.

        Returns:
            Self: A new LazyFrame with the pivoted data.
        """

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

        on_cols = try_iter(on).collect()
        idx_cols, val_cols = _get_idx_and_vals().unwrap()

        multi = val_cols.length() > 1
        agg = PIVOT_AGG[aggregate_function]

        def _aliased(col: str) -> SqlExpr:
            expr = sql.col(col).pipe(agg)
            return expr.alias(col) if multi else expr

        def _pivoted() -> Self:

            def _on_exprs() -> PyoIterable[exp.Expr] | PyoIterable[str]:
                converted = pc.Iter(on_columns).map(exp.convert).collect()
                expr = exp.In(this=on_cols.first(), expressions=converted)
                return pc.Iter.once(expr)

            def _group() -> exp.Group | None:
                group = idx_cols.then(
                    lambda cols: exp.Group(expressions=cols.into(list))
                )
                return group.unwrap() if group.is_some() else None

            def _pivot() -> exp.Expr:
                return exp.Pivot(
                    this=exp.to_table("rel"),
                    expressions=_on_exprs(),
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
                    on_values = pc.Iter(on_columns).map(str).collect()

                    def _rename_col(val_col: str) -> pc.Iter[SqlExpr]:
                        def _swap(on_val: str) -> SqlExpr:
                            in_ = f"{on_val}_{val_col}"
                            out = f"{val_col}{separator}{on_val}"
                            return sql.col(in_).alias(out)

                        return on_values.iter().map(_swap)

                    return (
                        idx_cols
                        .iter()
                        .map(sql.col)
                        .chain(val_cols.iter().flat_map(_rename_col))
                        .into(lf.select)
                    )
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
        """Unpivot from wide to long format.

        Args:
            on (TryIter[str] | None): Columns to unpivot.
            index (TryIter[str] | None): Columns to use as the index.
            variable_name (str): Name of the variable column.
            value_name (str): Name of the value column.
            order_by (TryIter[str] | None): Columns to order by.

        Returns:
            Self: A new LazyFrame with the unpivoted data.
        """
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
                this=exp.to_table("rel"),
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

    def with_row_index(self, name: str, *, order_by: TryIter[str]) -> Self:
        """Insert row index based on order_by.

        Args:
            name (str): The name of the new index column.
            order_by (TrySeq[str]): Columns to order by for row numbering.

        Returns:
            Self: A new LazyFrame with the row index added.
        """
        row_nb = (
            sql
            .row_number()
            .over(order_by=pc.Some(order_by))
            .sub(1)
            .alias(name)
            .into_duckdb()
        )
        self._schema.insert_at(0, name, Int64())
        lf = self.inner().select(row_nb, sql.all().into_duckdb())
        return self.__class__(lf, schema=self.schema)

    def top_k(
        self, k: int, by: TryIter[IntoExpr], *, reverse: TrySeq[bool] = False
    ) -> Self:
        """Return top k rows by column(s)."""

        def _descending() -> TrySeq[bool]:
            match reverse:
                case bool():
                    return not reverse
                case _:
                    return try_iter(reverse).map(op.not_).collect()

        return self.sort(by, descending=_descending()).head(k)

    def bottom_k(
        self, k: int, by: TryIter[IntoExpr], *, reverse: TrySeq[bool] = False
    ) -> Self:
        """Return bottom k rows by column(s)."""
        return self.sort(by, descending=reverse).head(k)

    def cast(self, dtypes: Mapping[str, DataType] | DataType) -> Self:
        """Cast columns to specified dtypes.

        Args:
            dtypes (Mapping[str, DataType] | DataType): The target data types for the columns.

        Returns:
            Self: A new LazyFrame with the columns cast to the specified dtypes.
        """
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
        """Reverse the order of rows.

        Returns:
            Self: A new LazyFrame with the rows reversed.
        """
        return (
            self
            .with_row_index(name=Marker.TEMP, order_by=self.columns)
            .sort(Marker.TEMP, descending=True)
            .drop(Marker.TEMP)
        )

    def drop_nans(self, subset: TryIter[str] = None) -> Self:
        """Drop rows that contain NaN values.

        Args:
            subset (TryIter[str] | None): Columns to consider for NaN values. If None, all columns are considered.

        Returns:
            Self: A new LazyFrame with rows containing NaN values dropped.
        """
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
