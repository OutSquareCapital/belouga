"""LazyFrame providing Polars-like API over DuckDB relations."""

from __future__ import annotations

import operator as op
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, SupportsInt, overload, override

from pyochain import (
    Dict,
    Err,
    Iter,
    Null,
    Ok,
    Option,
    Result,
    Seq,
    Set,
    SetMut,
    Some,
    Vec,
)
from sqlglot import exp

from . import datatypes as dt
from ._core import CoreHandler, into_expr
from ._expr import Expr
from ._funcs import all, col, lit, row_number, unnest
from ._plan import (
    ExprPlan,
    JoinBuilder,
    JoinKeys,
    Marker,
    Tables,
    pivot,
    resolve_unnest,
    unpivot,
)
from ._scans import ScanSource
from .utils import TryIter, TrySeq, select, try_iter, try_seq

if TYPE_CHECKING:
    import polars as pl
    from _duckdb import ExplainType
    from _duckdb._enums import (  # pyright: ignore[reportMissingModuleSource]
        ExplainTypeLiteral,
        RenderModeLiteral,
    )
    from _duckdb._typing import (  # pyright: ignore[reportMissingModuleSource]
        CsvCompression,
        ParquetFieldsOptions,
    )
    from duckdb import DuckDBPyRelation
    from pyochain.traits import PyoKeysView

    from ._groupby import LazyGroupBy
    from ._parser import ParsedQuery
    from .typing import (
        AsofJoinStrategy,
        FillNullStrategy,
        GroupByClause,
        IntoExpr,
        IntoExprColumn,
        IntoRel,
        JoinStrategy,
        Orientation,
        ParquetCompression,
        PivotAgg,
        PythonLiteral,
        Schema,
        UniqueKeepStrategy,
    )

MAX_I64 = 9_223_372_036_854_775_807


@dataclass(slots=True, init=False, repr=False)
class LazyFrame(CoreHandler[exp.Selectable]):
    """LazyFrame providing Polars-like API over DuckDB relations."""

    _inner: exp.Selectable
    _sources: Dict[str, ScanSource]
    _schema: Schema

    def __init__(self, data: IntoRel | Self, orient: Orientation = "col") -> None:
        match data:
            case LazyFrame():
                self._inner = data._inner
                self._sources = data._sources
                self._schema = data._schema
            case _:
                source = ScanSource.build(data, orient).set_alias()
                self._inner = exp.select(exp.Star()).from_(
                    exp.to_table(source.identity)
                )
                self._sources = Dict.from_ref({source.identity: source})
                self._schema = source.schema

    def _make(
        self, ast: exp.Selectable, sources: Dict[str, ScanSource], schema: Schema
    ) -> Self:
        out = self.__class__.__new__(self.__class__)
        out._inner = ast
        out._sources = sources
        out._schema = schema
        return out

    def _from_ast(self, ast: exp.Selectable, schema: Schema, **subs: Self) -> Self:
        def _replacer(
            node: exp.Selectable, subs: Mapping[str, LazyFrame]
        ) -> exp.Selectable:
            match node:
                case exp.Table() if node.name in subs:
                    pivots = node.args.get("pivots")
                    alias = exp.TableAlias(this=exp.to_identifier(node.alias_or_name))
                    this = subs[node.name]._inner
                    return exp.Subquery(this=this, alias=alias, pivots=pivots)
                case _:
                    return node

        new_sources = (
            Iter(subs.items())
            .map_star(lambda _name, v: v._sources.items().iter())
            .flatten()
            .chain(self._sources.items())
            .collect(Dict)
        )

        return ast.transform(_replacer, subs=subs, copy=False).pipe(
            self._make, new_sources, schema
        )

    @override
    def _cls(self, value: exp.Selectable) -> Self:
        return self._from_ast(value, src=self, schema=self._schema)

    def _materialize(self) -> DuckDBPyRelation:
        return self._inner.pipe(ScanSource.from_query, **self._sources).relation

    def _iter_slct(self, func: Callable[[Expr], Expr]) -> Self:
        return (
            self.columns
            .iter()
            .map(lambda c: col(c).pipe(func).alias(c).inner)
            .into(self.select)
        )

    @overload
    def _into_pl(self, *, lazy: Literal[True]) -> pl.LazyFrame: ...
    @overload
    def _into_pl(self, *, lazy: Literal[False]) -> pl.DataFrame: ...
    def _into_pl(self, *, lazy: bool) -> pl.LazyFrame | pl.DataFrame:
        df = self._materialize().pl(lazy=lazy)
        if Marker.TEMP in self.columns:
            return df.drop(Marker.TEMP)
        return df

    def lazy(self) -> pl.LazyFrame:
        """Get a Polars LazyFrame.

        Warning:
            There's currently a known bug for the interaction between `polars.LazyFrame` and `DuckDB`.

            `belugas.LazyFrame.lazy()` produces a Polars `LazyFrame` backed by a **`PYTHON SCAN`** (via `duckdb/polars_io.py`).
            Certain Polars operations that internally generate a `dynamic_predicate` optimization node cause a **panic** when collected.

            **Affected operations:** `.sort().limit()`, `.sort().head()`, `.top_k()`, `.bottom_k()`

            **Workaround:** `.collect().lazy()` works — it materializes to an in-memory `DataFrame` first, so the plan uses a native `DF [...]` scan instead of `PYTHON SCAN`.

            ### Mechanism

                1. Polars optimizes `sort + limit` into a single node with a `dynamic_predicate` — an internal filter that pre-screens rows before the full sort.
                2. This predicate gets pushed down to the DuckDB IO source plugin as the `predicate` callback argument.
                3. `_predicate_to_expression` in `polars_io.py` fails to convert the `dynamic_predicate` node to a DuckDB expression (correctly suppressed via `contextlib.suppress`).
                4. The fallback path (`polars_io.py:307`) calls `pl.from_arrow(batch).filter(predicate)`, which internally does `.lazy().filter(predicate).collect()`.
                5. The `dynamic_predicate` expression is an optimizer-internal node — Polars' own `expr_to_ir` converter doesn't handle it → **panic** at `expr_to_ir.rs:627`.

        Returns:
            pl.LazyFrame: A Polars LazyFrame representing the same query.
        """
        return self._into_pl(lazy=True)

    def collect(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame.

        Returns:
            pl.DataFrame: A Polars DataFrame representing the query result.
        """
        return self._into_pl(lazy=False)

    def select(
        self,
        exprs: TryIter[IntoExpr] = None,
        *more_exprs: IntoExpr,
        **named_exprs: IntoExpr,
    ) -> Self:
        """Context method to select columns or expressions.

        Args:
        exprs (TryIter[IntoExpr]): Expressions to select.
        *more_exprs (IntoExpr): Additional expressions to select.
        **named_exprs (IntoExpr): Expressions to select with aliases.

        Returns:
            Self: A new LazyFrame with the selected columns.
        """
        plan = self._schema.into(ExprPlan, exprs, more_exprs, named_exprs)
        return plan.select_ctx().map_or_else(
            lambda: self.__class__(ScanSource.from_none().relation),
            lambda ast: self._from_ast(ast, plan.select_schema(self._schema), src=self),
        )

    def with_columns(
        self,
        exprs: TryIter[IntoExpr] = None,
        *more_exprs: IntoExpr,
        **named_exprs: IntoExpr,
    ) -> Self:
        """Add or replace columns.

        Args:
            exprs (TryIter[IntoExpr]): Expressions to add or replace.
            *more_exprs (IntoExpr): Additional expressions to add or replace.
            **named_exprs (IntoExpr): Expressions to add or replace with aliases.

        Returns:
            Self: A new LazyFrame with the added or replaced columns.
        """
        plan = self._schema.into(ExprPlan, exprs, more_exprs, named_exprs)
        ast = plan.with_columns_ctx()
        return self._from_ast(ast, plan.with_columns_schema(self._schema), src=self)

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

        def _constraint(k: str, val: IntoExpr) -> Expr:
            return col(k).eq(into_expr(val, as_col=False))

        condition = (
            try_iter(predicates)
            .chain(more_predicates)
            .map(lambda value: Expr.new(value, as_col=True))
            .chain(Iter(constraints.items()).map_star(_constraint))
            .reduce(Expr.and_)
            .inner
        )
        return (
            exp
            .select(exp.Star())
            .from_(Tables.SRC, copy=False)
            .where(condition, copy=False)
            .pipe(self._cls)
        )

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
            .map(lambda key: Expr.new(key, as_col=True))
            .collect()
        )
        grouped_frame = (
            key_exprs.iter().map(lambda key: key.is_not_null()).into(self.filter)
            if drop_null_keys
            else self
        )
        return LazyGroupBy(grouped_frame, key_exprs, strategy)

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
        plan = self._schema.into(ExprPlan, exprs, more_exprs, named_exprs)
        return plan.group_by_all_ctx().pipe(
            self._from_ast,
            plan.agg_schema(Dict[str, exp.DataType].new()),
            src=self,
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

        def check_by_arg(
            compared: Seq[Expr],
            name: str,
            arg: TrySeq[bool],
        ) -> Result[Iter[bool], ValueError]:
            length = compared.length()
            match arg:
                case Sequence():
                    len_arg = len(arg)
                    if len_arg == length:
                        return Ok(try_iter(arg))
                    msg = f"the length of `{name}` ({len_arg}) does not match the length of `by` ({length})"
                    return Err(ValueError(msg))

                case _:
                    return Ok(try_iter(arg).cycle().take(length))

        order_exprs = (
            try_iter(by)
            .chain(more_by)
            .map(lambda v: Expr.new(v, as_col=True))
            .collect()
            .into(
                lambda sort_exprs: sort_exprs.iter().zip(
                    check_by_arg(sort_exprs, "descending", arg=descending).unwrap(),
                    check_by_arg(sort_exprs, "nulls_last", arg=nulls_last).unwrap(),
                )
            )
            .map_star(
                lambda expr, desc, nls: (
                    expr.order_by(descending=desc, nulls_last=nls).inner
                )
            )
        )
        return (
            exp
            .select(exp.Star())
            .from_(Tables.SRC, copy=False)
            .order_by(*order_exprs, copy=False)
            .pipe(self._cls)
        )

    def limit(self, n: int) -> Self:
        """Limit the number of rows.

        Args:
            n (int): The number of rows to limit.

        Returns:
            Self: A new LazyFrame with the limited rows.
        """
        return (
            exp
            .select(exp.Star())
            .from_(Tables.SRC, copy=False)
            .limit(exp.convert(n), copy=False)
            .pipe(self._cls)
        )

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

        def _qry(lf_length: Option[int], offset: int) -> Result[exp.Select, ValueError]:
            match (lf_length, offset):
                case (Some(length), _) if length < 0:
                    msg = f"negative slice lengths ({length}) are invalid for LazyFrame"
                    return Err(ValueError(msg))
                case (len_val, offset) if offset >= 0:
                    return Ok(
                        exp
                        .select(exp.Star())
                        .from_(Tables.SRC, copy=False)
                        .limit(exp.convert(len_val.unwrap_or(MAX_I64)), copy=False)
                        .offset(exp.convert(offset), copy=False)
                    )
                case (Some(0), _):
                    return Ok(
                        exp
                        .select(exp.Star())
                        .from_(Tables.SRC, copy=False)
                        .limit(exp.convert(0), copy=False)
                    )
                case (Some(length), offset):
                    slice_len_expr = col("slice_len")
                    stats = exp.select(lit(1).count().alias("slice_len").inner).from_(
                        Tables.SRC, copy=False
                    )
                    start_expr = slice_len_expr.add(offset).greatest(0).inner
                    return Ok(
                        exp
                        .select(exp.Star())
                        .from_(Tables.SRC, copy=False)
                        .with_("stats", as_=stats, copy=False)
                        .limit(
                            exp
                            .select(
                                slice_len_expr
                                .add(offset)
                                .add(length)
                                .least(slice_len_expr)
                                .sub(start_expr)
                                .greatest(0)
                                .inner
                            )
                            .from_(Tables.STATS, copy=False)
                            .subquery(copy=False)
                        )
                        .offset(
                            exp
                            .select(start_expr)
                            .from_(Tables.STATS, copy=False)
                            .subquery(copy=False)
                        )
                    )
                case (_, offset):
                    return Ok(
                        exp
                        .select(exp.Star())
                        .from_(Tables.SRC, copy=False)
                        .offset(
                            exp
                            .select(lit(1).count().inner)
                            .from_(Tables.SRC, copy=False)
                            .subquery(copy=False)
                            .pipe(Expr)
                            .add(offset)
                            .greatest(0)
                            .inner
                        )
                    )

        return _qry(Option(length), offset).map(self._cls).unwrap()

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
        cols = (
            try_iter(columns)
            .chain(more_columns)
            .map(lambda e: Expr.new(e, as_col=True))
            .collect()
        )
        to_drop = cols.iter().map(lambda e: e.inner.output_name).collect(Set)
        schema = (
            self._schema
            .items()
            .iter()
            .filter_star(lambda name, _: name not in to_drop)
            .collect(Dict)
        )
        return (
            exp
            .select(cols.into(all).inner)
            .from_(Tables.SRC, copy=False)
            .pipe(self._from_ast, schema, src=self)
        )

    def drop_nulls(self, subset: TryIter[str] = None) -> Self:
        """Drop rows that contain null values.

        Args:
            subset (TryIter[str]): Columns to consider for null values.

        Returns:
            Self: A new LazyFrame with rows containing null values dropped.
        """
        return (
            Option(subset)
            .map(try_iter)
            .unwrap_or_else(self.columns.iter)
            .map(lambda name: col(name).is_not_null())
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
            self._schema
            .into(ExprPlan, columns, more_columns, {})
            .projections.iter()
            .map(lambda r: r.name)
            .collect()
        )
        to_explode = to_explode_names.iter().map(col).collect()
        target = (
            to_explode.first()
            if to_explode.length() == 1
            else (to_explode.first().list.zip(*to_explode.iter().skip(1), lit(1).eq(1)))
        )

        zipped_index = (
            to_explode_names
            .iter()
            .enumerate()
            .map_star(lambda idx, name: (name, idx + 1))
            .collect(Dict)
        )
        is_single_explode = to_explode.length() == 1

        def _project_col(name: str, *, nested: bool, replace: Expr) -> Expr:
            match (nested, name in to_explode_names):
                case (True, True):
                    if is_single_explode:
                        return replace.alias(name)
                    field = zipped_index.get_item(name).unwrap()
                    return replace.struct.extract(field).alias(name)
                case (False, True):
                    return lit(None).alias(name)
                case _:
                    return col(name)

        def _proj(*, nested: bool) -> Iter[exp.Expr]:
            replace = unnest(target) if nested else lit(None)
            return self.columns.iter().map(
                lambda name: _project_col(name, nested=nested, replace=replace).inner
            )

        cond = target.is_not_null().and_(target.list.length().gt(0))

        rhs = (
            exp
            .select(*_proj(nested=False))
            .from_(Tables.EXPLODE_SRC, copy=False)
            .where(cond.not_().inner, copy=False)
        )
        return (
            exp
            .select(*_proj(nested=True))
            .from_(Tables.EXPLODE_SRC, copy=False)
            .where(cond.inner, copy=False)
            .pipe(exp.union, rhs, copy=False)
            .with_(
                Tables.EXPLODE_SRC.name, as_=self._inner, materialized=False, copy=False
            )
            .pipe(self._make, self._sources, self._schema)
        )

    def union(self, other: Self) -> Self:
        slct = exp.select(exp.Star()).from_
        lhs = slct(Tables.LHS)
        rhs = slct(Tables.RHS)
        return self._from_ast(exp.union(lhs, rhs), self._schema, lhs=self, rhs=other)

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns.

        Args:
            mapping (Mapping[str, str]): A dictionary mapping old column names to new column names.

        Returns:
            Self: A new LazyFrame with the renamed columns.
        """
        return (
            self.columns
            .iter()
            .map(lambda c: col(c).alias(mapping.get(c, c)))
            .into(self.select)
        )

    def sql_query(self) -> ParsedQuery:
        """Generate a `ParsedQuery` object.

        Allow to format and display prettified `SQL`.

        Returns:
            ParsedQuery
        """
        from ._parser import ParsedQuery

        return ParsedQuery(self._inner)

    def explain(self, kind: ExplainType | ExplainTypeLiteral = "standard") -> str:
        return self._materialize().explain(kind)

    def unnest(
        self, columns: TryIter[IntoExprColumn], *more_columns: IntoExprColumn
    ) -> Self:
        ast, schema = resolve_unnest(self._schema, columns, more_columns)
        return self._from_ast(ast, schema, src=self)

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
        return self._iter_slct(Expr.count)

    def describe(self) -> Self:
        """Return descriptive statistics."""
        return self.__class__(self._materialize().describe())

    def sum(self) -> Self:
        """Aggregate the sum of each column.

        Returns:
            Self: A new LazyFrame with the sum of each column.
        """
        return self._iter_slct(Expr.sum)

    def mean(self) -> Self:
        """Aggregate the mean of each column.

        Returns:
            Self: A new LazyFrame with the mean of each column.
        """
        return self._iter_slct(Expr.mean)

    def median(self) -> Self:
        """Aggregate the median of each column.

        Returns:
            Self: A new LazyFrame with the median of each column.
        """
        return self._iter_slct(Expr.median)

    def min(self) -> Self:
        """Aggregate the minimum of each column.

        Returns:
            Self: A new LazyFrame with the minimum of each column.
        """
        return self._iter_slct(Expr.min)

    def max(self) -> Self:
        """Aggregate the maximum of each column.

        Returns:
            Self: A new LazyFrame with the maximum of each column.
        """
        return self._iter_slct(Expr.max)

    def std(self, ddof: int = 1) -> Self:
        """Aggregate the standard deviation of each column.

        Args:
            ddof (int): Delta Degrees of Freedom.

        Returns:
            Self: A new LazyFrame with the standard deviation of each column.
        """
        fn = partial(Expr.std, ddof=ddof)
        return self._iter_slct(fn)

    def var(self, ddof: int = 1) -> Self:
        """Aggregate the variance of each column.

        Args:
            ddof (int): Delta Degrees of Freedom.

        Returns:
            Self: A new LazyFrame with the variance of each column.
        """
        fn = partial(Expr.var, ddof=ddof)
        return self._iter_slct(fn)

    def null_count(self) -> Self:
        """Return the null count of each column."""
        return self._iter_slct(Expr.null_count)

    def quantile(self, quantile: float) -> Self:
        """Compute quantile for each column.

        Args:
            quantile (float): The quantile to compute.

        Returns:
            Self: A new LazyFrame with the computed quantile for each column.
        """
        return self._iter_slct(lambda c: c.quantile_cont(quantile))

    def fill_nan(self, value: float | Expr | None) -> Self:
        """Fill NaN values.

        Args:
            value (float | Expr | None): The value to replace NaNs with.

        Returns:
            Self: A new LazyFrame with NaNs filled.
        """
        return self._iter_slct(lambda c: c.fill_nan(value))

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
        return self._iter_slct(lambda c: c.fill_null(value, strategy, limit))

    def shift(self, n: int = 1, *, fill_value: IntoExpr = None) -> Self:
        """Shift values by n positions.

        Args:
            n (int): Number of positions to shift.
            fill_value (IntoExpr | None): The value to use for filling empty positions.

        Returns:
            Self: A new LazyFrame with shifted values.
        """
        return self._iter_slct(lambda c: c.shift(n).coalesce(fill_value))

    def clone(self) -> Self:
        """Create a copy of the LazyFrame.

        Returns:
            Self: A new LazyFrame that is a copy of the current one.
        """
        return self._make(self._inner, self._sources, self._schema)

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
    def columns(self) -> PyoKeysView[str]:
        """Get column names."""
        return self._schema.keys()

    @property
    def dtypes(self) -> Iter[dt.DataType]:
        """Get column data types."""
        return self._schema.values().iter().map(dt.DataType.from_sql)

    @property
    def width(self) -> int:
        """Get number of columns."""
        return self._schema.length()

    @property
    def schema(self) -> Dict[str, dt.DataType]:
        from warnings import warn

        msg = "Resolving schema in a lazy context can be expensive. Use `collect_schema()` if you need to resolve the schema of a LazyFrame."
        warn(message=msg, stacklevel=2)
        return self.collect_schema()

    def collect_schema(self) -> Dict[str, dt.DataType]:
        """Collect the schema (same as schema property for lazy).

        Returns:
            Schema: The schema of the LazyFrame.
        """
        return (
            self._schema
            .items()
            .iter()
            .map_star(lambda name, dtype: (name, dt.DataType.from_sql(dtype)))
            .collect(Dict)
        )

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
        join_type = "full outer" if how == "outer" else how
        condition = join_keys.get_join_condition(builder)
        return (
            builder
            .get_join_cols(other, join_keys, how)
            .into(select)
            .from_(Tables.LHS, copy=False)
            .join("rhs", on=condition, join_type=join_type)
            .pipe(
                self._from_ast,
                schema=builder.join_schema(self._schema, other._schema, join_keys, how),
                lhs=self,
                rhs=other,
            )
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
        ast = (
            builder
            .get_join_cols_cross()
            .into(select)
            .from_(Tables.LHS, copy=False)
            .join("rhs", join_type="cross")
        )
        return self._from_ast(
            ast,
            schema=builder.join_schema_cross(self._schema, other._schema),
            lhs=self,
            rhs=other,
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
        on_opt = Option(on)
        on_keys = JoinKeys.from_on(on_opt, Option(left_on), Option(right_on)).unwrap()
        by_keys = JoinKeys.from_by(
            try_seq(by), try_seq(by_left), try_seq(by_right)
        ).unwrap()
        drop_keys = SetMut(by_keys.right)
        _ = on_opt.map(lambda _: drop_keys.add(on_keys.right))
        builder = JoinBuilder(suffix, self.columns, drop_keys)

        def _get_strategy(expr: Expr) -> Expr:
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
            .chain(builder.lhs(on_keys.left).pipe(_get_strategy).pipe(Iter.once))
            .reduce(Expr.and_)
            .inner
        )
        schema = (
            self._schema
            .items()
            .iter()
            .chain(
                other._schema
                .items()
                .iter()
                .filter_star(lambda name, _: name not in drop_keys)
                .map_star(
                    lambda name, dtype: (
                        f"{name}{suffix}" if name in self.columns else name,
                        dtype,
                    )
                )
            )
            .collect(Dict)
        )
        return (
            builder.left
            .iter()
            .map(builder.lhs)
            .chain(other.columns.iter().filter_map(builder.for_inner_left))
            .map(lambda c: c.inner)
            .into(select)
            .from_(Tables.LHS, copy=False)
            .join("rhs", on=by_cond, join_type="asof left")
            .pipe(self._from_ast, schema=schema, lhs=self, rhs=other)
        )

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

        def _query() -> Result[exp.Select, ValueError]:  # noqa: PLR0911
            match (keep, try_seq(order_by), try_seq(subset)):
                case ("none", _, Null()):
                    return Ok(
                        exp
                        .select(exp.Star())
                        .from_(Tables.SRC, copy=False)
                        .group_by("ALL")
                        .having(lit(1).count().eq(1).inner)
                    )
                case ("any", _, Null()):
                    return Ok(
                        exp.select(exp.Star()).from_(Tables.SRC, copy=False).distinct()
                    )
                case ("first" | "last", Some(_), Null()):
                    return Ok(
                        exp.select(exp.Star()).from_(Tables.SRC, copy=False).distinct()
                    )
                case ("none", _, Some(subset_names)):
                    subset_exprs = subset_names.iter().map(exp.column).collect()
                    rhs = (
                        exp
                        .select(*subset_exprs)
                        .from_(Tables.SRC, copy=False)
                        .group_by(*subset_exprs)
                        .having(lit(1).count().eq(1).inner)
                        .subquery(Tables.RHS.name, copy=False)
                    )
                    condition = (
                        subset_names
                        .iter()
                        .map(
                            lambda name: exp.NullSafeEQ(
                                this=col(name, table=Tables.LHS.name).inner,
                                expression=col(name, table=Tables.RHS.name).inner,
                            )
                        )
                        .map(Expr)
                        .reduce(Expr.and_)
                        .inner
                    )
                    res = (
                        exp
                        .select("lhs.*")
                        .from_("src AS lhs")
                        .join(rhs, on=condition, join_type="semi")
                    )
                    return Ok(res)
                case ("last", Some(order_cols), Some(subset_names)):
                    return Ok(
                        _distinct_on(
                            subset_names, order_cols, descending=True, nulls_last=True
                        )
                    )
                case ("any" | "first", order_cols, Some(subset_names)):
                    return Ok(
                        _distinct_on(
                            subset_names,
                            order_cols.unwrap_or_else(Seq[str].new),
                            descending=False,
                            nulls_last=False,
                        )
                    )
                case _:
                    msg = """`order_by` must be specified when `keep` is 'first' or 'last'
                    because LazyFrame makes no assumptions about row order."""
                    return Err(ValueError(msg))

        def _distinct_on(
            subset_names: Seq[str],
            order_names: Seq[str],
            *,
            descending: bool,
            nulls_last: bool,
        ) -> exp.Select:
            order_exprs = (
                subset_names
                .iter()
                .map(col)
                .chain(
                    order_names.iter().map(
                        lambda name: col(name).order_by(
                            descending=descending, nulls_last=nulls_last
                        )
                    )
                )
                .map(lambda expr: expr.inner)
            )
            return (
                exp
                .select(exp.Star())
                .from_(Tables.SRC, copy=False)
                .distinct(*subset_names)
                .order_by(*order_exprs)
            )

        return _query().unwrap().pipe(self._cls)

    def pivot(  # noqa: PLR0913
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
        pivoted, multi, idx_cols, val_cols = pivot(
            self.columns,
            on,
            on_columns,
            index,
            values,
            aggregate_function,
            maintain_order=maintain_order,
        )
        unknown = exp.DType.UNKNOWN.into_expr()
        on_values = Iter(on_columns).map(str).collect()
        pivot_schema = (
            idx_cols
            .iter()
            .map(lambda name: (name, self._schema.get_item(name).unwrap()))
            .chain(
                on_values.iter().flat_map(
                    lambda ov: (
                        Iter.once((ov, unknown))
                        if not multi
                        else val_cols.iter().map(
                            lambda vc: (f"{vc}{separator}{ov}", unknown)
                        )
                    )
                )
            )
            .collect(Dict)
        )

        def _handle_multi(expr: exp.Selectable) -> Self:
            if multi:
                on_values = Iter(on_columns).map(str).collect()

                def _rename_col(val_col: str) -> Iter[Expr]:
                    def _swap(on_val: str) -> Expr:
                        in_ = f"{on_val}_{val_col}"
                        out = f"{val_col}{separator}{on_val}"
                        return col(in_).alias(out)

                    return on_values.iter().map(_swap)

                return (
                    idx_cols
                    .iter()
                    .map(col)
                    .chain(val_cols.iter().flat_map(_rename_col))
                    .into(expr.pipe(self._from_ast, pivot_schema, src=self).select)
                )
            return expr.pipe(self._from_ast, pivot_schema, src=self)

        return pivoted.pipe(_handle_multi)

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
        index_set = try_iter(index).collect(dict.fromkeys)
        match on:
            case None:
                first_dtype = self.columns.iter().next()
            case _:
                first_dtype = try_iter(on).next()
        value_dtype = first_dtype.and_then(self._schema.get_item).unwrap_or_else(
            exp.DType.UNKNOWN.into_expr
        )
        unpivot_schema = (
            self._schema
            .items()
            .iter()
            .filter_star(lambda name, _: name in index_set)
            .chain((
                (variable_name, exp.DType.VARCHAR.into_expr()),
                (value_name, value_dtype),
            ))
            .collect(Dict)
        )
        return self.columns.into(
            unpivot, on, index, variable_name, value_name, order_by
        ).pipe(self._from_ast, unpivot_schema, src=self)

    def with_row_index(self, name: str, *, order_by: TryIter[str]) -> Self:
        """Insert row index based on order_by.

        Args:
            name (str): The name of the new index column.
            order_by (TrySeq[str]): Columns to order by for row numbering.

        Returns:
            Self: A new LazyFrame with the row index added.
        """
        row_nb = row_number().window(order_by=order_by).sub(1).alias(name).inner
        schema = (
            Iter
            .once((name, exp.DType.BIGINT.into_expr()))
            .chain(self._schema.items())
            .collect(Dict)
        )
        return self._from_ast(
            exp.select(row_nb, exp.Star()).from_(Tables.SRC, copy=False),
            schema,
            src=self,
        )

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

    def cast(self, dtypes: Mapping[str, dt.DataType] | dt.DataType) -> Self:
        """Cast columns to specified dtypes.

        Args:
            dtypes (Mapping[str, dt.DataType] | dt.DataType): The target data types for the columns.

        Returns:
            Self: A new LazyFrame with the columns cast to the specified dtypes.
        """
        match dtypes:
            case Mapping():
                dtype_map = Dict(dtypes)
                return self._iter_slct(
                    lambda c: (
                        dtype_map
                        .get_item(c.inner.output_name)
                        .map(lambda dtype: c.cast(dtype.raw))
                        .unwrap_or(c)
                    )
                )
            case _:
                return self._iter_slct(lambda c: c.cast(dtypes.raw))

    def sink_parquet(  # noqa: PLR0913
        self,
        path: str | Path,
        *,
        compression: ParquetCompression | None = None,
        field_ids: ParquetFieldsOptions | None = None,
        row_group_size_bytes: str | int | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: list[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None:
        """Write to Parquet file."""
        self._materialize().write_parquet(
            str(path),
            compression=compression,
            field_ids=field_ids,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size=row_group_size,
            overwrite=overwrite,
            per_thread_output=per_thread_output,
            use_tmp_file=use_tmp_file,
            partition_by=partition_by,
            write_partition_columns=write_partition_columns,
            append=append,
            filename_pattern=filename_pattern,
            file_size_bytes=file_size_bytes,
        )

    def sink_csv(  # noqa: PLR0913
        self,
        path: str | Path,
        *,
        separator: str = ",",
        include_header: bool = True,
        na_rep: str | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: str | None = None,
        compression: CsvCompression | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: list[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None:
        """Write to CSV file."""
        self._materialize().write_csv(
            str(path),
            sep=separator,
            header=include_header,
            na_rep=na_rep,
            quotechar=quotechar,
            escapechar=escapechar,
            date_format=date_format,
            timestamp_format=timestamp_format,
            quoting=quoting,
            encoding=encoding,
            compression=compression,
            overwrite=overwrite,
            per_thread_output=per_thread_output,
            use_tmp_file=use_tmp_file,
            partition_by=partition_by,
            write_partition_columns=write_partition_columns,
        )

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
            Option(subset)
            .map(try_iter)
            .unwrap_or_else(self.columns.iter)
            .map(lambda name: col(name).is_nan().not_())
            .into(self.filter)
        )

    def fetch_all(self) -> Vec[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
        return Vec.from_ref(self._materialize().fetchall())

    def show(
        self,
        max_width: SupportsInt | None = None,
        max_rows: SupportsInt | None = None,
        max_col_width: SupportsInt | None = None,
        null_value: str | None = None,
        render_mode: RenderModeLiteral | None = None,
    ) -> None:
        return self._materialize().show(
            max_width=max_width,
            max_rows=max_rows,
            max_col_width=max_col_width,
            null_value=null_value,
            render_mode=render_mode,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self._materialize().shape
