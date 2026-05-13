from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from duckdb import DuckDBPyRelation
from pyochain import Dict, Err, Iter, Null, Ok, Option, Result, Seq, Set, Some, Vec
from pyochain.traits import Pipeable
from sqlglot import exp

from belugas.typing import (
    IntoArrowArray,
    IntoArrowStream,
    IntoPlDataFrame,
    IntoPlLazyFrame,
    NPArrayLike,
)

from .._core import Marker
from ..utils import try_iter
from . import nodes, scans
from ._common import as_relation
from ._optimize import optimize_nodes

if TYPE_CHECKING:
    from .._expr import Cols, Expr
    from ..typing import IntoExpr, Schema, TryIter


class CompiledPlan(NamedTuple):
    ast: exp.Selectable
    schema: Schema
    sources: Dict[str, DuckDBPyRelation]


type Pending = Vec[nodes.DeferredNode]


def compile_plan(node: nodes.Node, *, optimize: bool = True) -> CompiledPlan:
    root = optimize_nodes(node) if optimize else node
    return _compile_node(root, Vec[nodes.DeferredNode].new()).unwrap()


class CompilationError(Exception):
    pass


def _compile_node(  # noqa: PLR0915
    node: nodes.Node, pending: Pending
) -> Result[CompiledPlan, CompilationError]:
    from . import ops

    match node:
        case nodes.BaseScan():
            return Ok(_compile_scan(node, pending))
        case nodes.DeferredNode():
            pending.insert(0, node)
            return Ok(_compile_node(node.inner, pending).unwrap())
        case nodes.GroupBy():
            return Ok(_compile_node(node.inner, pending).unwrap())
        case nodes.Agg() as agg_node:
            match node.inner:
                case nodes.GroupBy() as group_by:
                    source = _compile_node(
                        group_by.inner, Vec[nodes.DeferredNode].new()
                    ).unwrap()
                    ast, new_schema = ops.agg(
                        source.ast,
                        source.schema,
                        node.inner.keys,
                        agg_node.exprs,
                        agg_node.more_exprs,
                        agg_node.named,
                        node.inner.strategy,
                        drop_null_keys=node.inner.drop_null_keys,
                    )
                    return Ok(
                        _apply_deferred(
                            CompiledPlan(ast, new_schema, source.sources), pending
                        )
                    )
                case _:
                    msg = f"Unexpected inner node for Agg: {type(node.inner)}"
                    return Err(CompilationError(msg))
        case nodes.AggColumns() as agg_cols:
            match node.inner:
                case nodes.GroupBy():
                    source = _compile_node(
                        node.inner, Vec[nodes.DeferredNode].new()
                    ).unwrap()
                    ast, new_schema = ops.agg_columns(
                        source.ast,
                        source.schema,
                        node.inner.keys,
                        agg_cols.func,
                        drop_null_keys=node.inner.drop_null_keys,
                    )
                    return Ok(
                        _apply_deferred(
                            CompiledPlan(ast, new_schema, source.sources), pending
                        )
                    )
                case _:
                    msg = f"Unexpected inner node for Agg: {type(node.inner)}"
                    return Err(CompilationError(msg))
        case nodes.GroupByAll():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast, schema = ops.group_by_all(
                source.ast, source.schema, node.exprs, node.more_exprs, node.named
            )
            return Ok(
                _apply_deferred(CompiledPlan(ast, schema, source.sources), pending)
            )
        case nodes.Explode():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast = ops.explode(
                source.ast, source.schema, node.columns, node.more_columns
            )
            return Ok(
                _apply_deferred(
                    CompiledPlan(ast, source.schema, source.sources), pending
                )
            )
        case nodes.Unnest():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast, schema = ops.unnest(
                source.ast, source.schema, node.columns, node.more_columns
            )
            return Ok(
                _apply_deferred(CompiledPlan(ast, schema, source.sources), pending)
            )
        case nodes.Pivot():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast, new_schema = ops.pivot(
                source.ast,
                source.schema,
                node.on,
                node.on_columns,
                node.index,
                node.values,
                node.aggregate_function,
                maintain_order=node.maintain_order,
                separator=node.separator,
            )
            return Ok(
                _apply_deferred(CompiledPlan(ast, new_schema, source.sources), pending)
            )
        case nodes.Unpivot():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast, new_schema = ops.unpivot(
                source.ast,
                source.schema,
                node.on,
                node.index,
                node.variable_name,
                node.value_name,
                node.order_by,
            )
            return Ok(
                _apply_deferred(CompiledPlan(ast, new_schema, source.sources), pending)
            )
        case nodes.Slice():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast = ops.slice(source.ast, node.length, node.offset).unwrap()
            return Ok(
                _apply_deferred(
                    CompiledPlan(ast, source.schema, source.sources), pending
                )
            )
        case nodes.Unique():
            source = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            ast = ops.unique(source.ast, node.subset, node.keep, node.order_by).unwrap()
            return Ok(
                _apply_deferred(
                    CompiledPlan(ast, source.schema, source.sources), pending
                )
            )
        case nodes.Union():
            lhs = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            rhs = _compile_node(node.other, Vec[nodes.DeferredNode].new()).unwrap()
            ast = ops.union(lhs.ast, rhs.ast)
            lhs.sources.update(rhs.sources.items())
            return Ok(
                _apply_deferred(CompiledPlan(ast, lhs.schema, lhs.sources), pending)
            )
        case nodes.Join():
            lhs = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            rhs = _compile_node(node.other, Vec[nodes.DeferredNode].new()).unwrap()
            ast, schema = ops.join(
                lhs.ast,
                rhs.ast,
                lhs.schema,
                rhs.schema,
                node.on,
                node.how,
                node.left_on,
                node.right_on,
                node.suffix,
            )
            lhs.sources.update(rhs.sources.items())
            return Ok(_apply_deferred(CompiledPlan(ast, schema, lhs.sources), pending))
        case nodes.JoinCross():
            lhs = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            rhs = _compile_node(node.other, Vec[nodes.DeferredNode].new()).unwrap()
            ast, new_schema = ops.join_cross(
                lhs.ast, rhs.ast, lhs.schema, rhs.schema, node.suffix
            )
            lhs.sources.update(rhs.sources.items())
            return Ok(
                _apply_deferred(CompiledPlan(ast, new_schema, lhs.sources), pending)
            )
        case nodes.JoinAsof():
            lhs = _compile_node(node.inner, Vec[nodes.DeferredNode].new()).unwrap()
            rhs = _compile_node(node.other, Vec[nodes.DeferredNode].new()).unwrap()
            ast, schema = ops.join_asof(
                lhs.ast,
                rhs.ast,
                lhs.schema,
                rhs.schema,
                node.left_on,
                node.right_on,
                node.on,
                node.by_left,
                node.by_right,
                node.by,
                node.strategy,
                node.suffix,
            )
            lhs.sources.update(rhs.sources.items())
            return Ok(_apply_deferred(CompiledPlan(ast, schema, lhs.sources), pending))


def _compile_scan(node: nodes.BaseScan, pending: Pending) -> CompiledPlan:
    source = _resolve_scan(node).set_alias()  # pyright: ignore[reportArgumentType]
    base = CompiledPlan(
        exp.select(exp.Star()).from_(exp.to_table(source.identity)),
        source.schema,
        Dict([(source.identity, source.relation)]),
    )
    return _apply_deferred(base, pending)


def _resolve_scan(node: nodes.Scan) -> scans.ScanResult:
    match node:
        case nodes.ScanInMemory():
            match node.data:
                case None:
                    return scans.from_dict({Marker.TEMP: ()})
                case DuckDBPyRelation():
                    return scans.from_query(node.data)
                case Mapping():
                    return scans.from_dict(node.data)
                case NPArrayLike():
                    return scans.from_numpy(node.data, orient=node.orient)
                case IntoPlDataFrame() | IntoPlLazyFrame():
                    return scans.from_polars(node.data)
                case IntoArrowStream() | IntoArrowArray():
                    return scans.from_arrow(node.data)
                case Sequence():
                    return scans.from_records(node.data, orient=node.orient)
        case nodes.ScanTable():
            return scans.from_table(node.table)
        case nodes.ScanTableFunction():
            return scans.from_table_function(node.function)
        case nodes.ScanCSV():
            return scans.from_csv(node.path, node.connection, node.options)
        case nodes.ScanParquet():
            return scans.from_parquet(node.path, node.connection, node.options)
        case nodes.ScanJson():
            return scans.from_json(node.path, node.connection, node.options)


def _apply_deferred(plan: CompiledPlan, pending: Pending) -> CompiledPlan:
    def _apply(current: CompiledPlan, node: nodes.DeferredNode) -> CompiledPlan:
        from . import ops

        match node:
            case nodes.Select():
                ast, schema = ops.select(
                    current.ast,
                    current.schema,
                    node.exprs,
                    node.more_exprs,
                    node.named,
                )
                return CompiledPlan(ast, schema, current.sources)
            case nodes.SelectAll():
                ast, schema = ops.select_all(current.ast, current.schema, node.func)
                return CompiledPlan(ast, schema, current.sources)
            case nodes.WithColumns():
                ast, schema = ops.with_columns(
                    current.ast,
                    current.schema,
                    node.exprs,
                    node.more_exprs,
                    node.named,
                )
                return CompiledPlan(ast, schema, current.sources)
            case nodes.Filter():
                condition = ops.filter(
                    node.predicates,
                    node.more_predicates,
                    node.constraints,
                )
                ast = _apply_where(current.ast, condition)
                return CompiledPlan(ast, current.schema, current.sources)
            case nodes.Sort():
                order_exprs = ops.sort(
                    node.by,
                    node.more_by,
                    node.descending,
                    node.nulls_last,
                )
                ast = _apply_order(current.ast, order_exprs)
                return CompiledPlan(ast, current.schema, current.sources)
            case nodes.Limit():
                limit_expr = ops.limit(node.n)
                ast = _apply_limit(current.ast, limit_expr)
                return CompiledPlan(ast, current.schema, current.sources)
            case nodes.Drop():
                ast, schema = ops.drop(
                    current.ast,
                    current.schema,
                    node.columns,
                    node.more_columns,
                )
                return CompiledPlan(ast, schema, current.sources)
            case nodes.DropRows():
                condition = ops.drop_rows(
                    current.schema,
                    node.subset,
                    node.fn,
                )
                ast = _apply_where(current.ast, condition)
                return CompiledPlan(ast, current.schema, current.sources)
            case nodes.Rename():
                ast, schema = ops.rename(current.ast, current.schema, node.mapping)
                return CompiledPlan(ast, schema, current.sources)
            case nodes.Cast():
                ast, schema = ops.cast(current.ast, current.schema, node.dtypes)
                return CompiledPlan(ast, schema, current.sources)
            case nodes.WithRowIndex():
                ast, schema = ops.with_row_index(
                    current.ast,
                    current.schema,
                    node.name,
                    node.order_by,
                )
                return CompiledPlan(ast, schema, current.sources)
            case _:
                return current

    return pending.iter().fold(plan, _apply)


def _apply_where(ast: exp.Selectable, condition: exp.Expr) -> exp.Selectable:
    if isinstance(ast, exp.Select) and _can_inline_filter(ast):
        current_where = ast.args.get("where")
        existing_condition = (
            current_where.args.get("this")
            if isinstance(current_where, exp.Where)
            else None
        )
        if isinstance(existing_condition, exp.Expr):
            from .._expr import Expr

            condition = Expr(existing_condition).and_(Expr(condition)).inner
        ast.set("where", exp.Where(this=condition))
        return ast

    return (
        exp
        .select(exp.Star())
        .from_(as_relation(ast), copy=False)
        .where(condition, copy=False)
    )


def _apply_order(ast: exp.Selectable, order_exprs: Seq[exp.Expr]) -> exp.Selectable:
    if isinstance(ast, exp.Select) and _can_inline_order(ast):
        ast.set("order", exp.Order(expressions=list(order_exprs)))
        return ast

    return (
        exp
        .select(exp.Star())
        .from_(as_relation(ast), copy=False)
        .order_by(*order_exprs, copy=False)
    )


def _apply_limit(ast: exp.Selectable, limit_expr: exp.Expr) -> exp.Selectable:
    if isinstance(ast, exp.Select) and _can_inline_limit(ast):
        ast.set("limit", exp.Limit(expression=limit_expr))
        return ast

    return (
        exp
        .select(exp.Star())
        .from_(as_relation(ast), copy=False)
        .limit(limit_expr, copy=False)
    )


def _is_passthrough_select(ast: exp.Selectable) -> bool:
    if not isinstance(ast, exp.Select):
        return False
    if len(ast.expressions) != 1:
        return False
    if not isinstance(ast.expressions[0], exp.Star):
        return False
    if ast.args.get("group") is not None:
        return False
    if ast.args.get("having") is not None:
        return False
    return ast.args.get("distinct") is None


def _can_inline_filter(ast: exp.Selectable) -> bool:
    return (
        _is_passthrough_select(ast)
        and ast.args.get("limit") is None
        and ast.args.get("offset") is None
    )


def _can_inline_order(ast: exp.Selectable) -> bool:
    return (
        _is_passthrough_select(ast)
        and ast.args.get("limit") is None
        and ast.args.get("offset") is None
    )


def _can_inline_limit(ast: exp.Selectable) -> bool:
    return _is_passthrough_select(ast)


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
    has_distinct: bool
    is_pure_reducer: bool

    def __init__(self, expr: Expr, name: str) -> None:
        self.name = name
        self.expr, self.has_distinct, self.is_pure_reducer = into_resolved(expr)


def into_resolved(expr: Expr) -> tuple[Expr, bool, bool]:
    inner = expr.inner

    def _is_projection_distinct(node: exp.Expr) -> bool:
        return node.find_ancestor(exp.AggFunc, exp.List, exp.Window) is None

    def _classify(
        acc: tuple[bool, bool, bool], node: exp.Expr
    ) -> tuple[bool, bool, bool]:
        has_distinct, has_bare_agg, has_bare_col = acc
        match node, _is_projection_distinct(node):
            case exp.Distinct(), True:
                return (True, has_bare_agg, has_bare_col)
            case exp.Column(), True:
                return (has_distinct, has_bare_agg, True)
            case exp.AggFunc() | exp.List(), _ if not has_window_ancestor(node):
                return (has_distinct, True, has_bare_col)
            case _:
                return acc

    has_distinct, has_bare_agg, has_bare_col = Iter(inner.walk(bfs=False)).fold(
        (False, False, False), _classify
    )
    if has_distinct:

        def _strip(node: exp.Expr) -> exp.Expr:
            match node:
                case exp.Distinct(expressions=[exp.Expr() as expr]):
                    return expr
                case _:
                    return node

        expr = inner.transform(_strip).pipe(expr.__class__)
    return expr, has_distinct, has_bare_agg and not has_bare_col


def has_window_ancestor(node: exp.Expr) -> bool:
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


def _resolve(val: IntoExpr, schema: Schema) -> Iter[ResolvedExpr]:
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
                    match val.inner.find(exp.Columns, exp.Star):
                        case None:
                            name = extract_root_name(val.inner)
                            return ResolvedExpr(val, name).into(Iter.once)
                        case exp.Star() as star:
                            excepts: list[exp.Expr] | None = star.args.get("except_")
                            match excepts:
                                case None:
                                    base_names = schema.keys()
                                case list():
                                    excluded = (
                                        Iter(excepts).map(lambda c: c.name).collect(Set)
                                    )
                                    base_names = (
                                        schema
                                        .iter()
                                        .filter(lambda n: n not in excluded)
                                        .collect()
                                    )
                            return _expand_columns(val, base_names, base_names)
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


def extract_root_name(node: exp.Expr) -> str:
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
