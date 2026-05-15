from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from duckdb import DuckDBPyRelation
from pyochain import Dict, Err, Iter, Null, Ok, Option, Result, Seq, Set, Some
from pyochain.traits import Pipeable
from sqlglot import exp

from belugas.typing import (
    IntoArrowArray,
    IntoArrowStream,
    IntoPlDataFrame,
    IntoPlLazyFrame,
    NPArrayLike,
)

from .._core import Marker, Tables
from ..utils import try_iter
from . import nodes, scans
from ._optimize import optimize_nodes

if TYPE_CHECKING:
    from .._expr import Cols, Expr
    from ..typing import IntoExpr, Schema, TryIter


class CompiledPlan(NamedTuple):
    ast: exp.Select
    schema: Schema
    sources: Dict[str, DuckDBPyRelation]


def compile_plan(node: nodes.Node, *, optimize: bool = True) -> CompiledPlan:
    root = optimize_nodes(node) if optimize else node
    return _compile_scan(root)


def _compile_scan(node: nodes.Node) -> CompiledPlan:
    """Compile a scan node into a CompiledPlan containing the SQL AST, output schema, and source relations.

    We recursively descend the logical plan tree.

    Conceptually:

        ```text
        `_compile_scan(node)`
        | // Call _compile_scan on the inner node until we find a scan node
        ---->   `_compile_scan(node.inner)`
                |// Re-match on LogicalNode, so it will again call _compile_scan on its inner node
                |// Found the root scan node
                ---->`_compile_scan(node.inner)`
                    |
                // Call `_compile_tree` on the way back up, combining sources from child nodes
                `return CompiledPlan(ast, schema, sources)`
                |
            // Combine the compiled source with the current node to produce a new CompiledPlan
            `return _compile_tree(ast, schema, node)`
            |
        // Return the final compiled plan for the root node.
        `return CompiledPlan(ast, schema, sources)`
        ```

    Returns:
        CompiledPlan: A named tuple containing the compiled SQL AST, output schema, and source relations.
    """
    match node:
        case nodes.LogicalNode():
            compiled_src = _compile_scan(node.inner)
            compiled_node = compiled_src.ast.pipe(
                _compile_tree, compiled_src.schema, node
            ).unwrap()
            sources = (
                compiled_src.sources
                .items()
                .iter()
                .chain(compiled_node.sources.items())
                .collect(Dict)
            )
            return CompiledPlan(compiled_node.ast, compiled_node.schema, sources)
        case nodes.BaseScan():
            source = _resolve_scan(node).set_alias()  # pyright: ignore[reportArgumentType]
            ast = exp.select(exp.Star()).from_(exp.to_table(source.identity))
            return CompiledPlan(
                ast, source.schema, Dict([(source.identity, source.relation)])
            )


class CompilationError(Exception):
    pass


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


def _compile_tree(  # noqa: PLR0915
    src_ast: exp.Select, schema: Schema, node: nodes.Node
) -> Result[CompiledPlan, CompilationError]:
    from . import ops

    empty = Dict[str, DuckDBPyRelation].new()

    match node:
        case nodes.GroupBy():
            # GroupBy is a descriptor node consumed by Agg/AggColumns.
            # At this stage we keep the compiled source unchanged and let
            # the parent aggregation node emit the actual GROUP BY query.
            return Ok(CompiledPlan(src_ast, schema, empty))
        case nodes.Agg() as agg_node:
            match node.inner:
                case nodes.GroupBy():
                    ast, new_schema = ops.agg(
                        src_ast,
                        schema,
                        node.inner.keys,
                        agg_node.exprs,
                        agg_node.more_exprs,
                        agg_node.named,
                        node.inner.strategy,
                        drop_null_keys=node.inner.drop_null_keys,
                    )
                    return Ok(CompiledPlan(ast, new_schema, empty))
                case _:
                    msg = f"Unexpected inner node for Agg: {type(node.inner)}"
                    return Err(CompilationError(msg))
        case nodes.AggColumns() as agg_cols:
            match node.inner:
                case nodes.GroupBy():
                    ast, new_schema = ops.agg_columns(
                        src_ast,
                        schema,
                        node.inner.keys,
                        agg_cols.func,
                        drop_null_keys=node.inner.drop_null_keys,
                    )
                    return Ok(CompiledPlan(ast, new_schema, empty))
                case _:
                    msg = f"Unexpected inner node for Agg: {type(node.inner)}"
                    return Err(CompilationError(msg))
        case nodes.Select():
            ast, new_schema = ops.select(
                src_ast,
                schema,
                node.exprs,
                node.more_exprs,
                node.named,
            )
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.SelectAll():
            ast, new_schema = ops.select_all(src_ast, schema, node.func)
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.WithColumns():
            ast, new_schema = ops.with_columns(
                src_ast,
                schema,
                node.exprs,
                node.more_exprs,
                node.named,
            )
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.Filter():
            predicate = ops.filter(
                node.predicates,
                node.more_predicates,
                node.constraints,
            )
            ast = _apply_filter_clause(src_ast, predicate)
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Sort():
            order_exprs = ops.sort(
                node.by,
                node.more_by,
                descending=node.descending,
                nulls_last=node.nulls_last,
            )
            ast = src_ast.order_by(*order_exprs, copy=False)
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Limit():
            ast = src_ast.limit(exp.Literal.number(node.n), copy=False)
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Slice():
            ast = ops.slice(src_ast, node.length, node.offset).unwrap()
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Drop():
            expr, new_schema = ops.drop(schema, node.columns, node.more_columns)
            ast = _maybe_inline(expr, ast=src_ast)
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.DropRows():
            ast = _apply_filter_clause(
                src_ast, ops.drop_rows(schema, node.subset, node.fn)
            )
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Explode():
            ast = ops.explode(src_ast, schema, node.columns, node.more_columns)
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Unnest():
            exprs, new_schema = ops.unnest(schema, node.columns, node.more_columns)
            ast = _maybe_inline(*exprs, ast=src_ast)
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.Rename():
            exprs, new_schema = ops.rename(schema, node.mapping)
            ast = _maybe_inline(*exprs, ast=src_ast)
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.Cast():
            ast, new_schema = ops.cast(src_ast, schema, node.dtypes)
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.WithRowIndex():
            row_nb, new_schema = ops.with_row_index(schema, node.name, node.order_by)
            src_ast.selects.insert(0, row_nb)
            return Ok(CompiledPlan(src_ast, new_schema, empty))
        case nodes.GroupByAll():
            exprs, new_schema = ops.group_by_all(
                schema, node.exprs, node.more_exprs, node.named
            )
            ast = exprs.unpack_into(_maybe_inline, ast=src_ast).group_by(
                "ALL", copy=False
            )
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.Unique():
            ast = ops.unique(src_ast, node.subset, node.keep, node.order_by).unwrap()
            return Ok(CompiledPlan(ast, schema, empty))
        case nodes.Pivot():
            ast, new_schema = ops.pivot(
                src_ast,
                schema,
                node.on,
                node.on_columns,
                node.index,
                node.values,
                node.aggregate_function,
                maintain_order=node.maintain_order,
                separator=node.separator,
            )
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.Unpivot():
            ast, new_schema = ops.unpivot(
                src_ast,
                schema,
                node.on,
                node.index,
                node.variable_name,
                node.value_name,
                node.order_by,
            )
            return Ok(CompiledPlan(ast, new_schema, empty))
        case nodes.Union():
            other = compile_plan(node.other, optimize=False)
            ast = ops.union(src_ast, other.ast)
            return Ok(CompiledPlan(ast, schema, other.sources))
        case nodes.Join():
            other = compile_plan(node.other, optimize=False)
            ast, new_schema = ops.join(
                src_ast,
                other.ast,
                schema,
                other.schema,
                node.on,
                node.how,
                node.left_on,
                node.right_on,
                node.suffix,
            )
            return Ok(CompiledPlan(ast, new_schema, other.sources))
        case nodes.JoinCross():
            other = compile_plan(node.other, optimize=False)
            ast, new_schema = ops.join_cross(
                src_ast, other.ast, schema, other.schema, node.suffix
            )
            return Ok(CompiledPlan(ast, new_schema, other.sources))
        case nodes.JoinAsof():
            other = compile_plan(node.other, optimize=False)
            ast, new_schema = ops.join_asof(
                src_ast,
                other.ast,
                schema,
                other.schema,
                node.left_on,
                node.right_on,
                node.on,
                node.by_left,
                node.by_right,
                node.by,
                node.strategy,
                node.suffix,
            )
            return Ok(CompiledPlan(ast, new_schema, other.sources))
        case _:
            msg = f"Unsupported node type: {type(node)}"
            return Err(CompilationError(msg))


def _apply_filter_clause(src_ast: exp.Select, predicate: exp.Expr) -> exp.Select:
    match src_ast.args.get("group"):
        case exp.Group():
            return src_ast.having(predicate, copy=False)
        case _ if (
            src_ast.args.get("limit") is not None
            or src_ast.args.get("offset") is not None
            or not Iter(src_ast.selects).all(_is_passthrough_projection)
        ):
            return (
                exp
                .select(exp.Star(), copy=False)
                .from_(src_ast.subquery(Tables.SRC, copy=False))
                .where(predicate, copy=False)
            )
        case _:
            return src_ast.where(predicate, copy=False)


def _maybe_inline(*exprs: exp.Expr, ast: exp.Select) -> exp.Select:
    if can_inline_select(ast):
        return ast.select(*exprs, copy=False, append=False)
    return exp.select(*exprs).from_(ast.subquery(Tables.SRC, copy=False), copy=False)


# TODO: we should factorise those two funcs and check if we can do it more efficiently
def can_inline_select(select: exp.Select) -> bool:
    match select.selects:
        case [exp.Star()]:
            return True
        case _:
            return False


def _is_passthrough_projection(expr: exp.Expr) -> bool:
    match expr:
        case exp.Star():
            return True
        case _:
            match expr.unalias():
                case exp.Column() as col:
                    return expr.output_name == col.output_name
                case _:
                    return False


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
