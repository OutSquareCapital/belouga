from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Seq, Set, Vec
from sqlglot import exp

from ..._core import Tables
from ..._expr import Expr
from ..._funcs import col
from .._resolve import ResolvedExpr, lookup_type, resolve_all

if TYPE_CHECKING:
    from ...typing import GroupByClause, IntoExpr, Schema, TryIter


def group_by_all(
    schema: Schema,
    exprs: TryIter[IntoExpr],
    more_exprs: Iterable[IntoExpr],
    named_exprs: Dict[str, IntoExpr],
) -> tuple[Vec[exp.Expr], Schema]:
    # TODO: determine if this is really the best approach.
    # What can DuckDB do automatically here? What to consider VS our group_by.agg?
    # Because actually in the test it's not really clear what the difference is between group_by_all and group_by.agg.
    def _acc(
        acc: tuple[Vec[exp.Expr], Schema], proj: ResolvedExpr
    ) -> tuple[Vec[exp.Expr], Schema]:
        select_exprs, schema = acc
        base = lookup_type(proj.expr.inner, schema)
        _ = select_exprs.append(proj.expr.alias(proj.name).inner)
        _ = schema.insert(
            proj.name, _to_array(base, is_pure_reducer=proj.is_pure_reducer)
        )
        return acc

    return (
        resolve_all(schema, exprs, more_exprs, named_exprs)
        .iter()
        .fold((Vec(()), Dict(())), _acc)
    )


def agg_columns(
    ast: exp.Select,
    schema: Schema,
    keys: Seq[Expr],
    func: Callable[[Expr], Expr],
    *,
    drop_null_keys: bool,
) -> tuple[exp.Select, Schema]:

    key_names = keys.iter().map(lambda k: k.inner.output_name).collect(Set)
    agg_exprs = (
        schema
        .iter()
        .filter(lambda name: name not in key_names)
        .map(lambda name: col(name).pipe(func).alias(name))
    )
    return agg(
        ast, schema, keys, agg_exprs, (), Dict(()), None, drop_null_keys=drop_null_keys
    )


def agg(  # noqa: PLR0913, PLR0917
    ast: exp.Select,
    schema: Schema,
    keys: Seq[Expr],
    aggs: TryIter[IntoExpr],
    more_aggs: Iterable[IntoExpr],
    named_aggs: Dict[str, IntoExpr],
    strategy: GroupByClause | None,
    *,
    drop_null_keys: bool,
) -> tuple[exp.Select, Schema]:

    key_glots = keys.iter().map(lambda k: k.inner).collect(Vec)
    key_names = key_glots.iter().map(lambda e: e.output_name).collect(Set)
    key_schema, non_key_schema = (
        schema
        .items()
        .iter()
        .fold_star(
            (Dict(()), Dict(())),
            lambda acc, name, dtype: _split_by_key(key_names, acc, name, dtype),
        )
    )
    projections = resolve_all(non_key_schema, aggs, more_aggs, named_aggs)

    def _acc(
        acc: tuple[Vec[exp.Expr], Schema], proj: ResolvedExpr
    ) -> tuple[Vec[exp.Expr], Schema]:
        select_exprs, out_schema = acc
        match proj.expr.inner, proj.is_pure_reducer:
            case exp.Explode(this=exp.Expr() as inner), _:
                resolved = (
                    proj.expr
                    .__class__(inner)
                    .pipe(_exploded, is_distinct=proj.has_distinct)
                    .list.flatten()
                )
            case _, True:
                resolved = proj.expr
            case _, False:
                resolved = proj.expr.pipe(_exploded, is_distinct=proj.has_distinct)
        _ = select_exprs.append(resolved.alias(proj.name).inner)
        base = lookup_type(proj.expr.inner, non_key_schema)
        _ = out_schema.insert(
            proj.name, _to_array(base, is_pure_reducer=proj.is_pure_reducer)
        )
        return acc

    select_exprs, out_schema = projections.iter().fold(
        (key_glots, key_schema.items().iter().collect(Dict)),
        _acc,
    )
    ast = exp.select(*select_exprs).from_(
        ast.subquery(Tables.SRC, copy=False), copy=False
    )
    if drop_null_keys:
        null_cond = keys.iter().map(lambda k: k.is_not_null()).reduce(Expr.and_).inner
        ast = ast.where(null_cond, copy=False)
    key_glots = keys.iter().map(lambda k: k.inner)
    return ast.group_by(*_group_by_clause(strategy, key_glots), copy=False), out_schema


def _split_by_key(
    key_names: Set[str],
    acc: tuple[Schema, Schema],
    name: str,
    dtype: exp.DataType,
) -> tuple[Schema, Schema]:
    key_s, non_key_s = acc
    match name in key_names:
        case True:
            _ = key_s.insert(name, dtype)
        case _:
            _ = non_key_s.insert(name, dtype)
    return acc


def _to_array(base: exp.DataType, *, is_pure_reducer: bool) -> exp.DataType:
    if is_pure_reducer:
        return base
    return exp.DataType(this=exp.DataType.Type.ARRAY, expressions=[base], nested=True)


def _group_by_clause(
    strategy: GroupByClause | None, key_glots: Iter[exp.Expr]
) -> Iterable[exp.Expr]:
    match strategy:
        case "CUBE":
            return Iter.once(exp.Cube(expressions=list(key_glots)))
        case "ROLLUP":
            return Iter.once(exp.Rollup(expressions=list(key_glots)))
        case None:
            return key_glots


def _exploded(expr: Expr, *, is_distinct: bool) -> Expr:
    if is_distinct:
        return expr.implode().list.distinct()
    return expr.implode()
