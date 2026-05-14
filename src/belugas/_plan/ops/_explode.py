from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from pyochain import Dict, Iter
from sqlglot import exp

from ..._core import Tables
from ..._funcs import col, lit, unnest
from .._resolve import resolve_all

if TYPE_CHECKING:
    from pyochain.traits import PyoIterable

    from ..._expr import Expr
    from ...typing import IntoExprColumn, Schema, TryIter


def explode(
    src_ast: exp.Select | exp.Union,
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> exp.Union:

    to_explode = (
        resolve_all(schema, columns, more_columns, {})
        .iter()
        .enumerate()
        .map_star(lambda idx, r: (r.name, IndexedExpr(idx + 1, col(r.name))))
        .collect(Dict)
    )
    is_single_explode = to_explode.length() == 1
    target = (
        to_explode
        .values()
        .iter()
        .into(_get_target, is_single_explode=is_single_explode)
    )

    cond = target.is_not_null().and_(target.list.length().gt(0))
    transformer = partial(
        transform, schema, to_explode, target, is_single=is_single_explode
    )
    rhs = (
        exp
        .select(*transformer(nested=False))
        .from_(src_ast.subquery(Tables.SRC, copy=False), copy=False)
        .where(cond.not_().inner, copy=False)
    )
    return (
        exp
        .select(*transformer(nested=True))
        .from_(src_ast.subquery(Tables.SRC, copy=False), copy=False)
        .where(cond.inner, copy=False)
        .pipe(exp.union, rhs, copy=False)
    )


def _get_target(exprs: Iter[IndexedExpr], *, is_single_explode: bool) -> Expr:

    first_expr = exprs.next().unwrap().expr
    if is_single_explode:
        return first_expr
    return first_expr.list.zip(*exprs.map(lambda ie: ie.expr), lit(1).eq(1))


class IndexedExpr(NamedTuple):
    idx: int
    expr: Expr


def transform(
    columns: PyoIterable[str],
    to_explode: Dict[str, IndexedExpr],
    target: Expr,
    *,
    is_single: bool,
    nested: bool,
) -> Iter[exp.Expr]:

    def _project_col(name: str, replace: Expr) -> Expr:
        match (nested, name in to_explode):
            case (True, True):
                if is_single:
                    return replace.alias(name)
                field = to_explode.get_item(name).unwrap().idx
                return replace.struct.extract(field).alias(name)
            case (False, True):
                return lit(None).alias(name)
            case _:
                return col(name)

    replace = unnest(target) if nested else lit(None)
    return columns.iter().map(lambda name: _project_col(name, replace).inner)
