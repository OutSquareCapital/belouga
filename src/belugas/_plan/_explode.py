from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Seq
from sqlglot import exp

from ._meta import Tables, resolve_all

if TYPE_CHECKING:
    from pyochain.traits import PyoIterable

    from .._expr import Expr
    from ..typing import IntoExprColumn, Schema
    from ..utils import TryIter


def explode(
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> exp.Union:
    from .._funcs import col, lit

    to_explode_names = (
        resolve_all(schema, columns, more_columns, {})
        .iter()
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

    cond = target.is_not_null().and_(target.list.length().gt(0))

    rhs = (
        exp
        .select(
            *_proj(
                schema.keys(),
                zipped_index,
                to_explode_names,
                target,
                nested=False,
                is_single_explode=is_single_explode,
            )
        )
        .from_(Tables.EXPLODE_SRC, copy=False)
        .where(cond.not_().inner, copy=False)
    )
    return (
        exp
        .select(
            *_proj(
                schema.keys(),
                zipped_index,
                to_explode_names,
                target,
                nested=True,
                is_single_explode=is_single_explode,
            )
        )
        .from_(Tables.EXPLODE_SRC, copy=False)
        .where(cond.inner, copy=False)
        .pipe(exp.union, rhs, copy=False)
    )


def _proj(  # noqa: PLR0913
    columns: PyoIterable[str],
    zipped_index: Dict[str, int],
    to_explode_names: Seq[str],
    target: Expr,
    *,
    nested: bool,
    is_single_explode: bool,
) -> Iter[exp.Expr]:
    from .._funcs import col, lit, unnest

    def _project_col(name: str, replace: Expr) -> Expr:
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

    replace = unnest(target) if nested else lit(None)
    return columns.iter().map(lambda name: _project_col(name, replace).inner)
