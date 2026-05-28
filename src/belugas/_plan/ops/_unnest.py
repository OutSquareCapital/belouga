from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Set, Vec
from sqlglot import exp

from ... import datatypes as dt
from ..._funcs import col, unnest as unnest_fn
from ...utils import try_iter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...typing import IntoExprColumn, Schema, TryIter


def unnest(
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> tuple[Iter[exp.Expr], Schema]:

    def _project(name: str, raw: exp.DataType) -> None:
        def _project_field(field_name: str, field_dtype: exp.DataType) -> exp.Expr:
            _ = new_schema.insert(field_name, field_dtype)
            return col(name).struct.field(name=field_name).alias(field_name).inner

        match name in targets, raw.this:  # pyright: ignore[reportAny]
            case (True, exp.DType.STRUCT):
                return (
                    dt
                    .extract_struct_fields(raw)
                    .map_star(_project_field)
                    .into(exprs.extend)
                )
            case (True, exp.DType.ARRAY | exp.DType.LIST):
                _ = new_schema.insert(name, raw)
                return exprs.append(unnest_fn(col(name)).alias(name).inner)
            case _:
                _ = new_schema.insert(name, raw)
                return exprs.append(exp.column(name))

    targets = try_iter(columns).chain(more_columns).collect(Set)

    exprs = Vec[exp.Expr](())
    new_schema = Dict[str, exp.DataType](())

    schema.items().iter().for_each_star(_project)
    return exprs.iter(), new_schema
