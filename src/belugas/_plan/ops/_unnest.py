from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pyochain import Dict, Iter, Set
from sqlglot import exp

from ... import datatypes as dt
from ..._core import Tables
from ..._funcs import col, unnest as unnest_fn
from ...utils import try_iter

if TYPE_CHECKING:
    from ...typing import IntoExprColumn, Schema, TryIter


def unnest(
    schema: Schema,
    columns: TryIter[IntoExprColumn],
    more_columns: Iterable[IntoExprColumn],
) -> tuple[exp.Select, Schema]:

    targets = try_iter(columns).chain(more_columns).collect(Set)

    def _proj(name: str) -> Iter[exp.Expr]:
        dtype = schema.get_item(name).map(dt.DataType.from_sql).unwrap()
        match name in targets, dtype:
            case (True, dt.Struct()):
                return dtype.fields.iter().map(
                    lambda f: col(name).struct.field(name=f).alias(f).inner
                )
            case (True, dt.List() | dt.Array()):
                return Iter.once(unnest_fn(col(name)).alias(name).inner)
            case _:
                return Iter.once(col(name).inner)

    def _schema_proj(name: str, raw: exp.DataType) -> Iter[tuple[str, exp.DataType]]:
        match name in targets, raw.this:  # pyright: ignore[reportAny]
            case (True, exp.DType.STRUCT):
                exprs: list[exp.Expr] = raw.expressions
                return (
                    Iter(exprs)
                    .map(
                        lambda col_def: (
                            col_def.this.this,  # pyright: ignore[reportAny]
                            dt.DataType.from_sql(col_def.kind),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                        )
                    )
                    .map_star(lambda f, fd: (f, fd.raw))  # pyright: ignore[reportAny]
                )
            case _:
                return Iter.once((name, raw))

    new_schema = schema.items().iter().map_star(_schema_proj).flatten().collect(Dict)
    exprs = schema.iter().flat_map(_proj)
    return exp.select(*exprs).from_(Tables.SRC, copy=False), new_schema
