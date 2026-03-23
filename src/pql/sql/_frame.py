from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, ClassVar, Literal, Self

import pyochain as pc
from sqlglot import exp

from ._code_gen import Relation
from ._creation import from_query, into_relation
from ._funcs import into_expr
from .utils import TryIter, try_iter

if TYPE_CHECKING:
    from duckdb import DuckDBPyRelation
    from pyochain.traits import PyoIterable

    from .typing import IntoExpr, IntoExprColumn, IntoRel, Orientation, PythonLiteral


class Frame(Relation):
    _inner: DuckDBPyRelation
    __slots__: ClassVar[Iterable[str]] = ("_inner",)

    def __init__(self, data: IntoRel, orient: Orientation = "col") -> None:
        self._inner = into_relation(data, orient=orient)

    def _from_sql_expr(self, expr: exp.Expr, **kwargs: IntoRel) -> Self:
        qry = from_query(expr.sql(dialect="duckdb"), **kwargs)
        return self.__class__(qry)

    def join_asof(
        self,
        other: IntoRel,
        condition: IntoExpr,
        select_cols: Iterable[IntoExprColumn],
        how: Literal["left", "inner"],
    ) -> Self:
        join_type = "asof left" if how == "left" else "asof"
        qry = (
            exp.select(*pc.Iter(select_cols).map(into_expr).map(lambda c: c.inner()))  # pyright: ignore[reportUnknownMemberType]
            .from_("lhs")
            .join("rhs", on=into_expr(condition).inner(), join_type=join_type)
        )

        return self._from_sql_expr(qry, lhs=self.inner(), rhs=other)

    def pivot(
        self,
        on: TryIter[str],
        using: TryIter[IntoExprColumn],
        group_by: TryIter[str],
        in_values: pc.Option[Sequence[PythonLiteral]],
        order_by: TryIter[str],
    ) -> Self:

        def _on_exprs(on_iter: pc.Seq[str]) -> PyoIterable[exp.Expr] | PyoIterable[str]:
            match in_values:
                case pc.Some(vals):
                    converted = pc.Iter(vals).map(exp.convert).collect()
                    expr = exp.In(this=on_iter.first(), expressions=converted)
                    return pc.Iter.once(expr)
                case _:
                    return on_iter

        def _group() -> exp.Group | None:
            group = try_iter(group_by).then(
                lambda cols: exp.Group(expressions=cols.collect(list))
            )
            return group.unwrap() if group.is_some() else None

        def _pivot() -> exp.Expr:
            return exp.Pivot(
                this=exp.to_table("rel"),  # pyright: ignore[reportUnknownMemberType]
                expressions=try_iter(on).collect().into(_on_exprs),
                using=try_iter(using).map(into_expr).map(lambda c: c.inner()),
                group=_group(),
            )

        def _select_ordered(cols: Iterable[str]) -> exp.Expr:
            return (
                exp.select("*")  # pyright: ignore[reportUnknownMemberType]
                .from_(exp.Subquery(this=_pivot()))
                .order_by(*cols)
            )

        qry = try_iter(order_by).collect().then(_select_ordered).unwrap_or_else(_pivot)

        return self._from_sql_expr(qry, rel=self.inner())

    def unpivot(
        self,
        on: TryIter[str],
        index: TryIter[str],
        variable_name: str,
        value_name: str,
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
            return exp.select(*index_cols, variable_name, value_name).from_(sub_qry)  # pyright: ignore[reportUnknownMemberType]

        qry = (
            try_iter(order_by)
            .then(lambda cols: _select().order_by(*cols))  # pyright: ignore[reportUnknownMemberType]
            .unwrap_or_else(_select)
        )

        return self._from_sql_expr(qry, rel=self.inner())
