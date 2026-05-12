from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyochain import NONE, Option, Seq
from sqlglot import exp

from .._core import Tables

if TYPE_CHECKING:
    from ..typing import Schema


@dataclass(slots=True)
class ProjectionSpec:
    Exprs: Seq[exp.Expr]
    distinct: bool = False
    windowed_source: bool = False


@dataclass(slots=True)
class DeferredDelta:
    schema: Option[Schema] = field(default_factory=lambda: NONE)
    projection: Option[ProjectionSpec] = field(default_factory=lambda: NONE)
    where: Option[exp.Expr] = field(default_factory=lambda: NONE)
    order_by: Seq[exp.Expr] = field(default_factory=Seq[exp.Expr].new)
    limit: Option[exp.Expr] = field(default_factory=lambda: NONE)
    offset: Option[exp.Expr] = field(default_factory=lambda: NONE)


def as_relation(
    source: exp.Selectable | exp.Table,
    alias: str = Tables.SRC.name,
    *,
    copy_source: bool = False,
) -> exp.Table | exp.Subquery:
    match source:
        case exp.Table() as table:
            return exp.Table(
                this=exp.to_identifier(table.name),
                db=table.args.get("db"),
                catalog=table.args.get("catalog"),
                alias=exp.TableAlias(this=exp.to_identifier(alias)),
                pivots=table.args.get("pivots"),
            )
        case exp.Subquery(this=exp.Selectable() as inner):
            return exp.Subquery(
                this=inner.copy() if copy_source else inner,
                alias=exp.TableAlias(this=exp.to_identifier(alias)),
                pivots=source.args.get("pivots"),
            )
        case _:
            return exp.Subquery(
                this=source.copy() if copy_source else source,
                alias=exp.TableAlias(this=exp.to_identifier(alias)),
            )
