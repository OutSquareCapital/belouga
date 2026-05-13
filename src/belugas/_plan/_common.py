from __future__ import annotations

from sqlglot import exp

from .._core import Tables


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
