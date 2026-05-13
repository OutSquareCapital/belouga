from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from pyochain import Err, Iter, Ok, Result, Seq

from ..._expr import Expr
from ...utils import try_iter

if TYPE_CHECKING:
    from sqlglot import exp

    from ...typing import IntoExpr, TryIter, TrySeq


def sort(
    by: TryIter[IntoExpr],
    more_by: Iterable[IntoExpr],
    descending: TrySeq[bool],
    nulls_last: TrySeq[bool],
) -> Iter[exp.Expr]:

    return (
        try_iter(by)
        .chain(more_by)
        .map(lambda v: Expr.new(v, as_col=True))
        .collect()
        .into(
            lambda sort_exprs: sort_exprs.iter().zip(
                check_by_arg(sort_exprs, "descending", arg=descending).unwrap(),
                check_by_arg(sort_exprs, "nulls_last", arg=nulls_last).unwrap(),
            )
        )
        .map_star(
            lambda expr, desc, nls: expr.order_by(descending=desc, nulls_last=nls).inner
        )
    )


def check_by_arg(
    compared: Seq[Expr],
    name: str,
    arg: TrySeq[bool],
) -> Result[Iter[bool], ValueError]:
    length = compared.length()
    match arg:
        case Sequence():
            len_arg = len(arg)
            if len_arg == length:
                return Ok(try_iter(arg))
            msg = f"the length of `{name}` ({len_arg}) does not match the length of `by` ({length})"
            return Err(ValueError(msg))

        case _:
            return Ok(try_iter(arg).cycle().take(length))
