from pyochain import Err, Ok, Option, Result, Some
from sqlglot import exp

from ._meta import Tables

MAX_I64 = 9_223_372_036_854_775_807


def slice(lf_length: Option[int], offset: int) -> Result[exp.Select, ValueError]:
    from .._expr import Expr
    from .._funcs import col, lit

    match (lf_length, offset):
        case (Some(length), _) if length < 0:
            msg = f"negative slice lengths ({length}) are invalid for LazyFrame"
            return Err(ValueError(msg))
        case (len_val, offset) if offset >= 0:
            return Ok(
                exp
                .select(exp.Star())
                .from_(Tables.SRC, copy=False)
                .limit(exp.convert(len_val.unwrap_or(MAX_I64)), copy=False)
                .offset(exp.convert(offset), copy=False)
            )
        case (Some(0), _):
            return Ok(
                exp
                .select(exp.Star())
                .from_(Tables.SRC, copy=False)
                .limit(exp.convert(0), copy=False)
            )
        case (Some(length), offset):
            slice_len_expr = col("slice_len")
            stats = exp.select(lit(1).count().alias("slice_len").inner).from_(
                Tables.SRC, copy=False
            )
            start_expr = slice_len_expr.add(offset).greatest(0).inner
            return Ok(
                exp
                .select(exp.Star())
                .from_(Tables.SRC, copy=False)
                .with_("stats", as_=stats, copy=False)
                .limit(
                    exp
                    .select(
                        slice_len_expr
                        .add(offset)
                        .add(length)
                        .least(slice_len_expr)
                        .sub(start_expr)
                        .greatest(0)
                        .inner
                    )
                    .from_(Tables.STATS, copy=False)
                    .subquery(copy=False)
                )
                .offset(
                    exp
                    .select(start_expr)
                    .from_(Tables.STATS, copy=False)
                    .subquery(copy=False)
                )
            )
        case (_, offset):
            return Ok(
                exp
                .select(exp.Star())
                .from_(Tables.SRC, copy=False)
                .offset(
                    exp
                    .select(lit(1).count().inner)
                    .from_(Tables.SRC, copy=False)
                    .subquery(copy=False)
                    .pipe(Expr)
                    .add(offset)
                    .greatest(0)
                    .inner
                )
            )
