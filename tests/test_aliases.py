import polars as pl
import pytest
from pyochain.abc import PyoIterable

import belugas as bl

from ._utils import ExprPair

_LF = bl.LazyFrame({"x": [1], "y": [4]})


bl_x = bl.col("x")
bl_y = bl.col("y")
pl_x = pl.col("x")
pl_y = pl.col("y")


def test_alias_mutability() -> None:
    prefixed = bl_x.name.prefix("pre_")
    aliased = prefixed.alias("renamed")

    assert _slct(bl_x).first() == "x"
    assert _slct(prefixed).first() == "pre_x"
    assert _slct(aliased).first() == "renamed"


@pytest.mark.parametrize(
    "exprs",
    [
        ExprPair(
            bl.when(bl_x.gt(0)).then(bl_y).otherwise(bl_x),
            pl.when(pl_x.gt(0)).then(pl_y).otherwise(pl_x),
        ),
        ExprPair(
            bl.when(bl_x.gt(0)).then(1).otherwise(bl_x),
            pl.when(pl_x.gt(0)).then(1).otherwise(pl_x),
        ),
        ExprPair(
            bl.when(bl_x.gt(0)).then(bl_y.mul(2)).otherwise(bl_y),
            pl.when(pl_x.gt(0)).then(pl_y.mul(2)).otherwise(pl_y),
        ),
    ],
    ids=["then_y", "then_lit", "then_mul"],
)
def test_when_alias(exprs: ExprPair) -> None:
    pl_cols = _LF.collect().select(exprs.pl_expr).columns
    bl_cols = _slct(exprs.bl_expr).into(list)
    assert bl_cols == pl_cols


def test_when_alias_chained_then_lit() -> None:
    pl_cols = (
        _LF
        .collect()
        .select(pl.when(pl_x.gt(0)).then(1).when(pl_y.gt(0)).then(pl_y).otherwise(pl_x))
        .columns
    )
    bl_cols = _slct(
        bl.when(bl_x.gt(0)).then(1).when(bl_y.gt(0)).then(bl_y).otherwise(bl_x)
    ).into(list)
    assert bl_cols == pl_cols


def _slct(*exprs: bl.Expr) -> PyoIterable[str]:
    return _LF.select(*exprs).columns
