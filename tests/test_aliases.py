import polars as pl
from pyochain import Vec

import pql

_LF = pql.LazyFrame({"x": [1], "y": ["hello"]})


def _slct(*exprs: pql.Expr) -> Vec[str]:
    return _LF.select(*exprs).columns


def test_alias_mutability() -> None:
    original = pql.col("x")
    prefixed = original.name.prefix("pre_")
    aliased = prefixed.alias("renamed")

    assert _slct(original).first() == "x"
    assert _slct(prefixed).first() == "pre_x"
    assert _slct(aliased).first() == "renamed"


def test_when_alias() -> None:
    pql_x = pql.col("x")
    pql_y = pql.col("y")
    pl_x = pl.col("x")
    pl_y = pl.col("y")
    assert (
        _slct(
            pql.when(pql_x.gt(0)).then(pql_y).otherwise(pql_x),
            pql.when(pql_x.gt(0)).then(pql.lit(1)).otherwise(pql_x),
            pql.when(pql_x.gt(0)).then(pql_y.str.to_uppercase()).otherwise(pql_y),
        ).into(list)
        == _LF
        .lazy()
        .select(
            pl.when(pl_x.gt(0)).then(pl_y).otherwise(pl_x),
            pl.when(pl_x.gt(0)).then(pl.lit(1)).otherwise(pl_x),
            pl.when(pl_x.gt(0)).then(pl_y.str.to_uppercase()).otherwise(pl_y),
        )
        .columns
    )
