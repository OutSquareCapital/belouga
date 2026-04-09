from pyochain import Vec

import pql

_LF = pql.LazyFrame({"x": [1]})


def _slct(*exprs: pql.Expr) -> Vec[str]:
    return _LF.select(*exprs).columns


def test_alias_mutability() -> None:
    original = pql.col("x")
    prefixed = original.name.prefix("pre_")
    aliased = prefixed.alias("renamed")

    assert _slct(original).first() == "x"
    assert _slct(prefixed).first() == "pre_x"
    assert _slct(aliased).first() == "renamed"
