import pyochain as pc
import pytest

import pql


def _check_slots(obj: object) -> pc.Result[None, str]:
    try:
        _ = obj.__dict__
        msg = f"{obj.__class__.__name__} has __dict__, but should have __slots__"
        return pc.Err(msg)
    except AttributeError:
        return pc.Ok(None)


_OBJS = [
    pql.col(""),
    pql.sql.col(""),
    pql.sql.Relation(pql.sql.into_relation({"a": [1]})),
    pql.LazyFrame({"a": [1]}),
    pql.when(1),
    pql.when(1).then(2),
    pql.when(1).then(2).when(3).then(4),
    pql.sql.when(1),
    pql.sql.when(1).then(2),
    pql.sql.when(1).then(2).when(3).then(4),
    pql.selectors.all(),
    pql.Schema({}),
    *pql._datatypes.NON_NESTED_MAP.values(),  # pyright: ignore[reportPrivateUsage]
]


@pytest.mark.parametrize("obj", _OBJS)
def test_slots(obj: object) -> None:
    assert _check_slots(obj).is_ok()
