from collections.abc import Callable

import pyochain as pc
import pytest

import pql
from pql import meta


def _get_fn(name: str) -> Callable[..., pql.LazyFrame]:
    return getattr(meta, name)  # pyright: ignore[reportAny]


_META_FNS: pc.Seq[Callable[[], pql.LazyFrame]] = (
    pc
    .Iter(dir(meta))
    .map(_get_fn)
    .filter(lambda fn: callable(fn) and fn.__name__ != "LazyFrame")
    .collect()
)


@pytest.mark.parametrize("fns", _META_FNS)
def test_meta_fns(fns: Callable[..., pql.LazyFrame]) -> None:
    assert isinstance(fns(), pql.LazyFrame)
