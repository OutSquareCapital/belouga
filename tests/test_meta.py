from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyochain import Iter, Seq

import belugas as bl
from belugas import meta

if TYPE_CHECKING:
    from collections.abc import Callable


def _get_fn(name: str) -> Callable[..., bl.LazyFrame]:
    return getattr(meta, name)  # pyright: ignore[reportAny]


_META_FNS: Seq[Callable[[], bl.LazyFrame]] = (
    Iter(dir(meta))
    .map(_get_fn)
    .filter(lambda fn: callable(fn) and fn.__name__ != "LazyFrame")
    .collect()
)


@pytest.mark.parametrize("fns", _META_FNS)
def test_meta_fns(fns: Callable[..., bl.LazyFrame]) -> None:
    assert isinstance(fns(), bl.LazyFrame)
