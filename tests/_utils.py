from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import NamedTuple, override

import polars as pl
import pyochain as pc
from polars.testing import assert_frame_equal
from pyochain.traits import PyoIterable
from rich.traceback import install

import pql

from ._data import sample_lf, sample_pql

_ = install(show_locals=True)
type PlFn = Callable[..., pl.Expr]
type PqlFn = Callable[..., pql.Expr]


class Fns(NamedTuple):
    """Tuple used for parametrized tests."""

    pql_fn: PqlFn
    pl_fn: PlFn

    def call(self, *args: object, **kwargs: object) -> tuple[pql.Expr, pl.Expr]:
        return self.pql_fn(*args, **kwargs), self.pl_fn(*args, **kwargs)


class ExprPair(NamedTuple):
    pql_expr: pql.Expr
    pl_expr: pl.Expr


@dataclass(slots=True, init=False)
class FnsCat(PyoIterable[Fns]):
    fns: pc.Seq[Fns]

    def __init__(self, *fns: tuple[PqlFn, PlFn]) -> None:
        self.fns = pc.Iter(fns).map_star(Fns).collect()

    @override
    def __iter__(self) -> Iterator[Fns]:
        return self.fns.iter()

    def into_ids(self) -> pc.Iter[str]:
        return self.fns.iter().map(lambda x: x.pql_fn.__name__)


def _assert(
    left: pl.DataFrame | pl.LazyFrame, right: pl.DataFrame | pl.LazyFrame
) -> None:
    return assert_frame_equal(
        left.lazy().collect(),
        right.lazy().collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq(
    pql_expr: pql.Expr, polars_expr: pl.Expr, *, with_cols: bool = True
) -> None:
    _assert(sample_lf().select(polars_expr), sample_pql().select(pql_expr).collect())
    if with_cols:
        _assert(
            sample_lf().with_columns(polars_expr),
            sample_pql().with_columns(pql_expr).collect(),
        )


def assert_lf_eq(polars_lf: pl.LazyFrame, pql_lf: pql.LazyFrame) -> None:
    _assert(polars_lf, pql_lf.collect())


def on_simple_fn(pql_expr: object, pl_expr: object, fn_name: str) -> None:
    assert_eq(getattr(pql_expr, fn_name)(), getattr(pl_expr, fn_name)())  # pyright: ignore[reportAny]
