import operator
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import NamedTuple, override

import narwhals as nw
import polars as pl
import pyochain as pc
from polars.testing import assert_frame_equal
from pyochain.traits import PyoIterable
from rich.traceback import install

import pql

from ._data import sample_df

_ = install(show_locals=True)
type PlFn = Callable[..., pl.Expr]
type PqlFn = Callable[..., pql.Expr]


class Fns(NamedTuple):
    """Tuple used for parametrized tests."""

    pql_fn: PqlFn
    pl_fn: PlFn

    def call(self, *args: object, **kwargs: object) -> tuple[pql.Expr, pl.Expr]:
        return self.pql_fn(*args, **kwargs), self.pl_fn(*args, **kwargs)


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


def _assert_schema(lf: pql.LazyFrame) -> pql.LazyFrame:
    other = pql.Schema.from_frame(lf.inner())
    left_keys = lf.schema.keys().into(list)
    right_keys = other.keys().into(list)
    assert left_keys == right_keys, f"{left_keys} != {right_keys}"
    incorrect_dtype = (
        lf.schema
        .values()
        .iter()
        .zip(other.values())
        .map_star(lambda left, right: left.is_(right))
        .find(operator.not_)
    )
    assert incorrect_dtype.is_none(), f"Incorrect dtype: {incorrect_dtype.unwrap()}"
    return lf


def _run_pql(pql_exprs: pql.Expr | Iterable[pql.Expr]) -> pql.LazyFrame:
    return pql.LazyFrame(sample_df().to_native()).select(pql_exprs).pipe(_assert_schema)


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    assert_frame_equal(
        _run_pql(pql_exprs).collect(),
        sample_df().lazy().select(polars_exprs).to_native().pl(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    assert_frame_equal(
        _run_pql(pql_exprs).collect(),
        sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def assert_lf_eq_pl(pql_lf: pql.LazyFrame, polars_lf: pl.LazyFrame) -> None:
    assert_frame_equal(
        pql_lf.pipe(_assert_schema).collect(),
        polars_lf.collect(),
        check_dtypes=False,
        check_row_order=False,
    )


def on_simple_fn(pql_expr: object, pl_expr: object, fn_name: str) -> None:
    assert_eq_pl(getattr(pql_expr, fn_name)(), getattr(pl_expr, fn_name)())  # pyright: ignore[reportAny]
