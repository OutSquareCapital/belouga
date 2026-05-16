from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, NamedTuple, override

import polars as pl
from polars.testing import assert_frame_equal
from pyochain import Iter, Seq
from pyochain.traits import PyoIterable

import belugas as bl

from ._data import sample_bl, sample_lf

type PlFn = Callable[..., pl.Expr]
type PqlFn = Callable[..., bl.Expr]


class Fns(NamedTuple):
    """Tuple used for parametrized tests."""

    bl_fn: PqlFn
    pl_fn: PlFn

    def call(self, *args: object, **kwargs: object) -> tuple[bl.Expr, pl.Expr]:
        return self.bl_fn(*args, **kwargs), self.pl_fn(*args, **kwargs)


def into_ids(fns: Seq[tuple[Callable[..., Any], Callable[..., Any]]]) -> Iter[str]:
    return fns.iter().map_star(lambda f1, _f2: f1.__name__)


class ExprPair(NamedTuple):
    bl_expr: bl.Expr
    pl_expr: pl.Expr


@dataclass(slots=True, init=False)
class FnsCat(PyoIterable[Fns]):
    fns: Seq[Fns]

    def __init__(self, *fns: tuple[PqlFn, PlFn]) -> None:
        self.fns = Iter(fns).map_star(Fns).collect()

    @override
    def __iter__(self) -> Iterator[Fns]:
        return self.fns.iter()

    def into_ids(self) -> Iter[str]:
        return self.fns.iter().map(lambda x: x.bl_fn.__name__)


def assert_eq(
    bl_expr: bl.Expr, polars_expr: pl.Expr, *, with_cols: bool = True
) -> None:
    _assert(sample_lf().select(polars_expr), sample_bl().select(bl_expr).collect())
    if with_cols:
        _assert(
            sample_lf().with_columns(polars_expr),
            sample_bl().with_columns(bl_expr).collect(),
        )


def assert_lf_eq(polars_lf: pl.LazyFrame, bl_lf: bl.LazyFrame) -> None:
    try:
        _assert(polars_lf, bl_lf.collect())
    except AssertionError as e:
        sql = bl_lf.query.sql(pretty=True)
        ast = bl_lf.query.logical()
        msg = f"""
Not equal error!
----SQL----
{sql}
----AST----
{ast}
----Error----
{e}
"""
        raise AssertionError(msg) from None


def _assert(
    left: pl.DataFrame | pl.LazyFrame, right: pl.DataFrame | pl.LazyFrame
) -> None:
    return assert_frame_equal(
        left.lazy().collect(),
        right.lazy().collect(),
        check_dtypes=False,
        check_row_order=False,
    )
