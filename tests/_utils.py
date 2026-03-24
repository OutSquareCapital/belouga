from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import NamedTuple, override

import narwhals as nw
import polars as pl
import pyochain as pc
from polars.testing import assert_frame_equal
from pyochain.traits import PyoIterable

import pql

from ._data import sample_df

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


def _capture[T](
    func: Callable[[], T],
) -> pc.Result[T, tuple[Exception, pc.Option[TracebackType]]]:
    try:
        return pc.Ok(func())
    except Exception as exc:  # noqa: BLE001
        return pc.Err((exc, pc.Option(exc.__traceback__)))


def _with_tb(error: Exception, traceback: pc.Option[TracebackType]) -> Exception:
    return traceback.map_or(error, error.with_traceback)


def _eval_both[T, U](
    pql_func: Callable[[], T], polars_func: Callable[[], U]
) -> tuple[T, U]:
    pql_res = _capture(pql_func)
    polars_res = _capture(polars_func)

    match (pql_res, polars_res):
        case (pc.Err((exc, tb)), pc.Ok(_)):
            exc.add_note("PQL evaluation failed while the reference side succeeded.")
            raise _with_tb(exc, tb)
        case (pc.Ok(_), pc.Err((exc, tb))):
            exc.add_note("Reference evaluation failed while the PQL side succeeded.")
            raise _with_tb(exc, tb)
        case (pc.Err((pql_error, pql_tb)), pc.Err((polars_error, polars_tb))):
            pql_error.add_note("PQL evaluation failed.")
            polars_error.add_note("Reference evaluation failed.")
            msg = "Both PQL and reference evaluation failed."
            raise ExceptionGroup(
                msg, (_with_tb(pql_error, pql_tb), _with_tb(polars_error, polars_tb))
            )
        case _:
            return pql_res.unwrap(), polars_res.unwrap()


def assert_eq(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: nw.Expr | Iterable[nw.Expr]
) -> None:
    pql_df, polars_df = _eval_both(
        lambda: pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        lambda: sample_df().lazy().select(polars_exprs).to_native().pl(),
    )
    assert_frame_equal(
        pql_df,
        polars_df,
        check_dtypes=False,
        check_row_order=False,
    )


def assert_eq_pl(
    pql_exprs: pql.Expr | Iterable[pql.Expr], polars_exprs: pl.Expr | Iterable[pl.Expr]
) -> None:
    pql_df, polars_df = _eval_both(
        lambda: pql.LazyFrame(sample_df().to_native()).select(pql_exprs).collect(),
        lambda: sample_df().to_native().pl(lazy=True).select(polars_exprs).collect(),
    )
    assert_frame_equal(
        pql_df,
        polars_df,
        check_dtypes=False,
        check_row_order=False,
    )


def assert_lf_eq_pl(pql_lf: pql.LazyFrame, polars_lf: pl.LazyFrame) -> None:
    pql_df, polars_df = _eval_both(pql_lf.collect, polars_lf.collect)
    assert_frame_equal(pql_df, polars_df, check_dtypes=False, check_row_order=False)


def on_simple_fn(pql_expr: object, pl_expr: object, fn_name: str) -> None:
    assert_eq_pl(getattr(pql_expr, fn_name)(), getattr(pl_expr, fn_name)())  # pyright: ignore[reportAny]
