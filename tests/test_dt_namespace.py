from collections.abc import Callable

import polars as pl
import pytest
from pyochain import Seq

import pql
import pql.sql.typing as t

from ._utils import assert_eq

dt = "dt"
pql_dt = pql.col(dt).dt
pl_dt = pl.col(dt).dt


_SIMPLE_FNS = Seq((
    (pql_dt.microsecond, pl_dt.microsecond),
    (pql_dt.nanosecond, pl_dt.nanosecond),
    (pql_dt.millisecond, pl_dt.millisecond),
    (pql_dt.second, pl_dt.second),
    (pql_dt.minute, pl_dt.minute),
    (pql_dt.hour, pl_dt.hour),
    (pql_dt.day, pl_dt.day),
    (pql_dt.weekday, pl_dt.weekday),
    (pql_dt.ordinal_day, pl_dt.ordinal_day),
    (pql_dt.week, pl_dt.week),
    (pql_dt.month, pl_dt.month),
    (pql_dt.quarter, pl_dt.quarter),
    (pql_dt.year, pl_dt.year),
    (pql_dt.iso_year, pl_dt.iso_year),
    (pql_dt.century, pl_dt.century),
    (pql_dt.millennium, pl_dt.millennium),
    (pql_dt.date, pl_dt.date),
    (pql_dt.time, pl_dt.time),
    (pql_dt.month_start, pl_dt.month_start),
    (pql_dt.month_end, pl_dt.month_end),
))


@pytest.mark.parametrize(
    "fn", _SIMPLE_FNS, ids=_SIMPLE_FNS.iter().map_star(lambda f1, _f2: f1.__name__)
)
def test_simple_fns(fn: tuple[Callable[[], pql.Expr], Callable[[], pl.Expr]]) -> None:
    assert_eq(fn[0](), fn[1]())


@pytest.mark.parametrize("unit", t.EpochTimeUnit.__args__)
def test_epoch(unit: t.EpochTimeUnit) -> None:
    assert_eq(pql_dt.epoch_by(unit), pl_dt.epoch(unit))


@pytest.mark.parametrize("unit", t.TimeUnit.__args__)
def test_timestamp(unit: t.TimeUnit) -> None:
    assert_eq(pql_dt.timestamp(unit), pl_dt.timestamp(unit))


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%H:%M"])
def test_date_time_boundaries(fmt: str) -> None:
    assert_eq(pql_dt.to_string(fmt), pl_dt.to_string(fmt))


def test_truncate() -> None:
    assert_eq(pql_dt.truncate("month"), pl_dt.truncate("1mo"))


def test_round() -> None:
    assert_eq(pql_dt.round("month"), pl_dt.round("1mo"))


def test_offset_by() -> None:
    assert_eq(pql_dt.offset_by("1 day"), pl_dt.offset_by("1d"))
