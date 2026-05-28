from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from pyochain import Seq

import belugas as bl
import belugas.typing as t

from ._utils import assert_eq, into_ids

if TYPE_CHECKING:
    from collections.abc import Callable

dt = "dt"
bl_dt = bl.col(dt).dt
pl_dt = pl.col(dt).dt


_SIMPLE_FNS = Seq((
    (bl_dt.second, pl_dt.second),
    (bl_dt.minute, pl_dt.minute),
    (bl_dt.hour, pl_dt.hour),
    (bl_dt.day, pl_dt.day),
    (bl_dt.weekday, pl_dt.weekday),
    (bl_dt.ordinal_day, pl_dt.ordinal_day),
    (bl_dt.week, pl_dt.week),
    (bl_dt.month, pl_dt.month),
    (bl_dt.quarter, pl_dt.quarter),
    (bl_dt.year, pl_dt.year),
    (bl_dt.iso_year, pl_dt.iso_year),
    (bl_dt.century, pl_dt.century),
    (bl_dt.millennium, pl_dt.millennium),
    (bl_dt.date, pl_dt.date),
    (bl_dt.time, pl_dt.time),
    (bl_dt.month_start, pl_dt.month_start),
    (bl_dt.month_end, pl_dt.month_end),
))


@pytest.mark.parametrize("fn", _SIMPLE_FNS, ids=_SIMPLE_FNS.into(into_ids))
def test_simple_fns(fn: tuple[Callable[[], bl.Expr], Callable[[], pl.Expr]]) -> None:
    assert_eq(fn[0](), fn[1]())


_DUCKDB_SUBSECOND_FNS = Seq((
    (
        bl_dt.microsecond(),
        pl_dt.second().cast(pl.Int64).mul(1_000_000).add(pl_dt.microsecond()),
    ),
    (
        bl_dt.nanosecond(),
        pl_dt
        .second()
        .cast(pl.Int64)
        .mul(1_000_000_000)
        .add(pl_dt.nanosecond().cast(pl.Int64)),
    ),
    (
        bl_dt.millisecond(),
        pl_dt.second().cast(pl.Int64).mul(1_000).add(pl_dt.millisecond()),
    ),
))


@pytest.mark.parametrize(
    ("bl_expr", "pl_expr"),
    _DUCKDB_SUBSECOND_FNS,
    ids=("microsecond", "nanosecond", "millisecond"),
)
def test_subsecond_fns(bl_expr: bl.Expr, pl_expr: pl.Expr) -> None:
    """The results of the subsecond methods differ between DuckDB and Polars, so we need to test them separately."""
    assert_eq(bl_expr, pl_expr)


@pytest.mark.parametrize("unit", t.EpochTimeUnit.__args__)
def test_epoch(unit: t.EpochTimeUnit) -> None:
    assert_eq(bl_dt.epoch_by(unit), pl_dt.epoch(unit))


@pytest.mark.parametrize("unit", t.TimeUnit.__args__)
def test_timestamp(unit: t.TimeUnit) -> None:
    assert_eq(bl_dt.timestamp(unit), pl_dt.timestamp(unit))


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%H:%M"])
def test_date_time_boundaries(fmt: str) -> None:
    assert_eq(bl_dt.to_string(fmt), pl_dt.to_string(fmt))


def test_truncate() -> None:
    assert_eq(bl_dt.truncate("month"), pl_dt.truncate("1mo"))


def test_round() -> None:
    assert_eq(bl_dt.round("month"), pl_dt.round("1mo"))


def test_offset_by() -> None:
    assert_eq(bl_dt.offset_by("1 day"), pl_dt.offset_by("1d"))
