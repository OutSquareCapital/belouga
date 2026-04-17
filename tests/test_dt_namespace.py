import polars as pl
import pytest
from polars.expr.datetime import ExprDateTimeNameSpace

import pql
import pql.sql.typing as t
from pql._namespaces import ExprDateTimeNameSpace as PqlExprDateTimeNameSpace

from ._utils import assert_eq, on_simple_fn


def _pql_dt() -> PqlExprDateTimeNameSpace:
    return pql.col("dt").dt


def _pl_dt() -> ExprDateTimeNameSpace:
    return pl.col("dt").dt


_SIMPLE_FNS = {
    "microsecond",
    "nanosecond",
    "millisecond",
    "second",
    "minute",
    "hour",
    "day",
    "weekday",
    "ordinal_day",
    "week",
    "month",
    "quarter",
    "year",
    "iso_year",
    "century",
    "millennium",
    "date",
    "time",
    "month_start",
    "month_end",
}


@pytest.mark.parametrize("fn", sorted(_SIMPLE_FNS))
def test_simple_fns(fn: str) -> None:
    on_simple_fn(_pql_dt(), _pl_dt(), fn)


@pytest.mark.parametrize("unit", t.EpochTimeUnit.__args__)
def test_epoch(unit: t.EpochTimeUnit) -> None:
    assert_eq(_pql_dt().epoch(unit), _pl_dt().epoch(unit))


@pytest.mark.parametrize("unit", t.TimeUnit.__args__)
def test_timestamp(unit: t.TimeUnit) -> None:
    assert_eq(_pql_dt().timestamp(unit), _pl_dt().timestamp(unit))


@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%H:%M"])
def test_date_time_boundaries(fmt: str) -> None:
    assert_eq(_pql_dt().to_string(fmt), _pl_dt().to_string(fmt))


def test_truncate() -> None:
    assert_eq(_pql_dt().truncate("month"), _pl_dt().truncate("1mo"))


def test_round() -> None:
    assert_eq(_pql_dt().round("month"), _pl_dt().round("1mo"))


def test_offset_by() -> None:
    assert_eq(_pql_dt().offset_by("1 day"), _pl_dt().offset_by("1d"))
