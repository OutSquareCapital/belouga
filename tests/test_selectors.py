from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta

import polars as pl
import polars.selectors as cs_pl
import pytest
from pyochain import Seq

import pql
from pql import selectors as cs

from ._data import sample_lf, sample_pql
from ._utils import assert_eq, assert_lf_eq

skipped = pytest.mark.skip(reason="Temp deletion of selectors by dtype")


def test_with_columns() -> None:
    assert_lf_eq(
        sample_lf().select("s").with_columns(cs_pl.contains("x")),
        sample_pql().select("s").with_columns(cs.contains("x")),
    )


@skipped
def test_by_dtype_single() -> None:
    assert_eq(cs.by_dtype(pql.Boolean), cs_pl.by_dtype(pl.Boolean))


@skipped
def test_by_dtype_multiple() -> None:
    assert_eq(cs.by_dtype(pql.Float64, pql.Int64), cs_pl.by_dtype(pl.Float64, pl.Int64))


def test_union() -> None:
    assert_eq(
        cs.contains("arr_").union(cs.contains("a")),
        cs_pl.contains("arr_").__or__(cs_pl.contains("a")),
    )

    assert_eq(
        cs.starts_with("arr_").__or__(cs.contains("a")),
        cs_pl.starts_with("arr_").__or__(cs_pl.contains("a")),
    )


def test_intersection() -> None:
    assert_eq(
        cs.contains("x").intersection(cs.contains("a")),
        cs_pl.contains("x").__and__(cs_pl.contains("a")),
    )

    assert_eq(
        cs.contains("x").__and__(cs.contains("a")),
        cs_pl.contains("x").__and__(cs_pl.contains("a")),
    )


def test_difference() -> None:
    assert_eq(
        cs.contains("x").difference(cs.contains("a")),
        cs_pl.contains("x").__sub__(cs_pl.contains("a")),
    )
    assert_eq(
        cs.contains("x").__sub__(cs.contains("a")),
        cs_pl.contains("x").__sub__(cs_pl.contains("a")),
    )


def test_complement() -> None:
    assert_eq(cs.contains("x").complement(), cs_pl.contains("x").__invert__())
    assert_eq(cs.contains("x").__invert__(), cs_pl.contains("x").__invert__())


def test_selector_with_suffix() -> None:
    assert_eq(
        cs.by_name("x").name.suffix("_flag"), cs_pl.by_name("x").name.suffix("_flag")
    )


def test_selector_cast() -> None:
    assert_eq(cs.by_name("x").cast(pql.Boolean()), cs_pl.by_name("x").cast(pl.Boolean))


@pytest.mark.parametrize(
    ("pql_expr", "pl_expr"),
    [
        (cs.by_name("x").cast(pql.Boolean()), cs_pl.by_name("x").cast(pl.Boolean)),
        (cs.by_name("x").sum(), cs_pl.by_name("x").sum()),
        (cs.by_name("a").not_(), cs_pl.by_name("a").not_()),
        (
            cs.by_name("x").name.suffix("_flag"),
            cs_pl.by_name("x").name.suffix("_flag"),
        ),
        (
            cs.by_name("a").__or__(pql.lit(value=False)),
            cs_pl.by_name("a").__or__(pl.lit(value=False)),
        ),
        (
            cs.by_name("a").__and__(pql.lit(value=True)),
            cs_pl.by_name("a").__and__(pl.lit(value=True)),
        ),
        (
            cs.by_name("x").__sub__(pql.lit(1)),
            cs_pl.by_name("x").__sub__(pl.lit(1)),
        ),
        (
            cs.by_name("a", "b").__or__(pql.lit(value=False)).__invert__(),
            cs_pl.by_name("a", "b").__or__(pl.lit(value=False)).__invert__(),
        ),
        (
            cs.by_name("a", "b").not_().__invert__(),
            cs_pl.by_name("a", "b").not_().__invert__(),
        ),
    ],
    ids=(
        "cast",
        "sum",
        "not",
        "suffix",
        "or",
        "and",
        "sub",
        "or-then-invert",
        "not-then-invert",
    ),
)
def test_selector_into_expr(pql_expr: pql.Expr, pl_expr: pl.Expr) -> None:
    assert isinstance(pl_expr, pl.Expr)
    assert not cs_pl.is_selector(pl_expr)
    assert isinstance(pql_expr, pql.Expr)
    assert not isinstance(pql_expr, cs.Selector)
    assert_eq(pql_expr, pl_expr)


def test_selector_in_group_by_agg() -> None:
    """We need to filter null values to avoid errors on `sum`."""
    assert_lf_eq(
        sample_lf()
        .filter(pl.col("a").is_not_null())
        .group_by("a")
        .agg(cs_pl.contains("_vals"))
        .sort("a"),
        sample_pql()
        .filter(pql.col("a").is_not_null())
        .group_by("a")
        .agg(cs.contains("_vals"))
        .sort("a"),
    )


@pytest.mark.parametrize(
    "lf",
    [
        sample_pql().select(pql.col("a"), total=cs.contains("vals")),
        sample_pql().group_by("a").agg(total=cs.contains("vals")),
    ],
    ids=["select", "agg"],
)
def test_named_selector(lf: pql.LazyFrame) -> None:
    assert_lf_eq(lf.lazy(), lf)
    assert lf.schema.keys().into(list) == ["a", "total"]


def test_empty_selector() -> None:
    assert_lf_eq(
        sample_lf().select(pl.col("a")).select(cs_pl.contains("x")),
        sample_pql().select(pql.col("a")).select(cs.contains("x")),
    )


_SELECTOR_FNS = Seq((
    (cs.float, cs_pl.float),
    (cs.integer, cs_pl.integer),
    (cs.signed_integer, cs_pl.signed_integer),
    (cs.unsigned_integer, cs_pl.unsigned_integer),
    (cs.temporal, cs_pl.temporal),
    (cs.date, cs_pl.date),
    (cs.struct, cs_pl.struct),
    (cs.nested, cs_pl.nested),
    (cs.string, cs_pl.string),
    (cs.boolean, cs_pl.boolean),
    (cs.numeric, cs_pl.numeric),
    (cs.decimal, cs_pl.decimal),
    (cs.binary, cs_pl.binary),
    (cs.time, cs_pl.time),
))


@pytest.mark.parametrize(
    "fns",
    _SELECTOR_FNS,
    ids=_SELECTOR_FNS.iter().map_star(lambda f1, _f2: f1.__name__),
)
@skipped
def test_dtype_selector(
    fns: tuple[Callable[[], cs.Selector], Callable[[], cs_pl.Selector]],
) -> None:
    assert_eq(fns[0](), fns[1]())


def test_all_selector() -> None:
    assert_eq(cs.all(), cs_pl.all())


@skipped
def test_duration_selector() -> None:
    """Dedicated test: DuckDB INTERVAL can't roundtrip via Arrow to Polars."""
    col_names = ["dur"]
    pl_lf = pl.LazyFrame({"x": [1, 2], "dur": [timedelta(hours=1), timedelta(days=2)]})
    pql_lf = pql.LazyFrame(pl_lf)
    assert pql_lf.select(cs.duration()).columns.into(list) == col_names
    assert pl_lf.select(cs_pl.duration()).collect_schema().names() == col_names


@skipped
def test_enum() -> None:
    cats = ["foo", "bar", "baz"]
    lf = pql.LazyFrame(sample_lf())
    assert_lf_eq(
        lf
        .lazy()
        .with_columns(pl.col("enum").cast(pl.Enum(cats)))
        .select(cs_pl.enum().cast(pl.String)),
        lf.with_columns(pql.col("enum").cast(pql.Enum(cats))).select(
            cs.enum().cast(pql.String())
        ),
    )


def test_matches_select() -> None:
    assert_eq(cs.matches("^[abs]"), cs_pl.matches("^[abs]"))


def test_matches_no_match() -> None:
    assert_eq(cs.matches("^[abs]"), cs_pl.matches("^[abs]"))


def test_by_name_single() -> None:
    assert_eq(cs.by_name("x"), cs_pl.by_name("x"))


def test_by_name_multiple() -> None:
    assert_eq(cs.by_name("x", "s", "a"), cs_pl.by_name("x", "s", "a"))


def test_starts_with_select() -> None:
    assert_eq(cs.starts_with("s"), cs_pl.starts_with("s"))


def test_starts_with_multiple() -> None:
    assert_eq(cs.starts_with("s", "a"), cs_pl.starts_with("s", "a"))


def test_ends_with_select() -> None:
    assert_eq(cs.ends_with("s"), cs_pl.ends_with("s"))


def test_ends_with_multiple() -> None:
    assert_eq(cs.ends_with("s", "e"), cs_pl.ends_with("s", "e"))


def test_contains_select() -> None:
    assert_eq(cs.contains("al"), cs_pl.contains("al"))


def test_contains_multiple() -> None:
    assert_eq(cs.contains("al", "sted"), cs_pl.contains("al", "sted"))


# ──── compositions with new selectors ────


def test_contains_vals_minus_by_name() -> None:
    assert_eq(
        cs.contains("vals").__sub__(cs.by_name("nan_vals")),
        cs_pl.contains("vals").__sub__(cs_pl.by_name("nan_vals")),
    )


@skipped
def test_temporal_union_string() -> None:
    assert_eq(
        cs.temporal().__or__(cs.string()),
        cs_pl.temporal().__or__(cs_pl.string()),
    )


def test_all_minus_by_name() -> None:
    assert_eq(
        cs.all().__sub__(cs.by_name("x", "age")),
        cs_pl.all().__sub__(cs_pl.by_name("x", "age")),
    )


def test_integer_intersection_by_name() -> None:
    assert_eq(
        cs.starts_with("s").__and__(cs.by_name("x", "age")),
        cs_pl.starts_with("s").__and__(cs_pl.by_name("x", "age")),
    )


def test_starts_with_complement() -> None:
    assert_eq(
        cs.starts_with("s").__invert__(),
        cs_pl.starts_with("s").__invert__(),
    )


# ──── name-based selectors with transforms ────


def test_by_name_with_suffix() -> None:
    assert_eq(
        cs.by_name("x", "s").name.suffix("_v2"),
        cs_pl.by_name("x", "s").name.suffix("_v2"),
    )


def test_matches_cast() -> None:
    assert_eq(
        cs.matches("^[xn]$").cast(pql.Float64()),
        cs_pl.matches("^[xn]$").cast(pl.Float64),
    )


def test_complex_selector() -> None:
    """Complex check test case.

    Here we want all the cols that:

        - contains "al"
        - OR ends with "_vals"
        - AND doesn't contain "str"
    """
    pql_slctor = cs.contains("al").intersection(
        cs.ends_with("_vals").intersection(cs.contains("str").complement())
    )
    pl_slctor = cs_pl.contains("al").__and__(
        cs_pl.ends_with("_vals").__and__(cs_pl.contains("str").__invert__())
    )
    assert_lf_eq(
        sample_lf()
        .filter(pl.col("a").is_not_null())
        .group_by("a")
        .agg(pl_slctor)
        .sort("a"),
        pql
        .LazyFrame(sample_lf())
        .filter(pql.col("a").is_not_null())
        .group_by("a")
        .agg(pql_slctor)
        .sort("a"),
    )
