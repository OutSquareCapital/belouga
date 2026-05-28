from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs_pl
import pytest
from pyochain import Seq

import belugas as bl
from belugas import selectors as cs

from ._data import sample_bl, sample_lf
from ._utils import assert_eq, assert_lf_eq, into_ids

if TYPE_CHECKING:
    from collections.abc import Callable


def test_with_columns() -> None:
    assert_lf_eq(
        sample_lf().select("s").with_columns(cs_pl.contains("x")),
        sample_bl().select("s").with_columns(cs.contains("x")),
    )


def test_by_dtype_single() -> None:
    assert_eq(cs.by_dtype(bl.Boolean), cs_pl.by_dtype(pl.Boolean))


def test_by_dtype_multiple() -> None:
    assert_eq(cs.by_dtype(bl.Float64, bl.Int64), cs_pl.by_dtype(pl.Float64, pl.Int64))


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
    assert_eq(cs.by_name("x").cast(bl.Boolean()), cs_pl.by_name("x").cast(pl.Boolean))


@pytest.mark.parametrize(
    ("bl_expr", "pl_expr"),
    [
        (cs.by_name("x").cast(bl.Boolean()), cs_pl.by_name("x").cast(pl.Boolean)),
        (cs.by_name("x").sum(), cs_pl.by_name("x").sum()),
        (cs.by_name("a").not_(), cs_pl.by_name("a").not_()),
        (
            cs.by_name("x").name.suffix("_flag"),
            cs_pl.by_name("x").name.suffix("_flag"),
        ),
        (
            cs.by_name("a").__or__(bl.lit(value=False)),
            cs_pl.by_name("a").__or__(pl.lit(value=False)),
        ),
        (
            cs.by_name("a").__and__(bl.lit(value=True)),
            cs_pl.by_name("a").__and__(pl.lit(value=True)),
        ),
        (
            cs.by_name("x").__sub__(bl.lit(1)),
            cs_pl.by_name("x").__sub__(pl.lit(1)),
        ),
        (
            cs.by_name("a", "b").__or__(bl.lit(value=False)).__invert__(),
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
def test_selector_into_expr(bl_expr: bl.Expr, pl_expr: pl.Expr) -> None:
    assert isinstance(pl_expr, pl.Expr)
    assert not cs_pl.is_selector(pl_expr)
    assert isinstance(bl_expr, bl.Expr)
    assert not isinstance(bl_expr, cs.Selector)
    assert_eq(bl_expr, pl_expr)


def test_selector_in_group_by_agg() -> None:
    """We need to filter null values to avoid errors on `sum`."""
    assert_lf_eq(
        sample_lf()
        .filter(pl.col("a").is_not_null())
        .group_by("a")
        .agg(cs_pl.contains("_vals"))
        .sort("a"),
        sample_bl()
        .filter(bl.col("a").is_not_null())
        .group_by("a")
        .agg(cs.contains("_vals"))
        .sort("a"),
    )


@pytest.mark.parametrize(
    "lf",
    [
        sample_bl().select(bl.col("a"), total=cs.contains("vals")),
        sample_bl().group_by("a").agg(total=cs.contains("vals")),
    ],
    ids=["select", "agg"],
)
def test_named_selector(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy(), lf)
    assert lf.columns.into(list) == ["a", "total"]


def test_empty_selector() -> None:
    assert_lf_eq(
        sample_lf().select(pl.col("a")).select(cs_pl.contains("x")),
        sample_bl().select(bl.col("a")).select(cs.contains("x")),
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


@pytest.mark.parametrize("fns", _SELECTOR_FNS, ids=_SELECTOR_FNS.into(into_ids))
def test_dtype_selector(
    fns: tuple[Callable[[], cs.Selector], Callable[[], cs_pl.Selector]],
) -> None:
    assert_eq(fns[0](), fns[1]())


def test_all_selector() -> None:
    assert_eq(cs.all(), cs_pl.all())


def test_duration_selector() -> None:
    """Dedicated test: DuckDB INTERVAL can't roundtrip via Arrow to Polars."""
    col_names = ["dur"]
    pl_lf = pl.LazyFrame({"x": [1, 2], "dur": [timedelta(hours=1), timedelta(days=2)]})
    bl_lf = bl.LazyFrame(pl_lf)
    assert bl_lf.select(cs.duration()).columns.into(list) == col_names
    assert pl_lf.select(cs_pl.duration()).collect_schema().names() == col_names


def test_enum() -> None:
    cats = ["foo", "bar", "baz"]
    lf = bl.LazyFrame(sample_lf())
    assert_lf_eq(
        lf
        .lazy()
        .with_columns(pl.col("enum").cast(pl.Enum(cats)))
        .select(cs_pl.enum().cast(pl.String)),
        lf.with_columns(bl.col("enum").cast(bl.Enum(cats))).select(
            cs.enum().cast(bl.String())
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
        cs.matches("^[xn]$").cast(bl.Float64()),
        cs_pl.matches("^[xn]$").cast(pl.Float64),
    )


def test_complex_selector() -> None:
    """Complex check test case.

    Here we want all the cols that:

        - contains "al"
        - OR ends with "_vals"
        - AND doesn't contain "str"
    """
    bl_slctor = cs.contains("al").intersection(
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
        bl
        .LazyFrame(sample_lf())
        .filter(bl.col("a").is_not_null())
        .group_by("a")
        .agg(bl_slctor)
        .sort("a"),
    )


def test_selector_after_drop() -> None:
    data = {"x": [1], "s": ["a"], "f": [1.0]}
    bl_lf = bl.LazyFrame(data).drop("s").select(cs.integer())
    pl_lf = pl.LazyFrame(data).drop("s").select(cs_pl.integer())
    assert_lf_eq(pl_lf, bl_lf)


def test_selector_after_join() -> None:
    data_l = {"k": [1], "x": [10]}
    data_r = {"k": [1], "s": ["a"]}
    bl_l = bl.LazyFrame(data_l)
    bl_r = bl.LazyFrame(data_r)
    pl_l = pl.LazyFrame(data_l)
    pl_r = pl.LazyFrame(data_r)
    assert_lf_eq(
        pl_l.join(pl_r, on="k", how="left").select(cs_pl.integer()),
        bl_l.join(bl_r, on="k", how="left").select(cs.integer()),
    )


def test_selector_after_agg() -> None:
    data = {"k": ["a", "b"], "x": [1, 2], "f": [1.0, 2.0]}
    bl_lf = bl.LazyFrame(data).group_by("k").agg(bl.col("x").sum(), bl.col("f").sum())
    pl_lf = pl.LazyFrame(data).group_by("k").agg(pl.col("x").sum(), pl.col("f").sum())
    assert_lf_eq(pl_lf.sort("k"), bl_lf.sort("k"))
    assert bl_lf.select(cs.integer()).columns.into(list) == ["x"]


def test_selector_after_unpivot() -> None:
    data = {"id": ["a"], "x": [1], "y": [2]}
    bl_lf = bl.LazyFrame(data).unpivot(on=["x", "y"], index=["id"]).select(cs.string())
    pl_lf = (
        pl.LazyFrame(data).unpivot(on=["x", "y"], index=["id"]).select(cs_pl.string())
    )
    assert_lf_eq(pl_lf, bl_lf)


def test_selector_after_unnest() -> None:
    data = {"s": [{"x": 1, "y": 2}], "k": ["a"]}
    bl_lf = bl.LazyFrame(data).unnest("s").select(cs.integer())
    pl_lf = pl.LazyFrame(data).unnest("s").select(cs_pl.integer())
    assert_lf_eq(pl_lf, bl_lf)
