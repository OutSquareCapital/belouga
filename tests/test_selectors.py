from __future__ import annotations

from datetime import timedelta

import polars as pl
import polars.selectors as cs_pl
import pytest

import pql
import pql.selectors as cs

from ._data import sample_df
from ._utils import assert_eq_pl, assert_lf_eq_pl

_SAMPLE_DF = sample_df().to_native().pl(lazy=True)
_PQL_LF = pql.LazyFrame(_SAMPLE_DF)


def test_numeric_with_columns() -> None:
    assert_lf_eq_pl(
        _PQL_LF.select("s").with_columns(cs.numeric()),
        _SAMPLE_DF.select("s").with_columns(cs_pl.numeric()),
    )


def test_by_dtype_single() -> None:
    assert_eq_pl(cs.by_dtype(pql.Boolean), cs_pl.by_dtype(pl.Boolean))


def test_by_dtype_multiple() -> None:
    assert_eq_pl(
        cs.by_dtype(pql.Float64, pql.Int64), cs_pl.by_dtype(pl.Float64, pl.Int64)
    )


def test_union() -> None:
    assert_eq_pl(
        cs.numeric().union(cs.string()), cs_pl.numeric().__or__(cs_pl.string())
    )

    assert_eq_pl(
        cs.numeric().__or__(cs.string()), cs_pl.numeric().__or__(cs_pl.string())
    )

    assert_lf_eq_pl(
        _PQL_LF.select(cs.boolean().__or__(pql.lit(value=True))),
        _SAMPLE_DF.select(cs_pl.boolean().__or__(pl.lit(value=True))),
    )


def test_intersection() -> None:
    assert_eq_pl(
        cs.numeric().intersection(cs.by_dtype(pql.Int64)),
        cs_pl.numeric().__and__(cs_pl.by_dtype(pl.Int64)),
    )

    assert_eq_pl(
        cs.numeric().__and__(cs.by_dtype(pql.Int64)),
        cs_pl.numeric().__and__(cs_pl.by_dtype(pl.Int64)),
    )

    assert_lf_eq_pl(
        _PQL_LF.select(cs.boolean().__and__(pql.lit(value=True))),
        _SAMPLE_DF.select(cs_pl.boolean().__and__(pl.lit(value=True))),
    )


def test_difference() -> None:
    assert_eq_pl(
        cs.numeric().difference(cs.by_dtype(pql.Float64)),
        cs_pl.numeric().__sub__(cs_pl.by_dtype(pl.Float64)),
    )
    assert_eq_pl(
        cs.numeric().__sub__(cs.by_dtype(pql.Float64)),
        cs_pl.numeric().__sub__(cs_pl.by_dtype(pl.Float64)),
    )

    assert_lf_eq_pl(
        _PQL_LF.select(cs.numeric().__sub__(pql.lit(1))),
        _SAMPLE_DF.select(cs_pl.numeric().__sub__(pl.lit(1))),
    )


def test_complement() -> None:
    assert_eq_pl(cs.boolean().complement(), cs_pl.boolean().__invert__())
    assert_eq_pl(cs.boolean().__invert__(), cs_pl.boolean().__invert__())
    assert_eq_pl(cs.numeric().complement(), cs_pl.numeric().__invert__())


def test_selector_with_suffix() -> None:
    assert_eq_pl(
        cs.boolean().name.suffix("_flag"), cs_pl.boolean().name.suffix("_flag")
    )


def test_selector_cast() -> None:
    assert_eq_pl(cs.boolean().cast(pql.Int32()), cs_pl.boolean().cast(pl.Int32))


def test_selector_in_group_by_agg() -> None:
    """We need to filter null values to avoid errors on `sum`."""
    assert_lf_eq_pl(
        pql
        .LazyFrame(_SAMPLE_DF)
        .filter(pql.col("a").is_not_null())
        .group_by("a")
        .agg(cs.numeric().sum())
        .sort("a"),
        _SAMPLE_DF
        .filter(pl.col("a").is_not_null())
        .group_by("a")
        .agg(cs_pl.numeric().sum())
        .sort("a"),
    )


_selectors_lfs = [
    _PQL_LF.select(pql.col("a"), total=cs.numeric()),
    _PQL_LF.group_by("a").agg(total=cs.numeric().sum()),
]


@pytest.mark.parametrize("lf", _selectors_lfs)
def test_named_selector_collect(lf: pql.LazyFrame) -> None:
    assert_lf_eq_pl(lf, lf.collect().lazy())
    assert lf.schema.keys().into(list) == ["a", "total"]


@pytest.mark.parametrize("lf", _selectors_lfs)
def test_named_selector_lazy(lf: pql.LazyFrame) -> None:
    """Seems like when go `DuckDBPyRelation -> pl.LazyFrame`, it crashes with `pl.exceptions.ComputeError`, but not with `DuckDBPyRelation -> pl.DataFrame`."""
    msg = "column appears more than once"
    with pytest.raises(pl.exceptions.ComputeError, match=msg):
        _ = lf.lazy().collect()
    assert lf.schema.keys().into(list) == ["a", "total"]


def test_empty_selector() -> None:
    assert_lf_eq_pl(
        _PQL_LF.select(pql.col("a")).select(cs.boolean()),
        _SAMPLE_DF.select(pl.col("a")).select(cs_pl.boolean()),
    )


@pytest.mark.parametrize(
    "fn_name",
    [
        "all",
        "float",
        "integer",
        "signed_integer",
        "unsigned_integer",
        "temporal",
        "date",
        "struct",
        "nested",
        "string",
        "boolean",
        "numeric",
        "decimal",
        "binary",
        "time",
    ],
)
def test_simple_selector(fn_name: str) -> None:
    assert_eq_pl(getattr(cs, fn_name)(), getattr(cs_pl, fn_name)())  # pyright: ignore[reportAny]


def test_duration_selector() -> None:
    """Dedicated test: DuckDB INTERVAL can't roundtrip via Arrow to Polars."""
    col_names = ["dur"]
    pl_lf = pl.LazyFrame({"x": [1, 2], "dur": [timedelta(hours=1), timedelta(days=2)]})
    pql_lf = pql.LazyFrame(pl_lf)
    assert pql_lf.select(cs.duration()).columns.into(list) == col_names
    assert pl_lf.select(cs_pl.duration()).collect_schema().names() == col_names


def test_enum() -> None:
    cats = ["foo", "bar", "baz"]
    lf = pql.LazyFrame(_SAMPLE_DF)
    assert_lf_eq_pl(
        lf.with_columns(pql.col("enum").cast(pql.Enum(cats))).select(
            cs.enum().cast(pql.String())
        ),
        lf
        .lazy()
        .with_columns(pl.col("enum").cast(pl.Enum(cats)))
        .select(cs_pl.enum().cast(pl.String)),
    )


def test_matches_select() -> None:
    assert_eq_pl(cs.matches("^[abs]"), cs_pl.matches("^[abs]"))


def test_matches_no_match() -> None:
    assert_eq_pl(cs.matches("^[abs]"), cs_pl.matches("^[abs]"))


def test_by_name_single() -> None:
    assert_eq_pl(cs.by_name("x"), cs_pl.by_name("x"))


def test_by_name_multiple() -> None:
    assert_eq_pl(cs.by_name("x", "s", "a"), cs_pl.by_name("x", "s", "a"))


def test_starts_with_select() -> None:
    assert_eq_pl(cs.starts_with("s"), cs_pl.starts_with("s"))


def test_starts_with_multiple() -> None:
    assert_eq_pl(cs.starts_with("s", "a"), cs_pl.starts_with("s", "a"))


def test_ends_with_select() -> None:
    assert_eq_pl(cs.ends_with("s"), cs_pl.ends_with("s"))


def test_ends_with_multiple() -> None:
    assert_eq_pl(cs.ends_with("s", "e"), cs_pl.ends_with("s", "e"))


def test_contains_select() -> None:
    assert_eq_pl(cs.contains("al"), cs_pl.contains("al"))


def test_contains_multiple() -> None:
    assert_eq_pl(cs.contains("al", "sted"), cs_pl.contains("al", "sted"))


# ──── compositions with new selectors ────


def test_float_minus_by_name() -> None:
    assert_eq_pl(
        cs.float().__sub__(cs.by_name("nan_vals")),
        cs_pl.float().__sub__(cs_pl.by_name("nan_vals")),
    )


def test_temporal_union_string() -> None:
    assert_eq_pl(
        cs.temporal().__or__(cs.string()),
        cs_pl.temporal().__or__(cs_pl.string()),
    )


def test_all_minus_numeric() -> None:
    assert_eq_pl(
        cs.all().__sub__(cs.numeric()),
        cs_pl.all().__sub__(cs_pl.numeric()),
    )


def test_integer_intersection_by_name() -> None:
    assert_eq_pl(
        cs.integer().__and__(cs.by_name("x", "age")),
        cs_pl.integer().__and__(cs_pl.by_name("x", "age")),
    )


def test_starts_with_complement() -> None:
    assert_eq_pl(
        cs.starts_with("s").__invert__(),
        cs_pl.starts_with("s").__invert__(),
    )


# ──── name-based selectors with transforms ────


def test_by_name_with_suffix() -> None:
    assert_eq_pl(
        cs.by_name("x", "s").name.suffix("_v2"),
        cs_pl.by_name("x", "s").name.suffix("_v2"),
    )


def test_matches_cast() -> None:
    assert_eq_pl(
        cs.matches("^[xn]$").cast(pql.Float64()),
        cs_pl.matches("^[xn]$").cast(pl.Float64),
    )


def test_contains_sum_in_agg() -> None:
    assert_lf_eq_pl(
        pql
        .LazyFrame(_SAMPLE_DF)
        .filter(pql.col("a").is_not_null())
        .group_by("a")
        .agg(cs.contains("al").intersection(cs.numeric()).sum())
        .sort("a"),
        _SAMPLE_DF
        .filter(pl.col("a").is_not_null())
        .group_by("a")
        .agg(cs_pl.contains("al").__and__(cs_pl.numeric()).sum())
        .sort("a"),
    )
