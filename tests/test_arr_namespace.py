import narwhals as nw
import polars as pl
import pytest

import pql

from ._utils import assert_eq, assert_eq_pl


def test_len() -> None:
    assert_eq(pql.col("arr_num").arr.len(), nw.col("arr_num").list.len())


def test_unique() -> None:
    assert_eq(pql.col("arr_num").arr.unique(), nw.col("arr_num").list.unique())


def test_n_unique() -> None:
    assert_eq_pl(pql.col("arr_num").arr.n_unique(), pl.col("arr_num").arr.n_unique())


def test_contains() -> None:
    assert_eq(pql.col("arr_num").arr.contains(2), nw.col("arr_num").list.contains(2))
    assert_eq_pl(
        (
            pql.col("arr_num").arr.contains(pql.lit(None)).alias("x_nulls_neq"),
            pql.col("arr_num").arr.contains("x").alias("x_nulls_neq_y"),
            pql.col("arr_num").arr.contains(3).alias("x_y"),
        ),
        (
            pl
            .col("arr_num")
            .arr.contains(pl.lit(None), nulls_equal=False)
            .alias("x_nulls_neq"),
            pl
            .col("arr_num")
            .arr.contains(pl.col("x"), nulls_equal=False)
            .alias("x_nulls_neq_y"),
            pl.col("arr_num").arr.contains(3).alias("x_y"),
        ),
    )


def test_count_matches() -> None:
    assert_eq_pl(
        pql.col("arr_num").arr.count_matches(5), pl.col("arr_num").arr.count_matches(5)
    )


def test_drop_nulls() -> None:
    """Drop nulls don't exist for array in polars."""
    assert_eq_pl(
        pql.col("arr_booleans").arr.drop_nulls(),
        pl.col("arr_booleans").cast(pl.List(pl.Boolean)).list.drop_nulls(),
    )


def test_get_out_of_bounds() -> None:

    with pytest.raises(pl.exceptions.ComputeError, match="out of bounds"):
        assert_eq_pl(pql.col("arr_num").arr.get(10), pl.col("arr_num").arr.get(10))


@pytest.mark.parametrize("index", [0, 1, -1])
def test_get(index: int) -> None:
    assert_eq_pl(pql.col("arr_num").arr.get(index), pl.col("arr_num").arr.get(index))


def test_min() -> None:
    assert_eq(pql.col("arr_num").arr.min(), nw.col("arr_num").list.min())


def test_max() -> None:
    assert_eq(pql.col("arr_num").arr.max(), nw.col("arr_num").list.max())


def test_mean() -> None:
    assert_eq(pql.col("arr_num").arr.mean(), nw.col("arr_num").list.mean())


def test_median() -> None:
    assert_eq(pql.col("arr_num").arr.median(), nw.col("arr_num").list.median())


def test_sum() -> None:
    assert_eq(pql.col("arr_num").arr.sum(), nw.col("arr_num").list.sum())


def test_sort() -> None:
    assert_eq(
        (
            pql.col("arr_num").arr.sort().alias("x_sorted"),
            pql
            .col("arr_num")
            .arr.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
        (
            nw.col("arr_num").list.sort().alias("x_sorted"),
            nw
            .col("arr_num")
            .list.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
    )


def test_eval_num() -> None:
    assert_eq_pl(
        pql.col("arr_num").arr.eval(pql.element().mul(2)),
        pl.col("arr_num").arr.eval(pl.element().mul(2)),
    )


def test_eval_str() -> None:
    assert_eq_pl(
        pql.col("arr_str_vals").arr.eval(pql.element().str.to_uppercase()),
        pl.col("arr_str_vals").arr.eval(pl.element().str.to_uppercase()),
    )


def test_eval_bool() -> None:
    assert_eq_pl(
        pql.col("arr_num").arr.eval(pql.element() > 3),
        pl.col("arr_num").arr.eval(pl.element() > 3),
    )


def test_first() -> None:
    assert_eq_pl(pql.col("arr_num").arr.first(), pl.col("arr_num").arr.first())


def test_last() -> None:
    assert_eq_pl(pql.col("arr_num").arr.last(), pl.col("arr_num").arr.last())


def test_std() -> None:
    assert_eq_pl(pql.col("arr_num").arr.std(), pl.col("arr_num").arr.std())
    assert_eq_pl(pql.col("arr_num").arr.std(ddof=0), pl.col("arr_num").arr.std(ddof=0))


def test_var() -> None:
    assert_eq_pl(pql.col("arr_num").arr.var(), pl.col("arr_num").arr.var())
    assert_eq_pl(pql.col("arr_num").arr.var(ddof=0), pl.col("arr_num").arr.var(ddof=0))


def test_reverse() -> None:
    assert_eq_pl(pql.col("arr_num").arr.reverse(), pl.col("arr_num").arr.reverse())


def test_any() -> None:
    assert_eq_pl(pql.col("arr_booleans").arr.any(), pl.col("arr_booleans").arr.any())


def test_all() -> None:
    assert_eq_pl(pql.col("arr_booleans").arr.all(), pl.col("arr_booleans").arr.all())


def test_join() -> None:
    sep = pql.lit("-")
    assert_eq_pl(
        pql.col("arr_str_vals").arr.join(sep), pl.col("arr_str_vals").arr.join("-")
    )
    assert_eq_pl(
        pql.col("arr_str_vals").arr.join(sep, ignore_nulls=False),
        pl.col("arr_str_vals").arr.join("-", ignore_nulls=False),
    )


def test_filter() -> None:
    assert_eq_pl(
        pql.col("arr_num").arr.filter(pql.element().gt(3)),
        pl.col("arr_num").cast(pl.List(pl.UInt16)).list.filter(pl.element().gt(3)),
    )
