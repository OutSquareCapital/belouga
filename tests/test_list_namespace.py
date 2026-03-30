import narwhals as nw
import polars as pl
import pytest

import pql

from ._utils import assert_eq, assert_eq_pl


def test_len() -> None:
    assert_eq(pql.col("list_num").list.len(), nw.col("list_num").list.len())


def test_unique() -> None:
    assert_eq(pql.col("list_num").list.unique(), nw.col("list_num").list.unique())


def test_n_unique() -> None:
    assert_eq_pl(
        pql.col("list_num").list.n_unique(), pl.col("list_num").list.n_unique()
    )


def test_contains() -> None:
    assert_eq(pql.col("list_num").list.contains(2), nw.col("list_num").list.contains(2))
    assert_eq_pl(
        (
            pql.col("list_num").list.contains(pql.lit(None)).alias("x_nulls_neq"),
            pql.col("list_num").list.contains(pql.col("x")).alias("x_nulls_neq_y"),
            pql.col("list_num").list.contains(pql.col("x")).alias("x_y"),
        ),
        (
            pl
            .col("list_num")
            .list.contains(None, nulls_equal=False)
            .alias("x_nulls_neq"),
            pl
            .col("list_num")
            .list.contains(pl.col("x"), nulls_equal=False)
            .alias("x_nulls_neq_y"),
            pl.col("list_num").list.contains(pl.col("x")).alias("x_y"),
        ),
    )


def test_count_matches() -> None:
    assert_eq_pl(
        pql.col("list_num").list.count_matches(5),
        pl.col("list_num").list.count_matches(5),
    )
    assert_eq_pl(
        pql.col("list_str_vals").list.count_matches(pql.lit("matches")),
        pl.col("list_str_vals").list.count_matches("matches"),
    )


def test_drop_nulls() -> None:
    assert_eq_pl(
        pql.col("list_booleans").list.drop_nulls(),
        pl.col("list_booleans").list.drop_nulls(),
    )


def test_get() -> None:
    assert_eq(
        pql.col("list_num").list.get(0),
        nw.col("list_num").list.get(0),
    )
    assert_eq_pl(
        pql.col("list_num").list.get(-1),
        pl.col("list_num").list.get(-1),
    )
    with pytest.raises(pl.exceptions.ComputeError, match="get index is out of bounds"):
        assert_eq_pl(
            pql.col("list_num").list.get(10),
            pl.col("list_num").list.get(10),
        )


def test_min() -> None:
    assert_eq(pql.col("list_num").list.min(), nw.col("list_num").list.min())


def test_max() -> None:
    assert_eq(pql.col("list_num").list.max(), nw.col("list_num").list.max())


def test_mean() -> None:
    assert_eq(pql.col("list_num").list.mean(), nw.col("list_num").list.mean())


def test_median() -> None:
    assert_eq(pql.col("list_num").list.median(), nw.col("list_num").list.median())


def test_sum() -> None:
    assert_eq(pql.col("list_num").list.sum(), nw.col("list_num").list.sum())


def test_sort() -> None:
    assert_eq(
        (
            pql.col("list_num").list.sort().alias("x_sorted"),
            pql
            .col("list_num")
            .list.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
        (
            nw.col("list_num").list.sort().alias("x_sorted"),
            nw
            .col("list_num")
            .list.sort(descending=True, nulls_last=True)
            .alias("x_sorted_desc"),
        ),
    )


def test_eval_num() -> None:
    assert_eq_pl(
        pql.col("list_num").list.eval(pql.element().mul(2)),
        pl.col("list_num").list.eval(pl.element().mul(2)),
    )


def test_eval_str() -> None:
    assert_eq_pl(
        pql.col("list_str_vals").list.eval(pql.element().str.to_uppercase()),
        pl.col("list_str_vals").list.eval(pl.element().str.to_uppercase()),
    )


def test_eval_bool() -> None:
    assert_eq_pl(
        pql.col("list_num").list.eval(pql.element() > 3),
        pl.col("list_num").list.eval(pl.element() > 3),
    )


def test_first() -> None:
    assert_eq_pl(pql.col("list_num").list.first(), pl.col("list_num").list.first())


def test_last() -> None:
    assert_eq_pl(pql.col("list_num").list.last(), pl.col("list_num").list.last())


def test_std() -> None:
    assert_eq_pl(pql.col("list_num").list.std(), pl.col("list_num").list.std())
    assert_eq_pl(
        pql.col("list_num").list.std(ddof=0), pl.col("list_num").list.std(ddof=0)
    )


def test_var() -> None:
    assert_eq_pl(pql.col("list_num").list.var(), pl.col("list_num").list.var())
    assert_eq_pl(
        pql.col("list_num").list.var(ddof=0), pl.col("list_num").list.var(ddof=0)
    )


def test_reverse() -> None:
    assert_eq_pl(pql.col("list_num").list.reverse(), pl.col("list_num").list.reverse())


def test_any() -> None:
    assert_eq_pl(
        pql.col("list_booleans").list.any(), pl.col("list_booleans").list.any()
    )


def test_all() -> None:
    assert_eq_pl(
        pql.col("list_booleans").list.all(), pl.col("list_booleans").list.all()
    )


def test_join() -> None:
    sep = pql.lit("-")
    assert_eq_pl(
        pql.col("list_str_vals").list.join(sep), pl.col("list_str_vals").list.join("-")
    )
    assert_eq_pl(
        pql.col("list_str_vals").list.join(sep, ignore_nulls=False),
        pl.col("list_str_vals").list.join("-", ignore_nulls=False),
    )


def test_filter() -> None:
    assert_eq_pl(
        pql.col("list_num").list.filter(pql.element().gt(3)),
        pl.col("list_num").list.filter(pl.element().gt(3)),
    )


def test_explode() -> None:
    assert_eq_pl(
        pql.col("list_num").list.explode(),
        pl.col("list_num").list.explode(),
    )
