import polars as pl
import pytest

import pql

from ._utils import assert_eq

pql_list_num = pql.col("list_num").list
pl_list_num = pl.col("list_num").list


def test_len() -> None:
    assert_eq(pql_list_num.len(), pl_list_num.len())


def test_unique() -> None:
    assert_eq(
        pql_list_num.unique().list.sort(),
        pl_list_num.unique().list.sort(),
    )


def test_n_unique() -> None:
    assert_eq(pql_list_num.n_unique(), pl_list_num.n_unique())


def test_contains() -> None:
    assert_eq(pql_list_num.contains(2), pl_list_num.contains(2))
    assert_eq(
        pql_list_num.contains(pql.lit(None)),
        pl_list_num.contains(pl.lit(None), nulls_equal=False),
    )
    assert_eq(pql_list_num.contains(pql.col("x")), pl_list_num.contains(pl.col("x")))


def test_count_matches() -> None:
    assert_eq(pql_list_num.count_matches(5), pl_list_num.count_matches(5))
    assert_eq(
        pql.col("list_str_vals").list.count_matches("matches"),
        pl.col("list_str_vals").list.count_matches("matches"),
    )


def test_drop_nulls() -> None:
    assert_eq(
        pql.col("list_booleans").list.drop_nulls(),
        pl.col("list_booleans").list.drop_nulls(),
    )


def test_get() -> None:
    assert_eq(pql_list_num.get(0), pl_list_num.get(0))
    assert_eq(pql_list_num.get(-1), pl_list_num.get(-1))
    with pytest.raises(pl.exceptions.ComputeError, match="get index is out of bounds"):
        assert_eq(pql_list_num.get(10), pl_list_num.get(10))


def test_min() -> None:
    assert_eq(pql_list_num.min(), pl_list_num.min())


def test_max() -> None:
    assert_eq(pql_list_num.max(), pl_list_num.max())


def test_mean() -> None:
    assert_eq(pql_list_num.mean(), pl_list_num.mean())


def test_median() -> None:
    assert_eq(pql_list_num.median(), pl_list_num.median())


def test_sum() -> None:
    assert_eq(pql_list_num.sum(), pl_list_num.sum())


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("nulls_last", [True, False])
def test_sort(descending: bool, nulls_last: bool) -> None:
    assert_eq(
        pql_list_num.sort(descending=descending, nulls_last=nulls_last),
        pl_list_num.sort(descending=descending, nulls_last=nulls_last),
    )


def test_eval_num() -> None:
    assert_eq(
        pql_list_num.eval(pql.element().mul(2)), pl_list_num.eval(pl.element().mul(2))
    )


def test_eval_str() -> None:
    assert_eq(
        pql.col("list_str_vals").list.eval(pql.element().str.to_uppercase()),
        pl.col("list_str_vals").list.eval(pl.element().str.to_uppercase()),
    )


def test_eval_bool() -> None:
    assert_eq(
        pql_list_num.eval(pql.element().gt(3)), pl_list_num.eval(pl.element().gt(3))
    )


def test_first() -> None:
    assert_eq(pql_list_num.first(), pl_list_num.first())


def test_last() -> None:
    assert_eq(pql_list_num.last(), pl_list_num.last())


def test_std() -> None:
    assert_eq(pql_list_num.std(), pl_list_num.std())
    assert_eq(pql_list_num.std(ddof=0), pl_list_num.std(ddof=0))


def test_var() -> None:
    assert_eq(pql_list_num.var(), pl_list_num.var())
    assert_eq(pql_list_num.var(ddof=0), pl_list_num.var(ddof=0))


def test_reverse() -> None:
    assert_eq(pql_list_num.reverse(), pl_list_num.reverse())


def test_any() -> None:
    assert_eq(pql.col("list_booleans").list.any(), pl.col("list_booleans").list.any())


def test_all() -> None:
    assert_eq(pql.col("list_booleans").list.all(), pl.col("list_booleans").list.all())


def test_join() -> None:
    sep = "-"
    assert_eq(
        pql.col("list_str_vals").list.join(sep), pl.col("list_str_vals").list.join(sep)
    )
    assert_eq(
        pql.col("list_str_vals").list.join(sep, ignore_nulls=False),
        pl.col("list_str_vals").list.join(sep, ignore_nulls=False),
    )


def test_filter() -> None:
    assert_eq(
        pql_list_num.filter(pql.element().gt(3)),
        pl_list_num.filter(pl.element().gt(3)),
    )


def test_explode() -> None:
    assert_eq(
        pql_list_num.explode(),
        pl_list_num.explode(),
        with_cols=False,
    )
