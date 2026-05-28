from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

import belugas as bl

from ._utils import assert_eq, assert_lf_eq

bl_arr = bl.col("arr_num").arr
pl_arr = pl.col("arr_num").arr
bl_arr_str_vals = bl.col("arr_str_vals").arr
bl_arr_bool = bl.col("arr_booleans").arr
pl_arr_str_vals = pl.col("arr_str_vals").arr
pl_arr_bool = pl.col("arr_booleans").arr
type Fns = tuple[Callable[[], bl.Expr], Callable[[], pl.Expr]]
type FnsParam[**P] = tuple[Callable[P, bl.Expr], Callable[P, pl.Expr]]


def test_unique() -> None:
    assert_eq(bl_arr.unique().arr.sort(), pl_arr.unique().list.sort())


def test_contains() -> None:
    bl_f = bl_arr.contains
    pl_f = pl_arr.contains
    assert_eq(bl_f(bl.lit(None)), pl_f(pl.lit(None), nulls_equal=False))
    assert_eq(bl_f(3), pl_f(3))
    assert_eq(bl_arr_str_vals.contains("b"), pl_arr_str_vals.contains("b"))


def test_count_matches() -> None:
    assert_eq(bl_arr.count_matches(5), pl_arr.count_matches(5))


def test_drop_nulls() -> None:
    """Drop nulls don't exist for array in polars."""
    col = "arr_booleans"
    assert_eq(
        bl.col(col).arr.drop_nulls(),
        pl.col(col).cast(pl.List(pl.Boolean)).list.drop_nulls(),
    )


def test_get_out_of_bounds() -> None:

    with pytest.raises(pl.exceptions.ComputeError, match="out of bounds"):
        assert_eq(bl_arr.get(10), pl_arr.get(10))


@pytest.mark.parametrize("index", [0, 1, -1])
def test_get(index: int) -> None:
    assert_eq(bl_arr.get(index), pl_arr.get(index))


@pytest.mark.parametrize("n", [0, 1, 2, 20])
def test_head(n: int) -> None:
    assert_eq(bl_arr.head(n), pl_arr.head(n))


@pytest.mark.parametrize("n", [0, 1, 2, 20])
def test_tail(n: int) -> None:
    assert_eq(bl_arr.tail(n), pl_arr.tail(n))


def test_head_expr() -> None:
    assert_eq(bl_arr.head(bl.lit(2)), pl_arr.head(pl.lit(2)))


def test_tail_expr() -> None:
    assert_eq(bl_arr.tail(bl.lit(2)), pl_arr.tail(pl.lit(2)))


def test_head_tail_with_str_n() -> None:
    data = pl.DataFrame(
        {"vals": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "n": [1, 2, 3]},
        schema_overrides={"vals": pl.Array(pl.Int64, shape=3)},
    )
    assert_lf_eq(
        data.lazy().select(
            pl.col("vals").arr.head("n").alias("head"),
            pl.col("vals").arr.tail("n").alias("tail"),
        ),
        bl.LazyFrame(data).select(
            bl.col("vals").arr.head("n").alias("head"),
            bl.col("vals").arr.tail("n").alias("tail"),
        ),
    )


@pytest.mark.parametrize(
    "fns",
    [
        (bl_arr.min, pl_arr.min),
        (bl_arr.max, pl_arr.max),
        (bl_arr.mean, pl_arr.mean),
        (bl_arr.median, pl_arr.median),
        (bl_arr.sum, pl_arr.sum),
        (bl_arr.first, pl_arr.first),
        (bl_arr.last, pl_arr.last),
        (bl_arr.reverse, pl_arr.reverse),
        (bl_arr.len, pl_arr.len),
        (bl_arr.n_unique, pl_arr.n_unique),
    ],
)
def test_simple_method(fns: Fns) -> None:
    assert_eq(fns[0](), fns[1]())


@pytest.mark.parametrize(
    "fns",
    [(bl_arr_bool.any, pl_arr_bool.any), (bl_arr_bool.all, pl_arr_bool.all)],
)
def test_simple_bool_methods(fns: Fns) -> None:
    assert_eq(fns[0](), fns[1]())


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort(descending: bool, nulls_last: bool) -> None:
    assert_eq(
        bl_arr.sort(descending=descending, nulls_last=nulls_last),
        pl_arr.sort(descending=descending, nulls_last=nulls_last),
    )


def test_eval_num() -> None:

    assert_eq(bl_arr.eval(bl.element().mul(2)), pl_arr.eval(pl.element().mul(2)))


def test_eval_str() -> None:
    assert_eq(
        bl_arr_str_vals.eval(bl.element().str.to_uppercase()),
        pl_arr_str_vals.eval(pl.element().str.to_uppercase()),
    )


def test_eval_bool() -> None:
    assert_eq(bl_arr.eval(bl.element().gt(3)), pl_arr.eval(pl.element().gt(3)))


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("fns", [(bl_arr.var, pl_arr.var), (bl_arr.std, pl_arr.std)])
def test_std_var(fns: FnsParam[int], ddof: int) -> None:
    assert_eq(fns[0](ddof), fns[1](ddof))


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_join(ignore_nulls: bool) -> None:
    assert_eq(
        bl_arr_str_vals.join("-", ignore_nulls=ignore_nulls),
        pl_arr_str_vals.join("-", ignore_nulls=ignore_nulls),
    )


def test_filter() -> None:
    assert_eq(
        bl_arr.filter(bl.element().gt(3)),
        pl.col("arr_num").cast(pl.List(pl.UInt16)).list.filter(pl.element().gt(3)),
    )
