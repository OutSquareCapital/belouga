from collections.abc import Callable

import polars as pl
import pytest

import pql

from ._utils import assert_eq

pql_arr = pql.col("arr_num").arr
pl_arr = pl.col("arr_num").arr
pql_arr_str_vals = pql.col("arr_str_vals").arr
pql_arr_bool = pql.col("arr_booleans").arr
pl_arr_str_vals = pl.col("arr_str_vals").arr
pl_arr_bool = pl.col("arr_booleans").arr
type Fns = tuple[Callable[[], pql.Expr], Callable[[], pl.Expr]]
type FnsParam[**P] = tuple[Callable[P, pql.Expr], Callable[P, pl.Expr]]


def test_unique() -> None:
    assert_eq(pql_arr.unique().arr.sort(), pl_arr.unique().list.sort())


def test_contains() -> None:
    pql_f = pql_arr.contains
    pl_f = pl_arr.contains
    assert_eq(pql_f(pql.lit(None)), pl_f(pl.lit(None), nulls_equal=False))
    assert_eq(pql_f(3), pl_f(3))
    assert_eq(pql_arr_str_vals.contains("b"), pl_arr_str_vals.contains("b"))


def test_count_matches() -> None:
    assert_eq(pql_arr.count_matches(5), pl_arr.count_matches(5))


def test_drop_nulls() -> None:
    """Drop nulls don't exist for array in polars."""
    col = "arr_booleans"
    assert_eq(
        pql.col(col).arr.drop_nulls(),
        pl.col(col).cast(pl.List(pl.Boolean)).list.drop_nulls(),
    )


def test_get_out_of_bounds() -> None:

    with pytest.raises(pl.exceptions.ComputeError, match="out of bounds"):
        assert_eq(pql_arr.get(10), pl_arr.get(10))


@pytest.mark.parametrize("index", [0, 1, -1])
def test_get(index: int) -> None:
    assert_eq(pql_arr.get(index), pl_arr.get(index))


@pytest.mark.parametrize(
    "fns",
    [
        (pql_arr.min, pl_arr.min),
        (pql_arr.max, pl_arr.max),
        (pql_arr.mean, pl_arr.mean),
        (pql_arr.median, pl_arr.median),
        (pql_arr.sum, pl_arr.sum),
        (pql_arr.first, pl_arr.first),
        (pql_arr.last, pl_arr.last),
        (pql_arr.reverse, pl_arr.reverse),
        (pql_arr.len, pl_arr.len),
        (pql_arr.n_unique, pl_arr.n_unique),
    ],
)
def test_simple_method(fns: Fns) -> None:
    assert_eq(fns[0](), fns[1]())


@pytest.mark.parametrize(
    "fns",
    [(pql_arr_bool.any, pl_arr_bool.any), (pql_arr_bool.all, pl_arr_bool.all)],
)
def test_simple_bool_methods(fns: Fns) -> None:
    assert_eq(fns[0](), fns[1]())


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort(descending: bool, nulls_last: bool) -> None:
    assert_eq(
        pql_arr.sort(descending=descending, nulls_last=nulls_last),
        pl_arr.sort(descending=descending, nulls_last=nulls_last),
    )


def test_eval_num() -> None:

    assert_eq(pql_arr.eval(pql.element().mul(2)), pl_arr.eval(pl.element().mul(2)))


def test_eval_str() -> None:
    assert_eq(
        pql_arr_str_vals.eval(pql.element().str.to_uppercase()),
        pl_arr_str_vals.eval(pl.element().str.to_uppercase()),
    )


def test_eval_bool() -> None:
    assert_eq(pql_arr.eval(pql.element().gt(3)), pl_arr.eval(pl.element().gt(3)))


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("fns", [(pql_arr.var, pl_arr.var), (pql_arr.std, pl_arr.std)])
def test_std_var(fns: FnsParam[int], ddof: int) -> None:
    assert_eq(fns[0](ddof), fns[1](ddof))


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_join(ignore_nulls: bool) -> None:
    assert_eq(
        pql_arr_str_vals.join("-", ignore_nulls=ignore_nulls),
        pl_arr_str_vals.join("-", ignore_nulls=ignore_nulls),
    )


def test_filter() -> None:
    assert_eq(
        pql_arr.filter(pql.element().gt(3)),
        pl.col("arr_num").cast(pl.List(pl.UInt16)).list.filter(pl.element().gt(3)),
    )
