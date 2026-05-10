import polars as pl
import pytest
from polars._typing import JoinStrategy as PlJoinStrategy

import belugas as bl
import belugas.typing as t

from ._utils import assert_lf_eq

LEFT = pl.DataFrame({"id1": [1, 2, 3], "id2": ["a", "b", "c"], "a": [10, 20, 30]})
RIGHT = pl.DataFrame({"id1": [2, 3, 4], "id2": ["b", "c", "d"], "b": [200, 300, 400]})


@pytest.mark.parametrize("on", [["id1", "id2"], ["id1"]])
@pytest.mark.parametrize("how", t.JoinStrategy.__args__)
def test_join_on(on: list[str], how: t.JoinStrategy) -> None:
    assert_lf_eq(
        LEFT.lazy().join(RIGHT.lazy(), on=on, how=_pl_how(how)),
        bl.LazyFrame(LEFT).join(bl.LazyFrame(RIGHT), on=on, how=how),
    )


@pytest.mark.parametrize("on", [["id1", "id2"], ["id1"]])
@pytest.mark.parametrize("how", t.JoinStrategy.__args__)
def test_join_left_on_right_on(on: list[str], how: t.JoinStrategy) -> None:
    assert_lf_eq(
        LEFT.lazy().join(RIGHT.lazy(), left_on=on, right_on=on, how=_pl_how(how)),
        bl.LazyFrame(LEFT).join(bl.LazyFrame(RIGHT), left_on=on, right_on=on, how=how),
    )


def _pl_how(how: t.JoinStrategy) -> PlJoinStrategy:
    return "full" if how == "outer" else how


def test_join_cross() -> None:
    assert_lf_eq(
        LEFT.lazy().join(RIGHT.lazy(), how="cross"),
        bl.LazyFrame(LEFT).join_cross(bl.LazyFrame(RIGHT)),
    )


@pytest.mark.parametrize("strategy", t.AsofJoinStrategy.__args__)
def test_join_asof_strat(strategy: t.AsofJoinStrategy) -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"u": [0, 3, 8], "g2": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_lf_eq(
        left.lazy().join_asof(
            right.lazy(),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy=strategy,
        ),
        bl.LazyFrame(left).join_asof(
            bl.LazyFrame(right),
            left_on="t",
            right_on="u",
            by_left="g",
            by_right="g2",
            strategy=strategy,
        ),
    )


LEFT_ERR = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
RIGHT_ERR = pl.DataFrame({"u": [0, 3, 8], "b": [100, 200, 300]})


def test_join_asof_error_on_and_left_on() -> None:
    with pytest.raises(ValueError, match="If `on` is specified"):
        _ = (
            bl
            .LazyFrame(LEFT_ERR)
            .join_asof(bl.LazyFrame(RIGHT_ERR), on="t", left_on="t", right_on="u")
            .collect()
        )


def test_join_asof_error_no_keys() -> None:
    with pytest.raises(ValueError, match="Either"):
        _ = bl.LazyFrame(LEFT_ERR).join_asof(bl.LazyFrame(RIGHT_ERR)).collect()


def test_join_asof_error_left_on_without_right_on() -> None:
    with pytest.raises(ValueError, match="Either"):
        _ = (
            bl
            .LazyFrame(LEFT_ERR)
            .join_asof(bl.LazyFrame(RIGHT_ERR), left_on="t")
            .collect()
        )


left_asof_error = bl.LazyFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
right_asof_error = bl.LazyFrame({
    "u": [0, 3, 8],
    "g2": ["x", "x", "y"],
    "b": [100, 200, 300],
})


def test_join_asof_error_by_and_by_left() -> None:
    with pytest.raises(ValueError, match="If `by` is specified"):
        _ = left_asof_error.join_asof(
            right_asof_error, left_on="t", right_on="u", by="g", by_left="g"
        ).collect()


def test_join_asof_error_by_left_without_by_right() -> None:
    with pytest.raises(ValueError, match="Can not specify only"):
        _ = left_asof_error.join_asof(
            right_asof_error, left_on="t", right_on="u", by_left="g"
        ).collect()


def test_join_asof_error_unequal_by_lengths() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        _ = left_asof_error.join_asof(
            right_asof_error,
            left_on="t",
            right_on="u",
            by_left="g",
            by_right=["g2", "b"],
        ).collect()


def test_join_left_on_right_on_length_mismatch() -> None:
    left = pl.DataFrame({"id1": [1, 2], "id2": ["a", "b"], "a": [10, 20]})
    right = pl.DataFrame({"id1": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="same length"):
        _ = (
            bl
            .LazyFrame(left)
            .join(bl.LazyFrame(right), left_on=["id1", "id2"], right_on="id1")
            .collect()
        )


def test_join_asof_with_by() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "g": ["x", "x", "y"], "a": [1, 2, 3]})
    right = pl.DataFrame({"t": [0, 3, 8], "g": ["x", "x", "y"], "b": [100, 200, 300]})
    assert_lf_eq(
        left.lazy().join_asof(
            right.lazy(),
            on="t",
            by="g",
            strategy="backward",
        ),
        bl.LazyFrame(left).join_asof(
            bl.LazyFrame(right),
            on="t",
            by="g",
            strategy="backward",
        ),
    )


def test_join_asof_overlap_column_suffix() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"t": [0, 3, 8], "a": [100, 200, 300]})
    assert_lf_eq(
        left.lazy().join_asof(right.lazy(), on="t", strategy="backward"),
        bl.LazyFrame(left).join_asof(bl.LazyFrame(right), on="t", strategy="backward"),
    )


def test_join_without_keys_error() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="Either"):
        _ = bl.LazyFrame(left).join(bl.LazyFrame(right), how="inner").collect()


def test_join_on_and_left_right_on_error() -> None:
    left = pl.DataFrame({"id": [1, 2], "a": [10, 20]})
    right = pl.DataFrame({"id": [1, 2], "b": [100, 200]})
    with pytest.raises(ValueError, match="If `on` is specified"):
        _ = (
            bl
            .LazyFrame(left)
            .join(
                bl.LazyFrame(right),
                on="id",
                left_on="id",
                right_on="id",
                how="inner",
            )
            .collect()
        )


def test_join_asof_on_without_by() -> None:
    left = pl.DataFrame({"t": [1, 4, 9], "a": [1, 2, 3]})
    right = pl.DataFrame({"t": [0, 3, 8], "b": [100, 200, 300]})
    assert_lf_eq(
        left.lazy().join_asof(right.lazy(), on="t", strategy="backward"),
        bl.LazyFrame(left).join_asof(bl.LazyFrame(right), on="t", strategy="backward"),
    )
