from __future__ import annotations

import polars as pl
import pyochain as pc
import pytest
from polars.testing import assert_frame_equal

import pql
import pql._typing as t  # noqa: PLC2701

from ._utils import assert_lf_eq_pl

_DF = pl.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "sex": ["F", "M", "M", "M", "F"],
    "age": [25, 30, 35, 28, 22],
    "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
    "department": [
        "Engineering",
        "Sales",
        "Engineering",
        "Sales",
        "Engineering",
    ],
    "is_active": [True, True, False, True, True],
    "value": [10.0, None, 30.0, None, 50.0],
    "category": ["A", "B", None, "A", "B"],
})


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return _DF


def test_properties(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert lf.width == sample_df.width
    assert lf.columns.into(list) == sample_df.columns
    assert set(lf.schema.keys()) == set(sample_df.columns)
    assert lf.schema == lf.collect_schema()
    assert lf.shape == sample_df.shape
    assert isinstance(lf.lazy(), pl.LazyFrame)


def test_schema_columns_follow_derived_frame(sample_df: pl.DataFrame) -> None:
    renamed = pql.LazyFrame(sample_df).rename({"age": "years"})
    expected_cols = [
        "id",
        "name",
        "sex",
        "years",
        "salary",
        "department",
        "is_active",
        "value",
        "category",
    ]
    assert renamed.columns.into(list) == expected_cols
    assert renamed.filter(pql.col("years").gt(20)).columns.into(list) == expected_cols


def test_show(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert lf.show() is None


def test_empty_frame(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).select([]), sample_df.lazy().select([]))
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).with_columns(pql.col("age").sum()).select([]),
        sample_df.lazy().with_columns(pl.col("age").sum()).select(),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).drop("age").select([]),
        sample_df.lazy().drop("age").select(),
    )


def test_repr(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    assert repr(lf) == repr(lf.inner())


def test_clone(sample_df: pl.DataFrame) -> None:
    lf = pql.LazyFrame(sample_df)
    cloned = lf.clone()
    assert_lf_eq_pl(lf, cloned.lazy())
    cloned_modified = cloned.filter(pql.col("age").gt(25)).collect()
    assert lf.collect().height != cloned_modified.height


def test_sql_query(sample_df: pl.DataFrame) -> None:
    parsed = (
        pql
        .LazyFrame(sample_df)
        .filter(pql.col("age").gt(25))
        .select("name", "age")
        .sql_query()
    )
    assert parsed.raw != parsed.prettify().raw
    assert parsed.tokenize() != parsed.prettify().tokenize()
    assert "SELECT" in parsed.raw
    assert "WHERE" in parsed.raw
    assert parsed.raw.upper().count("WHERE") == 1


@pytest.mark.parametrize("theme", t.Themes.__args__)
def test_sql_show(sample_df: pl.DataFrame, theme: t.Themes) -> None:
    pql.LazyFrame(sample_df).select(
        pql.col("salary").cast(pql.Float64())
    ).sql_query().show(theme)


def test_explain(sample_df: pl.DataFrame) -> None:
    explained = (
        pql.LazyFrame(sample_df).filter(pql.col("age").gt(25)).explain("standard")
    )
    assert isinstance(explained, str)


def test_select_single_column(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select("name"), sample_df.lazy().select("name")
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select(pql.col("name")),
        sample_df.lazy().select(pl.col("name")),
    )

    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select("name", "age", "salary", "id"),
        sample_df.lazy().select("name", "age", "salary", "id"),
    )

    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select(
            pql.col("name"),
            pql.col("salary").mul(1.1).alias("salary_increase"),
            vals=pql.col("id"),
            other_vals=42,
        ),
        sample_df.lazy().select(
            pl.col("name"),
            pl.col("salary").mul(1.1).alias("salary_increase"),
            vals=pl.col("id"),
            other_vals=42,
        ),
    )


def test_with_columns_name_prefix(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).with_columns(pql.col("name").name.prefix("new_")),
        sample_df.lazy().with_columns(pl.col("name").name.prefix("new_")),
    )


def test_select_unique_name_prefix(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select(
            pql.col("department").unique().name.prefix("u_")
        ),
        sample_df.lazy().select(pl.col("department").unique().name.prefix("u_")),
    )


def test_sort(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).sort("age"), sample_df.lazy().sort("age"))
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).sort("salary", descending=True),
        sample_df.lazy().sort("salary", descending=True),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).sort(
            pql.col("department"), "age", descending=[False, True]
        ),
        sample_df.lazy().sort(pl.col("department"), "age", descending=[False, True]),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).sort(
            "department", "age", descending=True, nulls_last=True
        ),
        sample_df.lazy().sort("department", "age", descending=True, nulls_last=True),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).sort("age", nulls_last=True),
        sample_df.lazy().sort("age", nulls_last=True),
    )

    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).sort("age", "department", nulls_last=[True, False]),
        sample_df.lazy().sort("age", "department", nulls_last=[True, False]),
    )
    with pytest.raises(ValueError, match="length of `descending`"):
        _ = pql.LazyFrame(sample_df).sort("age", "salary", descending=[True])

    with pytest.raises(ValueError, match="length of `nulls_last`"):
        _ = pql.LazyFrame(sample_df).sort(
            "age", "salary", nulls_last=[True, False, True]
        )


def test_limit(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).sort("id").limit(3),
        sample_df.lazy().sort("id").limit(3),
    )


def test_slice(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).slice(1, 3), sample_df.lazy().slice(1, 3))
    assert_lf_eq_pl(pql.LazyFrame(sample_df).slice(-2), sample_df.lazy().slice(-2))
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).slice(-4, 2), sample_df.lazy().slice(-4, 2)
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).slice(-2, 0), sample_df.lazy().slice(-2, 0)
    )
    with pytest.raises(ValueError, match="negative slice lengths"):
        _ = pql.LazyFrame(sample_df).slice(0, -1)
    with pytest.raises(ValueError, match="negative slice lengths"):
        _ = sample_df.lazy().slice(0, -1)


def test_tail(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).tail(2), sample_df.lazy().tail(2))
    assert_lf_eq_pl(pql.LazyFrame(sample_df).tail(0), sample_df.lazy().tail(0))
    with pytest.raises(ValueError, match="`n` must be greater than or equal to 0"):
        _ = pql.LazyFrame(sample_df).tail(-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = sample_df.lazy().tail(-1)


def test_last(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).last(), sample_df.lazy().last())


def test_filter(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).filter(pql.col("salary").mul(12).gt(600000)),
        sample_df.lazy().filter(pl.col("salary").mul(12).gt(600000)),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).filter(
            pql.col("salary").mul(12).gt(600000), pql.col("age").lt(50)
        ),
        sample_df.lazy().filter(
            pl.col("salary").mul(12).gt(600000), pl.col("age").lt(50)
        ),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).filter([
            pql.col("salary").mul(12).gt(600000),
            pql.col("age").lt(50),
        ]),
        sample_df.lazy().filter([
            pl.col("salary").mul(12).gt(600000),
            pl.col("age").lt(50),
        ]),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).filter(
            pql.col("age").gt(20), is_active=True, department="Sales"
        ),
        sample_df.lazy().filter(
            pl.col("age").gt(20), is_active=True, department="Sales"
        ),
    )


def test_first(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).first(), sample_df.lazy().first())


def test_count(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("id")).count()
    expected = sample_df.lazy().select(pl.col("id")).count()
    assert_lf_eq_pl(result, expected)


def test_sum(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age"), pql.col("salary")).sum()
    expected = sample_df.lazy().select(pl.col("age"), pl.col("salary")).sum()
    assert_lf_eq_pl(result, expected)


def test_mean(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).mean()
    expected = sample_df.lazy().select(pl.col("age")).mean()
    assert_lf_eq_pl(result, expected)


def test_median(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("salary")).median()
    expected = sample_df.lazy().select(pl.col("salary")).median()
    assert_lf_eq_pl(result, expected)


def test_min(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).min()
    expected = sample_df.lazy().select(pl.col("age")).min()
    assert_lf_eq_pl(result, expected)


def test_max(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("age")).max()
    expected = sample_df.lazy().select(pl.col("age")).max()
    assert_lf_eq_pl(result, expected)


def test_null_count(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select(pql.col("value")).null_count()
    expected = sample_df.lazy().select(pl.col("value")).null_count()
    assert_lf_eq_pl(result, expected)


def test_top_k(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).top_k(3, by="age")
    expected = sample_df.lazy().top_k(3, by="age")
    assert_lf_eq_pl(result, expected)


def test_bottom_k(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).bottom_k(3, by="age"),
        sample_df.lazy().bottom_k(3, by="age"),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).bottom_k(3, by=["age", "salary"]),
        sample_df.lazy().bottom_k(3, by=["age", "salary"]),
    )


def test_cast(sample_df: pl.DataFrame) -> None:
    assert_frame_equal(
        pql
        .LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast({"age": pql.Float64()})
        .collect(),
        sample_df
        .lazy()
        .select(pl.col("age"), pl.col("id"))
        .cast({"age": pl.Float64})
        .collect(),
    )
    assert_frame_equal(
        pql
        .LazyFrame(sample_df)
        .select(pql.col("age"), pql.col("id"))
        .cast(pql.String())
        .collect(),
        sample_df.lazy().select(pl.col("age"), pl.col("id")).cast(pl.String).collect(),
    )


def test_pipe(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).pipe(lambda lf: lf),
        sample_df.lazy().pipe(lambda df: df),
    )


@pytest.mark.parametrize("cols", [["age"], ["age", "salary"]])
def test_drop_single_column(sample_df: pl.DataFrame, cols: list[str]) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).drop(*cols), sample_df.lazy().drop(*cols))


@pytest.mark.parametrize(
    "mapping", [{"age": "years"}, {"age": "years", "name": "full_name"}]
)
def test_rename(sample_df: pl.DataFrame, mapping: dict[str, str]) -> None:
    result = pql.LazyFrame(sample_df).rename(mapping)
    expected = sample_df.lazy().rename(mapping)
    assert_lf_eq_pl(result, expected)


def test_with_columns_add_only_uses_star(sample_df: pl.DataFrame) -> None:
    """Add-only with_columns must generate SELECT * instead of enumerating existing columns."""
    parsed = (
        pql
        .LazyFrame(sample_df)
        .with_columns(pql.col("age").mul(2).alias("age2"))
        .sql_query()
    )
    outermost_select = parsed.raw.split("FROM")[0]
    assert "SELECT *" in outermost_select


def test_with_columns_override_enumerates_columns(sample_df: pl.DataFrame) -> None:
    """Override with_columns must enumerate columns (no SELECT *) to preserve order."""
    parsed = (
        pql
        .LazyFrame(sample_df)
        .with_columns(pql.col("age").mul(2).alias("age"))
        .sql_query()
    )
    outermost_select = parsed.raw.split("FROM")[0]
    assert "SELECT *" not in outermost_select


def test_with_columns_single_expr(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        (
            pql.LazyFrame(sample_df).with_columns(
                pql.col("age").mul(2).alias("age_doubled"), x=42
            )
        ),
        (
            sample_df.lazy().with_columns(
                pl.col("age").mul(2).alias("age_doubled"), x=42
            )
        ),
    )

    assert_lf_eq_pl(
        (
            pql.LazyFrame(sample_df).with_columns(
                pql.col("age").mul(2).alias("age_doubled"),
                pql.col("salary").truediv(12).alias("monthly_salary"),
            )
        ),
        (
            sample_df.lazy().with_columns(
                (pl.col("age") * 2).alias("age_doubled"),
                (pl.col("salary") / 12).alias("monthly_salary"),
            )
        ),
    )


def test_std_default(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select("age").std()
    expected = sample_df.lazy().select("age").std()
    assert_lf_eq_pl(result, expected)


def test_var_default(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).select("age").var()
    expected = sample_df.lazy().select("age").var()
    assert_lf_eq_pl(result, expected)


def test_fill_nan_with_value() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
    result = pql.LazyFrame(df).fill_nan(0.0)
    expected = df.lazy().fill_nan(0.0)
    assert_lf_eq_pl(result, expected)


def test_fill_null_with_value(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select("value", "age").fill_null(0),
        sample_df.lazy().select("value", "age").fill_null(0),
    )


@pytest.mark.parametrize("strategy", pql.sql.typing.FillNullStrategy.__args__)
def test_fill_null_with_strategy(strategy: pql.sql.typing.FillNullStrategy) -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0, None]})
    assert_lf_eq_pl(
        pql.LazyFrame(df).fill_null(strategy=strategy),
        df.lazy().fill_null(strategy=strategy),
    )


@pytest.mark.parametrize(
    "strategy",
    ["forward", "backward"],
)
def test_fill_null_with_strategy_limit(
    strategy: pql.sql.typing.FillNullStrategy,
) -> None:
    df = pl.DataFrame({"a": [1, None, None, 4, None]})
    assert_lf_eq_pl(
        pql.LazyFrame(df).fill_null(strategy=strategy, limit=1),
        df.lazy().fill_null(strategy=strategy, limit=1),
    )


def test_fill_null_with_value_limit_error() -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        _ = pql.LazyFrame(df).fill_null(0, limit=1)


@pytest.mark.parametrize("strategy", ["min", "max", "mean", "zero", "one"])
def test_fill_null_with_non_directional_strategy_limit_error(
    strategy: pql.sql.typing.FillNullStrategy,
) -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        _ = pql.LazyFrame(df).fill_null(strategy=strategy, limit=1)


def test_fill_null_with_negative_limit_error() -> None:
    df = pl.DataFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(
        pc.ResultUnwrapError, match="Can't process negative `limit` value for fill_null"
    ):
        _ = pql.LazyFrame(df).fill_null(strategy="forward", limit=-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = df.lazy().fill_null(strategy="forward", limit=-1)


def test_shift() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_lf_eq_pl(pql.LazyFrame(df).shift(2), df.lazy().shift(2))
    assert_lf_eq_pl(pql.LazyFrame(df).shift(-2), df.lazy().shift(-2))

    assert_lf_eq_pl(
        pql.LazyFrame(df).shift(2, fill_value=0),
        df.lazy().shift(2, fill_value=0),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(df).shift(1, fill_value=999),
        df.lazy().shift(1, fill_value=999),
    )


def test_std_var_ddof() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    assert_lf_eq_pl(
        pql.LazyFrame(df).select("a").std(ddof=0),
        df.lazy().select("a").std(ddof=0),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(df).select("a").var(ddof=0),
        df.lazy().select("a").var(ddof=0),
    )


def test_top_k_with_multiple_cols(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).top_k(3, by=["department", "age"]),
        sample_df.lazy().top_k(3, by=["department", "age"]),
    )
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).top_k(2, by="age", reverse=True),
        sample_df.lazy().top_k(2, by="age", reverse=True),
    )
    assert_lf_eq_pl(
        (
            pql.LazyFrame(sample_df).top_k(
                3, by=["department", "age"], reverse=[False, True]
            )
        ),
        (sample_df.lazy().top_k(3, by=["department", "age"], reverse=[False, True])),
    )


def test_bottom_k_with_multiple_cols(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).bottom_k(3, by=["department", "age"]),
        sample_df.lazy().bottom_k(3, by=["department", "age"]),
    )


def test_bottom_k_with_reverse(sample_df: pl.DataFrame) -> None:
    result = pql.LazyFrame(sample_df).bottom_k(2, by="age", reverse=True)
    expected = sample_df.lazy().bottom_k(2, by="age", reverse=True)
    assert_lf_eq_pl(result, expected)


def test_hash_seed0() -> None:
    df = pl.DataFrame({"text": ["apple", "banana", "apple"]})
    result = pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect()
    # Check that same input produces same hash
    hashes = result["h"].to_list()
    assert hashes[0] == hashes[2], "Same input should produce same hash"


def test_hash_seed42() -> None:
    df = pl.DataFrame({"text": ["apple", "banana", "apple"]})
    result = (
        pql.LazyFrame(df).select(pql.col("text").hash(seed=42).alias("h")).collect()
    )
    # Check that same input produces same hash with different seed
    hashes = result["h"].to_list()
    assert hashes[0] == hashes[2], "Same input should produce same hash"
    # Different seed should produce different hash
    result_seed0 = (
        pql.LazyFrame(df).select(pql.col("text").hash(seed=0).alias("h")).collect()
    )
    assert hashes[0] != result_seed0["h"][0], (
        "Different seeds should produce different hashes"
    )


@pytest.mark.parametrize("subset", [None, "value"])
def test_drop_nulls(sample_df: pl.DataFrame, subset: list[str]) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).drop_nulls(subset), sample_df.lazy().drop_nulls(subset)
    )


def test_explode() -> None:
    data = pl.DataFrame({"id": [1, 2, 3], "vals": [[10, 11], None, []]})
    assert_lf_eq_pl(
        pql.LazyFrame(data).explode("vals"),
        data.lazy().explode("vals"),
    )
    data = pl.DataFrame({
        "id": [1, 2, 3, 4],
        "vals1": [[10, 11], [], None, [70]],
        "vals2": [[100, 110], [], None, [700]],
    })
    assert_lf_eq_pl(
        pql.LazyFrame(data).explode("vals1", "vals2"),
        data.lazy().explode("vals1", "vals2"),
    )


def test_gather_every(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).gather_every(2, offset=1),
        sample_df.lazy().gather_every(2, offset=1),
    )


def test_with_row_index(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).with_row_index("row_num", order_by="sex"),
        sample_df.lazy().sort("sex").with_row_index("row_num"),
    )


def test_describe(sample_df: pl.DataFrame) -> None:
    assert (
        pql.LazyFrame(sample_df).select("age", "salary").describe().collect().height > 0
    )


def test_unnest() -> None:
    df = pl.DataFrame({
        "id": [1, 2],
        "nested": [{"a": 10, "b": 100}, {"a": 20, "b": 200}],
    })
    assert_lf_eq_pl(
        pql.LazyFrame(df).unnest("nested"),
        df.lazy().unnest("nested"),
    )


@pytest.mark.parametrize("strategy", t.UniqueKeepStrategy.__args__)
def test_unique(sample_df: pl.DataFrame, strategy: t.UniqueKeepStrategy) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).unique(
            subset=["department"], keep=strategy, order_by="id"
        ),
        sample_df.lazy().unique(subset=["department"], keep=strategy).sort("id"),
    )


@pytest.mark.parametrize("strategy", ["first", "last"])
def test_unique_without_order_by_error(strategy: t.UniqueKeepStrategy) -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="`order_by` must be specified"):
        _ = pql.LazyFrame(df).unique(keep=strategy)


def test_unique_with_multiple_order_by() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [10, 20, 30]})
    result = pql.LazyFrame(df).unique(keep="first", order_by=["a", "b"]).collect()
    assert result.height > 0


def test_select_with_named_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = pql.LazyFrame(df).select(pql.col("a"), doubled=pql.col("b").mul(2))
    expected = df.lazy().select(pl.col("a"), doubled=pl.col("b").mul(2))
    assert_lf_eq_pl(result, expected)


def test_quantile(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(
        pql.LazyFrame(sample_df).select("age", "salary").quantile(0.5),
        sample_df.lazy().select("age", "salary").quantile(0.5),
    )


def test_reverse(sample_df: pl.DataFrame) -> None:
    assert_lf_eq_pl(pql.LazyFrame(sample_df).reverse(), sample_df.lazy().reverse())


@pytest.mark.parametrize("subset", [["a"], ["b"], ["a", "b"]])
def test_drop_nans(subset: list[str]) -> None:
    df = pl.DataFrame({
        "a": [1.0, float("nan"), 3.0, 4.0],
        "b": [float("nan"), 2.0, 3.0, 4.0],
    })
    assert_lf_eq_pl(
        pql.LazyFrame(df).drop_nans(subset=subset), df.lazy().drop_nans(subset=subset)
    )
