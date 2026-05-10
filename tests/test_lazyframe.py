from __future__ import annotations

import polars as pl
import pytest
from pyochain import ResultUnwrapError
from sqlglot import exp

import belugas as bl
import belugas._plan as m  # noqa: PLC2701
import belugas.typing as t

from ._utils import assert_lf_eq

bl_age = bl.col("age")
bl_text = bl.col("text")
bl_salary = bl.col("salary")
pl_age = pl.col("age")
pl_salary = pl.col("salary")
_DF = bl.LazyFrame({
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
def lf() -> bl.LazyFrame:
    return _DF


def test_properties(lf: bl.LazyFrame) -> None:
    df = lf.collect()
    schema = lf._schema  # pyright: ignore[reportPrivateUsage]
    assert lf.width == df.width
    assert lf.columns.into(list) == df.columns
    assert set(schema.keys()) == set(df.columns)
    assert lf.shape == df.shape
    assert isinstance(lf.lazy(), pl.LazyFrame)
    assert isinstance(df, pl.DataFrame)


def test_schema_columns_follow_derived_frame(lf: bl.LazyFrame) -> None:
    renamed = lf.rename({"age": "years"})
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
    assert renamed.filter(bl.col("years").gt(20)).columns.into(list) == expected_cols


def test_show(lf: bl.LazyFrame) -> None:
    assert lf.show() is None


def test_empty_contexts(lf: bl.LazyFrame) -> None:
    pl_lf = lf.lazy()
    assert_lf_eq(pl_lf.with_columns(), lf.with_columns())
    assert_lf_eq(pl_lf.select(), lf.select())
    assert_lf_eq(pl_lf.select().with_columns(), lf.select().with_columns())


def test_repr(lf: bl.LazyFrame) -> None:
    assert repr(lf) == repr(lf.inner)


def test_clone(lf: bl.LazyFrame) -> None:
    df = lf.collect()
    cloned = lf.clone()
    assert_lf_eq(cloned.lazy(), lf)
    cloned_modified = cloned.filter(bl_age.gt(25)).collect()
    assert df.height != cloned_modified.height


def test_sql_query(lf: bl.LazyFrame) -> None:
    query = lf.filter(bl_age.gt(25)).select("name", "age")
    parsed = query.sql_query()
    assert parsed.sql() == query.inner.sql(dialect="duckdb")
    assert parsed.sql() != parsed.sql(pretty=True)
    assert parsed.tokenize() != parsed.sql(pretty=True)
    assert "SELECT" in parsed.sql()
    assert "WHERE" in parsed.sql()
    assert parsed.sql().upper().count("WHERE") == 1


@pytest.mark.parametrize("theme", t.Themes.__args__)
def test_sql_show(lf: bl.LazyFrame, theme: t.Themes) -> None:
    lf.select(bl_salary.cast(bl.Float64())).sql_query().show(theme)


def test_explain(lf: bl.LazyFrame) -> None:
    explained = lf.filter(bl_age.gt(25)).explain("standard")
    assert isinstance(explained, str)


@pytest.mark.parametrize(
    "cols", [["age"], ["age", "salary"], ["name", "age", "salary", "id"]]
)
def test_select(lf: bl.LazyFrame, cols: list[str]) -> None:
    assert_lf_eq(lf.lazy().select(cols), lf.select(cols))

    assert_lf_eq(
        lf.lazy().select(*cols, vals="id", other_vals=42),
        lf.select(*cols, vals="id", other_vals=42),
    )


def test_with_columns_name_prefix(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().with_columns(pl.col("name").name.prefix("new_")),
        lf.with_columns(bl.col("name").name.prefix("new_")),
    )


def test_select_unique_name_prefix(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().select(pl.col("department").unique().name.prefix("u_")),
        lf.select(bl.col("department").unique().name.prefix("u_")),
    )


@pytest.mark.parametrize("col", ["age", "salary"])
@pytest.mark.parametrize("descending", [True, False])
def test_sort_single_col(lf: bl.LazyFrame, col: str, descending: bool) -> None:
    assert_lf_eq(lf.lazy().sort(col), lf.sort(col))
    assert_lf_eq(
        lf.lazy().sort(col, descending=descending),
        lf.sort(col, descending=descending),
    )


sort_flags = [True, False, [True, False], [False, True], [False, False], [True, True]]


@pytest.mark.parametrize(
    "cols", [["age", "salary"], ["department", "age"], ["salary", "department"]]
)
@pytest.mark.parametrize("descending", sort_flags)
@pytest.mark.parametrize("nulls_last", sort_flags)
def test_sort_multiple_cols(
    lf: bl.LazyFrame,
    cols: list[str],
    descending: bool | list[bool],
    nulls_last: bool | list[bool],
) -> None:
    assert_lf_eq(
        lf.lazy().sort(cols, descending=descending, nulls_last=nulls_last),
        lf.sort(cols, descending=descending, nulls_last=nulls_last),
    )


def test_sort_errors(lf: bl.LazyFrame) -> None:
    with pytest.raises(ValueError, match="length of `descending`"):
        _ = lf.sort("age", "salary", descending=[True])

    with pytest.raises(ValueError, match="length of `nulls_last`"):
        _ = lf.sort("age", "salary", nulls_last=[True, False, True])


def test_limit(lf: bl.LazyFrame) -> None:
    """Affected by the buggy `lazy` duckdb to polars conversions."""
    assert_lf_eq(lf.collect().lazy().sort("id").limit(3), lf.sort("id").limit(3))


@pytest.mark.parametrize("offset", [1, -2, -4, -10])
@pytest.mark.parametrize("length", [2, None, 4, 0])
def test_slice(lf: bl.LazyFrame, offset: int, length: int | None) -> None:
    assert_lf_eq(
        lf.lazy().slice(offset, length),
        lf.slice(offset, length),
    )


def test_slice_errors(lf: bl.LazyFrame) -> None:
    with pytest.raises(ValueError, match="negative slice lengths"):
        _ = lf.slice(0, -1)
    with pytest.raises(ValueError, match="negative slice lengths"):
        _ = lf.lazy().slice(0, -1)


@pytest.mark.parametrize("n", [0, 2, 5])
def test_tail(lf: bl.LazyFrame, n: int) -> None:
    assert_lf_eq(lf.lazy().tail(n), lf.tail(n))
    with pytest.raises(ValueError, match="`n` must be greater than or equal to 0"):
        _ = lf.tail(-1)
    with pytest.raises(OverflowError, match="can't convert negative int to unsigned"):
        _ = lf.lazy().tail(-1)


def test_last(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().last(), lf.last())


def test_filter(lf: bl.LazyFrame) -> None:
    salary_bl = bl_salary.mul(12).gt(600000)
    salary_pl = pl_salary.mul(12).gt(600000)
    age_bl = bl_age.lt(50)
    age_pl = pl_age.lt(50)
    assert_lf_eq(lf.lazy().filter(salary_pl), lf.filter(salary_bl))
    assert_lf_eq(
        lf.lazy().filter(salary_pl, age_pl),
        lf.filter(salary_bl, age_bl),
    )
    assert_lf_eq(
        lf.lazy().filter([salary_pl, age_pl]),
        lf.filter([salary_bl, age_bl]),
    )
    assert_lf_eq(
        lf.lazy().filter(age_pl, is_active=True, department="Sales"),
        lf.filter(age_bl, is_active=True, department="Sales"),
    )


def test_first(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().first(), lf.first())


def test_count(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().select("id").count(), lf.select("id").count())


def test_sum(lf: bl.LazyFrame) -> None:
    cols = ("age", "salary")
    assert_lf_eq(lf.lazy().select(cols).sum(), lf.select(cols).sum())


def test_mean(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().select("age").mean(), lf.select("age").mean())


def test_median(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().select("salary").median(), lf.select("salary").median())


def test_min(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().select("age").min(), lf.select("age").min())


def test_max(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().select("age").max(), lf.select("age").max())


def test_null_count(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().select("value").null_count(), lf.select("value").null_count()
    )


def test_cast(lf: bl.LazyFrame) -> None:
    cols = ("age", "id")
    assert_lf_eq(
        lf.lazy().select(cols).cast({"age": pl.Float64}),
        lf.select(cols).cast({"age": bl.Float64()}),
    )
    assert_lf_eq(
        lf.lazy().select(cols).cast(pl.String), lf.select(cols).cast(bl.String())
    )


def test_pipe(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().pipe(lambda df: df), lf.pipe(lambda lf: lf))


@pytest.mark.parametrize("cols", [["age"], ["age", "salary"]])
def test_drop_single_column(lf: bl.LazyFrame, cols: list[str]) -> None:
    assert_lf_eq(lf.lazy().drop(*cols), lf.drop(*cols))


@pytest.mark.parametrize(
    "mapping", [{"age": "years"}, {"age": "years", "name": "full_name"}]
)
def test_rename(lf: bl.LazyFrame, mapping: dict[str, str]) -> None:
    result = lf.rename(mapping)
    expected = lf.lazy().rename(mapping)
    assert_lf_eq(expected, result)


def test_with_columns_star_exprs(lf: bl.LazyFrame) -> None:
    cols = lf._schema  # pyright: ignore[reportPrivateUsage]

    def _plan(expr: bl.Expr) -> exp.Star | None:
        ast, _ = m.with_columns(cols, expr, (), {})
        return ast.find(exp.Star)

    assert _plan(bl_age) is None
    assert _plan(bl_age.alias("age2")) is not None


def test_with_columns_single_expr(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().with_columns(pl_age.mul(2), x=42),
        lf.with_columns(bl_age.mul(2), x=42),
    )

    assert_lf_eq(
        lf.lazy().with_columns(pl_age.mul(2), pl_salary.truediv(12)),
        lf.with_columns(bl_age.mul(2), bl_salary.truediv(12)),
    )


def test_fill_nan_with_value() -> None:
    df = bl.LazyFrame({"a": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
    result = bl.LazyFrame(df).fill_nan(0.0)
    expected = df.lazy().fill_nan(0.0)
    assert_lf_eq(expected, result)


def test_fill_null_with_value(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().select("value", "age").fill_null(0),
        lf.select("value", "age").fill_null(0),
    )


@pytest.mark.parametrize("strategy", t.FillNullStrategy.__args__)
def test_fill_null_with_strategy(strategy: t.FillNullStrategy) -> None:
    df = bl.LazyFrame({"a": [1.0, None, None, 4.0, None]})
    assert_lf_eq(
        df.lazy().fill_null(strategy=strategy), df.fill_null(strategy=strategy)
    )


@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_fill_null_with_strategy_limit(strategy: t.FillNullStrategy) -> None:
    df = bl.LazyFrame({"a": [1, None, None, 4, None]})
    assert_lf_eq(
        df.lazy().fill_null(strategy=strategy, limit=1),
        df.fill_null(strategy=strategy, limit=1),
    )


def test_fill_null_with_value_limit_error() -> None:
    df = bl.LazyFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        _ = df.fill_null(0, limit=1)


@pytest.mark.parametrize("strategy", ["min", "max", "mean", "zero", "one"])
def test_fill_null_with_non_directional_strategy_limit_error(
    strategy: t.FillNullStrategy,
) -> None:
    df = bl.LazyFrame({"a": [1.0, None, None, 4.0]})
    with pytest.raises(ValueError, match="can only specify `limit`"):
        _ = df.fill_null(strategy=strategy, limit=1)


def test_fill_null_with_negative_limit_error() -> None:
    df = bl.LazyFrame({"a": [1.0, None, None, 4.0]})
    msg = "Can't process negative `limit` value for fill_null"
    with pytest.raises(ResultUnwrapError, match=msg):
        _ = df.fill_null(strategy="forward", limit=-1)


@pytest.mark.parametrize("n", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("fill_value", [None, 0, 999])
def test_shift(n: int, fill_value: int | None) -> None:
    df = bl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    assert_lf_eq(
        df.lazy().shift(n, fill_value=fill_value), df.shift(n, fill_value=fill_value)
    )


@pytest.mark.parametrize("ddof", [0, 1])
def test_std_var(ddof: int) -> None:
    df = bl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    assert_lf_eq(df.lazy().select("a").std(ddof=ddof), df.select("a").std(ddof=ddof))
    assert_lf_eq(df.lazy().select("a").var(ddof=ddof), df.select("a").var(ddof=ddof))


@pytest.mark.parametrize("by", [["age", "department"], ["age", "salary"]])
@pytest.mark.parametrize(
    "reverse", [[True, False], [False, True], [False, False], [True, True]]
)
@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_bottom_k(
    lf: bl.LazyFrame, k: int, by: list[str] | str, reverse: bool | list[bool]
) -> None:
    """Affected by the buggy `lazy` duckdb to polars conversions."""
    assert_lf_eq(
        lf.lazy().top_k(k, by=by, reverse=reverse),
        lf.top_k(k, by=by, reverse=reverse),
    )
    assert_lf_eq(
        lf.lazy().bottom_k(k, by=by, reverse=reverse),
        lf.bottom_k(k, by=by, reverse=reverse),
    )


@pytest.mark.parametrize("by", ["age", "salary"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("k", [1, 3, 5])
def test_top_bottom_k_single(
    lf: bl.LazyFrame, k: int, by: list[str] | str, reverse: bool | list[bool]
) -> None:
    assert_lf_eq(
        lf.collect().lazy().top_k(k, by=by, reverse=reverse),
        lf.top_k(k, by=by, reverse=reverse),
    )
    assert_lf_eq(
        lf.collect().lazy().bottom_k(k, by=by, reverse=reverse),
        lf.bottom_k(k, by=by, reverse=reverse),
    )


@pytest.mark.parametrize("seed", [0, 42, 12345])
def test_hash_seed(seed: int) -> None:
    df = bl.LazyFrame({"text": ["apple", "banana", "apple"]})
    hashes = df.select(bl_text.hash(seed)).collect().get_column("text").to_list()
    # Check that same input produces same hash
    assert hashes[0] == hashes[2], "Same input should produce same hash"
    # Different seed should produce different hash
    other_seed = 1 if seed == 0 else 0
    hashes_other = (
        df.select(bl_text.hash(other_seed)).collect().get_column("text").to_list()
    )
    assert hashes[0] != hashes_other[0], (
        "Different seeds should produce different hashes"
    )


@pytest.mark.parametrize("subset", [None, "value"])
def test_drop_nulls(lf: bl.LazyFrame, subset: list[str]) -> None:
    assert_lf_eq(lf.lazy().drop_nulls(subset), lf.drop_nulls(subset))


@pytest.mark.parametrize("columns", ["vals1", "vals2", ["vals1", "vals2"]])
def test_explode(columns: list[str] | str) -> None:
    data = bl.LazyFrame({
        "id": [1, 2, 3],
        "vals1": [[10, 11], [], None, [70]],
        "vals2": [[100, 110], [], None, [700]],
    })
    assert_lf_eq(data.lazy().explode(columns), data.explode(columns))


def test_gather_every(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().gather_every(2, offset=1),
        lf.gather_every(2, offset=1),
    )


def test_with_row_index(lf: bl.LazyFrame) -> None:
    assert_lf_eq(
        lf.lazy().sort("sex").with_row_index("row_num"),
        lf.with_row_index("row_num", order_by="sex"),
    )


def test_describe(lf: bl.LazyFrame) -> None:
    assert lf.select("age", "salary").describe().collect().height > 0


def test_unnest() -> None:
    df = bl.LazyFrame({
        "id": [1, 2],
        "nested": [{"a": 10, "b": 100}, {"a": 20, "b": 200}],
    })
    assert_lf_eq(df.lazy().unnest("nested"), df.unnest("nested"))


pl_nested = pl.LazyFrame({
    "id": [1, 2, 3],
    "nested": [
        {"a": 10, "b": 100},
        {"a": 20, "b": 200},
        {"a": 30, "b": 300},
    ],
})
bl_unnested = bl.LazyFrame(pl_nested).unnest("nested")
pl_unnested = pl_nested.unnest("nested")


def test_unnest_columns_property() -> None:
    assert bl_unnested.columns.into(list) == pl_unnested.collect_schema().names()


def test_unnest_then_select_all() -> None:
    assert_lf_eq(pl_unnested.select(pl.all()), bl_unnested.select(bl.all()))


def test_unnest_then_with_columns_overlap() -> None:
    assert_lf_eq(
        pl_unnested.with_columns(a=pl.col("a").add(1)),
        bl_unnested.with_columns(a=bl.col("a").add(1)),
    )


def test_unnest_then_drop() -> None:
    assert_lf_eq(pl_unnested.drop("b"), bl_unnested.drop("b"))


def test_unnest_then_rename() -> None:
    assert_lf_eq(pl_unnested.rename({"a": "x"}), bl_unnested.rename({"a": "x"}))


def test_unnest_then_filter_then_select_all() -> None:
    assert_lf_eq(
        pl_unnested.filter(pl.col("a").gt(10)).select(pl.all()),
        bl_unnested.filter(bl.col("a").gt(10)).select(bl.all()),
    )


@pytest.mark.parametrize("strategy", t.UniqueKeepStrategy.__args__)
@pytest.mark.parametrize("subset", [None, "department", ["department", "sex"]])
def test_unique(
    lf: bl.LazyFrame, strategy: t.UniqueKeepStrategy, subset: str | list[str] | None
) -> None:
    assert_lf_eq(
        lf.lazy().unique(subset=subset, keep=strategy).sort("id"),
        lf.unique(subset=subset, keep=strategy, order_by="id"),
    )


@pytest.mark.parametrize("strategy", ["first", "last"])
def test_unique_without_order_by_error(strategy: t.UniqueKeepStrategy) -> None:
    df = bl.LazyFrame({"a": [1, 1, 2], "b": [1, 2, 3]})
    with pytest.raises(ValueError, match="`order_by` must be specified"):
        _ = df.unique(keep=strategy)


def test_unique_with_multiple_order_by() -> None:
    df = bl.LazyFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": [10, 20, 30]})
    result = df.unique(keep="first", order_by=["a", "b"]).collect()
    assert result.height > 0


def test_select_with_named_expr() -> None:
    df = bl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.select("a", doubled=bl.col("b").mul(2))
    expected = df.lazy().select("a", doubled=pl.col("b").mul(2))
    assert_lf_eq(expected, result)


@pytest.mark.parametrize("quantile", [0.0, 0.5, 1.0])
def test_quantile(lf: bl.LazyFrame, quantile: float) -> None:
    assert_lf_eq(
        lf.lazy().select("age", "salary").quantile(quantile),
        lf.select("age", "salary").quantile(quantile),
    )


def test_reverse(lf: bl.LazyFrame) -> None:
    assert_lf_eq(lf.lazy().reverse(), lf.reverse())


@pytest.mark.parametrize("subset", [["a"], ["b"], ["a", "b"]])
def test_drop_nans(subset: list[str]) -> None:
    """We convert data from polars to bl instead of the other way around in this test.

    we do this because `DuckDB` converts `NaN` to `NULL` at ingestion,
    so `NaN` values never survive in `DuckDB` relations.
    """
    df = pl.DataFrame({
        "a": [1.0, float("nan"), 3.0, 4.0],
        "b": [float("nan"), 2.0, 3.0, 4.0],
    })
    assert_lf_eq(
        df.lazy().drop_nans(subset=subset),
        df.pipe(bl.LazyFrame).drop_nans(subset=subset),
    )
