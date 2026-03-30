"""Complex real-life query tests combining multiple pql features."""

from __future__ import annotations

import polars as pl

import pql

from ._utils import assert_lf_eq_pl

_EMPLOYEES = pl.DataFrame({
    "id": [1, 2, 3, 4, 5, 6, 7, 8],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi"],
    "department": [
        "Engineering",
        "Sales",
        "Engineering",
        "Sales",
        "Engineering",
        "HR",
        "HR",
        "Engineering",
    ],
    "age": [25, 30, 35, 28, 22, 40, 33, 29],
    "salary": [
        90000.0,
        60000.0,
        95000.0,
        55000.0,
        80000.0,
        70000.0,
        72000.0,
        85000.0,
    ],
    "years_exp": [2, 5, 10, 3, 1, 15, 8, 4],
    "is_active": [True, True, False, True, True, True, False, True],
    "score": [8.5, 7.0, 9.5, 6.5, 8.0, 7.5, 7.8, 8.2],
    "manager_id": [None, 1, 1, 2, 1, None, 6, 1],
})

_ORDERS = pl.DataFrame({
    "order_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "employee_id": [1, 2, 1, 3, 2, 4, 1, 5],
    "amount": [1200.0, 800.0, 450.0, 2300.0, 950.0, 300.0, 1800.0, 600.0],
    "region": ["North", "South", "North", "East", "South", "West", "North", "East"],
    "category": ["A", "B", "A", "C", "B", "A", "C", "B"],
})
_EMPLOYEES_LF = _EMPLOYEES.lazy()
_ORDERS_LF = _ORDERS.lazy()
_EMPLOYEES_PQL = pql.LazyFrame(_EMPLOYEES)
_ORDERS_PQL = pql.LazyFrame(_ORDERS)


def test_groupby_filter_having() -> None:
    """Group by department, compute aggregates, filter groups by condition."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .group_by("department")
        .agg(
            pql.col("salary").mean().alias("avg_salary"),
            pql.col("salary").max().alias("max_salary"),
            pql.col("id").count().alias("headcount"),
            pql.col("years_exp").sum().alias("total_exp"),
        )
        .filter(pql.col("headcount").gt(1))
        .sort("department"),
        _EMPLOYEES_LF
        .group_by("department")
        .agg(
            pl.col("salary").mean().alias("avg_salary"),
            pl.col("salary").max().alias("max_salary"),
            pl.col("id").count().alias("headcount"),
            pl.col("years_exp").sum().alias("total_exp"),
        )
        .filter(pl.col("headcount").gt(1))
        .sort("department"),
    )


def test_join_then_aggregate() -> None:
    """Join employees with orders, then aggregate per employee."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .join(_ORDERS_PQL, left_on="id", right_on="employee_id", how="inner")
        .group_by("name", "department")
        .agg(
            pql.col("amount").sum().alias("total_amount"),
            pql.col("order_id").count().alias("order_count"),
            pql.col("amount").mean().alias("avg_amount"),
        )
        .sort("name"),
        _EMPLOYEES_LF
        .join(_ORDERS_LF, left_on="id", right_on="employee_id", how="inner")
        .group_by("name", "department")
        .agg(
            pl.col("amount").sum().alias("total_amount"),
            pl.col("order_id").count().alias("order_count"),
            pl.col("amount").mean().alias("avg_amount"),
        )
        .sort("name"),
    )


def test_top_n_per_group_nested_window() -> None:
    """Top-2 earners per department: rank within partition, filter on rank."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .with_columns(
            pql
            .col("salary")
            .rank(method="ordinal", descending=True)
            .over("department")
            .alias("dept_rank"),
        )
        .filter(pql.col("dept_rank").le(2))
        .with_columns(
            pql.col("salary").mean().over("department").alias("top2_dept_avg"),
        )
        .select("name", "department", "salary", "dept_rank", "top2_dept_avg")
        .sort("department", "dept_rank"),
        _EMPLOYEES_LF
        .with_columns(
            pl
            .col("salary")
            .rank(method="ordinal", descending=True)
            .over("department")
            .alias("dept_rank"),
        )
        .filter(pl.col("dept_rank").le(2))
        .with_columns(
            pl.col("salary").mean().over("department").alias("top2_dept_avg"),
        )
        .select("name", "department", "salary", "dept_rank", "top2_dept_avg")
        .sort("department", "dept_rank"),
    )


def test_implode_list_ops_then_explode() -> None:
    """Implode salaries per dept, apply list ops, explode back."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .group_by("department")
        .agg(
            pql.col("salary").implode().alias("salaries"),
        )
        .with_columns(
            pql.col("salaries").list.len().alias("n"),
            pql.col("salaries").list.mean().alias("list_mean"),
            pql.col("salaries").list.max().alias("list_max"),
            pql.col("salaries").list.sort(descending=True).alias("salaries_sorted"),
        )
        .with_columns(
            pql.col("salaries_sorted").list.get(0).alias("highest"),
        )
        .sort("department"),
        _EMPLOYEES_LF
        .group_by("department")
        .agg(
            pl.col("salary").implode().alias("salaries"),
        )
        .with_columns(
            pl.col("salaries").list.len().alias("n"),
            pl.col("salaries").list.mean().alias("list_mean"),
            pl.col("salaries").list.max().alias("list_max"),
            pl.col("salaries").list.sort(descending=True).alias("salaries_sorted"),
        )
        .with_columns(
            pl.col("salaries_sorted").list.get(0).alias("highest"),
        )
        .sort("department"),
    )


def test_frame_explode_then_reaggregate() -> None:
    """Frame-level explode of an imploded list, then re-aggregate."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .group_by("department")
        .agg(pql.col("salary").implode().alias("salaries"))
        .explode("salaries")
        .filter(pql.col("salaries").gt(70000.0))
        .group_by("department")
        .agg(pql.col("salaries").mean().alias("high_earner_avg"))
        .sort("department"),
        _EMPLOYEES_LF
        .group_by("department")
        .agg(pl.col("salary").implode().alias("salaries"))
        .explode("salaries")
        .filter(pl.col("salaries").gt(70000.0))
        .group_by("department")
        .agg(pl.col("salaries").mean().alias("high_earner_avg"))
        .sort("department"),
    )


def test_expr_list_explode_in_agg() -> None:
    data = pl.DataFrame({
        "grp": ["a", "a", "b"],
        "vals": [[1, 2], [2, 3], [4, 5]],
        "arr": [[10, 20], [20, 30], [40, 50]],
    })
    assert_lf_eq_pl(
        pql
        .LazyFrame(data)
        .group_by("grp")
        .agg(
            pql.col("vals").list.sort().list.explode(),
            pql
            .col("arr")
            .cast(pql.Array(pql.Int64(), size=2))
            .arr.sort()
            .arr.explode(),
        )
        .sort("grp", "vals"),
        data
        .lazy()
        .group_by("grp")
        .agg(
            pl.col("vals").list.sort().list.explode(),
            pl
            .col("arr")
            .cast(pl.Array(pl.Int64(), shape=(2,)))
            .arr.sort()
            .arr.explode(),
        )
        .sort("grp", "vals"),
    )


def test_window_diff_pct_change_over_partition() -> None:
    """shift/diff/pct_change within department partition ordered by years_exp."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .sort("department", "years_exp")
        .with_columns(
            pql
            .col("salary")
            .shift(1)
            .over("department", order_by="years_exp")
            .alias("prev_salary"),
            pql
            .col("salary")
            .pct_change(1)
            .over("department", order_by="years_exp")
            .alias("salary_pct_change"),
        )
        .select(
            "name",
            "department",
            "years_exp",
            "salary",
            "prev_salary",
            "salary_pct_change",
        )
        .sort("department", "years_exp"),
        _EMPLOYEES_LF
        .sort("department", "years_exp")
        .with_columns(
            pl
            .col("salary")
            .shift(1)
            .over("department", order_by="years_exp")
            .alias("prev_salary"),
            pl
            .col("salary")
            .pct_change(1)
            .over("department", order_by="years_exp")
            .alias("salary_pct_change"),
        )
        .select(
            "name",
            "department",
            "years_exp",
            "salary",
            "prev_salary",
            "salary_pct_change",
        )
        .sort("department", "years_exp"),
    )


def test_join_window_then_list_agg() -> None:
    """Join, compute per-employee order rank within region, implode order amounts."""
    assert_lf_eq_pl(
        _ORDERS_PQL
        .with_columns(
            pql
            .col("amount")
            .rank(method="ordinal", descending=True)
            .over("region")
            .alias("region_rank"),
        )
        .join(
            _EMPLOYEES_PQL.select("id", "department", "salary"),
            left_on="employee_id",
            right_on="id",
            how="inner",
        )
        .group_by("region", "department")
        .agg(
            pql.col("amount").implode().alias("amounts"),
            pql.col("region_rank").min().alias("best_rank"),
        )
        .with_columns(
            pql.col("amounts").list.sort().alias("amounts"),
            pql.col("amounts").list.sum().alias("total"),
            pql.col("amounts").list.len().alias("n_orders"),
        )
        .sort("region", "department"),
        _ORDERS_LF
        .with_columns(
            pl
            .col("amount")
            .rank(method="ordinal", descending=True)
            .over("region")
            .alias("region_rank"),
        )
        .join(
            _EMPLOYEES_LF.select("id", "department", "salary"),
            left_on="employee_id",
            right_on="id",
            how="inner",
        )
        .group_by("region", "department")
        .agg(
            pl.col("amount").implode().alias("amounts"),
            pl.col("region_rank").min().alias("best_rank"),
        )
        .with_columns(
            pl.col("amounts").list.sort().alias("amounts"),
            pl.col("amounts").list.sum().alias("total"),
            pl.col("amounts").list.len().alias("n_orders"),
        )
        .sort("region", "department"),
    )


def test_multi_join_with_aggregation() -> None:
    """Inner join on two conditions, aggregate per region and category."""
    assert_lf_eq_pl(
        _EMPLOYEES_PQL
        .filter(pql.col("is_active"))
        .join(_ORDERS_PQL, left_on="id", right_on="employee_id", how="inner")
        .group_by("region", "category")
        .agg(
            pql.col("amount").sum().alias("total"),
            pql.col("order_id").count().alias("n_orders"),
            pql.col("salary").mean().alias("avg_seller_salary"),
        )
        .sort("region", "category"),
        _EMPLOYEES_LF
        .filter(pl.col("is_active"))
        .join(_ORDERS_LF, left_on="id", right_on="employee_id", how="inner")
        .group_by("region", "category")
        .agg(
            pl.col("amount").sum().alias("total"),
            pl.col("order_id").count().alias("n_orders"),
            pl.col("salary").mean().alias("avg_seller_salary"),
        )
        .sort("region", "category"),
    )
