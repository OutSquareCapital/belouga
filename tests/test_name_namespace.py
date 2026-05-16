import polars as pl
import pytest

import belugas as bl

from ._utils import assert_eq

bl_x = bl.col("x")
bl_s = bl.col("s")
bl_salary = bl.col("salary").name
pl_x = pl.col("x")
pl_s = pl.col("s")
pl_salary = pl.col("salary").name


def test_name_keep_after_alias() -> None:
    assert_eq(bl_x.alias("renamed").name.keep(), pl_x.alias("renamed").name.keep())


def test_name_map() -> None:
    assert_eq(
        bl_x.name.map(lambda name: f"mapped_{name}"),
        pl_x.name.map(lambda name: f"mapped_{name}"),
    )


def test_name_prefix_suffix() -> None:
    assert_eq(
        bl_x.name.prefix("pre_").name.suffix("_suf"),
        pl_x.name.prefix("pre_").name.suffix("_suf"),
    )


def test_name_case_transform() -> None:
    assert_eq(
        bl_x.name.to_uppercase().name.to_lowercase(),
        pl_x.name.to_uppercase().name.to_lowercase(),
    )


case_skip = pytest.mark.xfail(
    reason="This test is currently failing due to an issue with column name traduction to polars dataframe/pandas"
)


@case_skip
def test_name_to_uppercase_all() -> None:
    assert_eq(bl.all().name.to_uppercase(), pl.all().name.to_uppercase())


@pytest.mark.parametrize("literal", [True, False])
def test_name_replace(literal: bool) -> None:
    assert_eq(
        bl_x.name.replace("x", "y", literal=literal),
        pl_x.name.replace("x", "y", literal=literal),
    )
    assert_eq(
        bl_s.name.replace("s", "t", literal=literal),
        pl_s.name.replace("s", "t", literal=literal),
    )
    assert_eq(
        bl_salary.replace("salary", "income", literal=literal),
        pl_salary.replace("salary", "income", literal=literal),
    )

    assert_eq(
        bl_salary.replace("a", "b", literal=literal),
        pl_salary.replace("a", "b", literal=literal),
    )


@case_skip
def test_name_replace_case_sensitivity(literal: bool) -> None:
    assert_eq(
        bl_salary.replace("l", "L", literal=literal),
        pl_salary.replace("l", "L", literal=literal),
    )
    assert_eq(
        bl_salary.replace("sal", "SAL", literal=literal),
        pl_salary.replace("sal", "SAL", literal=literal),
    )
