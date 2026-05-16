import polars as pl
import pytest

import belugas as bl

from ._utils import assert_eq

bl_struct = bl.col("structs").struct
pl_struct = pl.col("structs").struct


def test_field() -> None:
    assert_eq(bl_struct.field("a"), pl_struct.field("a"))


def test_with_fields() -> None:
    assert_eq(
        bl_struct.with_fields(
            "structs",
            bl_struct.field("a").alias("e"),
            bl_struct.field("b").alias("f"),
            g=bl_struct.field("c"),
            h="structs",
        ),
        pl_struct.with_fields(
            "structs",
            pl_struct.field("a").alias("e"),
            pl_struct.field("b").alias("f"),
            g=pl_struct.field("c"),
            h="structs",
        ),
    )


skip_field = pytest.mark.xfail(
    reason="Field overwrite is currently not supported in Belugas"
)


@skip_field
def test_with_fields_arithmetic() -> None:
    assert_eq(
        bl_struct.with_fields(
            bl_struct.field("a").add(bl_struct.field("b")).alias("a")
        ),
        pl_struct.with_fields(
            pl_struct.field("a").add(pl_struct.field("b")).alias("a")
        ),
    )


@skip_field
def test_with_fields_cast() -> None:
    assert_eq(
        bl_struct.with_fields(bl_struct.field("a").cast(bl.Float64).alias("a")),
        pl_struct.with_fields(pl_struct.field("a").cast(pl.Float64).alias("a")),
    )


@skip_field
def test_with_fields_chained() -> None:
    assert_eq(
        bl_struct.with_fields(bl_struct.field("a").add(1).mul(2).alias("computed")),
        pl_struct.with_fields(pl_struct.field("a").add(1).mul(2).alias("computed")),
    )


def test_with_fields_add_new_field() -> None:
    assert_eq(
        bl_struct.with_fields(bl_struct.field("a").alias("new")),
        pl_struct.with_fields(pl_struct.field("a").alias("new")),
    )


def test_json_encode() -> None:
    assert_eq(bl_struct.json_encode(), pl_struct.json_encode())
