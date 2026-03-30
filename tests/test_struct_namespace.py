import narwhals as nw
import polars as pl

import pql

from ._utils import assert_eq, assert_eq_pl


def test_field() -> None:
    assert_eq(pql.col("structs").struct.field("a"), nw.col("structs").struct.field("a"))


def test_with_fields() -> None:
    assert_eq_pl(
        pql
        .col("structs")
        .struct.with_fields(
            "structs",
            pql.col("structs").struct.field("a").alias("e"),
            pql.col("structs").struct.field("b").alias("f"),
            g=pql.col("structs").struct.field("c"),
            h="structs",
        )
        .alias("structs"),
        pl
        .col("structs")
        .struct.with_fields(
            "structs",
            pl.col("structs").struct.field("a").alias("e"),
            pl.col("structs").struct.field("b").alias("f"),
            g=pl.col("structs").struct.field("c"),
            h="structs",
        )
        .alias("structs"),
    )


def test_json_encode() -> None:
    assert_eq_pl(
        pql.col("structs").struct.json_encode(), pl.col("structs").struct.json_encode()
    )
