import polars as pl

import pql

from ._utils import assert_eq

pql_struct = pql.col("structs").struct
pl_struct = pl.col("structs").struct


def test_field() -> None:
    assert_eq(pql_struct.field("a"), pl_struct.field("a"))


def test_with_fields() -> None:
    assert_eq(
        pql_struct.with_fields(
            "structs",
            pql_struct.field("a").alias("e"),
            pql_struct.field("b").alias("f"),
            g=pql_struct.field("c"),
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


def test_json_encode() -> None:
    assert_eq(pql_struct.json_encode(), pl_struct.json_encode())
