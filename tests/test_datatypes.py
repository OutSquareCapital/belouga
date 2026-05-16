from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, cast

import duckdb
import pytest

import belugas as bl

if TYPE_CHECKING:
    from _duckdb._typing import (
        StrIntoPyType,
    )
    from pyochain import Dict


class MyEnum(Enum):
    A = "A"
    B = "B"
    C = "C"


_DATA = bl.LazyFrame({
    "numeric": [1, 2, 3],
    "1d": [[1, 2], [3, 4], [5, 6]],
    "2d": [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
    "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
    "hours": [
        "2021-01-01 00:00:00",
        "2021-01-01 01:00:00",
        "2021-01-01 02:00:00",
    ],
    "blobs": [b"foo", b"bar", b"baz"],
    "nanoseconds": [
        "2021-01-01 00:00:00.000000000",
        "2021-01-01 01:00:00.000000000",
        "2021-01-01 02:00:00.000000000",
    ],
    "seconds": [
        "2021-01-01 00:00:00",
        "2021-01-01 01:00:00",
        "2021-01-01 02:00:00",
    ],
    "millis": [
        "2021-01-01 00:00:00.000",
        "2021-01-01 01:00:00.000",
        "2021-01-01 02:00:00.000",
    ],
    "micros": [
        "2021-01-01 00:00:00.000000",
        "2021-01-01 01:00:00.000000",
        "2021-01-01 02:00:00.000000",
    ],
    "time": ["12:00:00", "13:00:00", "14:00:00"],
    "time_tz": ["12:00:00+00", "13:00:00+00", "14:00:00+00"],
    "duration": ["1 day", "2 days", "3 days"],
    "enumerated": ["A", "B", "C"],
    "mapped": [
        {"a": 1, "b": 2, "c": 3},
        {"a": 4, "b": 5, "c": 6},
        {"a": 7, "b": 8, "c": 9},
    ],
    "structured": [
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "c": [True, False, True],
        },
        {
            "a": [4, 5, 6],
            "b": ["x", "y", "z"],
            "c": [True, False, True],
        },
        {
            "a": [7, 8, 9],
            "b": ["x", "y", "z"],
            "c": [True, False, True],
        },
    ],
    "unioned": [1, "two", 3.0],
    "bits": [b"\x01", b"\x02", b"\x03"],
    "uuid_data": [
        "550e8400-e29b-41d4-a716-446655440000",
        "550e8400-e29b-41d4-a716-446655440001",
        "550e8400-e29b-41d4-a716-446655440002",
    ],
    "geometry": [
        "POINT (30 10)",
        "LINESTRING (30 10, 10 30, 40 40)",
        "POLYGON ((30 10, 40 40, 20 40, 10 20, 30 10))",
    ],
})


@pytest.fixture(scope="session")
def sample_data() -> bl.LazyFrame:
    return _DATA


@pytest.fixture(scope="session")
def cast_schema(sample_data: bl.LazyFrame) -> Dict[str, bl.DataType]:
    return _create_schema(sample_data)


def _create_schema(sample_data: bl.LazyFrame) -> Dict[str, bl.DataType]:
    return sample_data.select(_exprs()).collect_schema()


def _exprs() -> tuple[bl.Expr, ...]:
    numeric = bl.col("numeric")
    return (
        numeric.cast(bl.Int8()).alias("i8"),
        numeric.cast(bl.Int16()).alias("i16"),
        numeric.cast(bl.Int32()).alias("i32"),
        numeric.cast(bl.Int64()).alias("i64"),
        numeric.cast(bl.Int128()).alias("i128"),
        numeric.cast(bl.UInt8()).alias("u8"),
        numeric.cast(bl.UInt16()).alias("u16"),
        numeric.cast(bl.UInt32()).alias("u32"),
        numeric.cast(bl.UInt64()).alias("u64"),
        numeric.cast(bl.UInt128()).alias("u128"),
        numeric.cast(bl.Float32()).alias("f32"),
        numeric.cast(bl.Float64()).alias("f64"),
        numeric.cast(bl.Boolean()).alias("bool"),
        numeric.cast(bl.Decimal(10, 2)).alias("dec"),
        numeric.cast(bl.String()).alias("s"),
        numeric.cast(bl.Number()).alias("num"),
        bl.col("time").cast(bl.Time()).alias("time"),
        bl.col("time_tz").cast(bl.TimeTZ()).alias("time_tz"),
        bl.col("dates").cast(bl.Date()).alias("dates"),
        bl.col("hours").cast(bl.DatetimeTZ()).alias("hours"),
        bl.col("seconds").cast(bl.Datetime(time_unit="s")).alias("datetime_s"),
        bl.col("millis").cast(bl.Datetime(time_unit="ms")).alias("datetime_ms"),
        bl.col("micros").cast(bl.Datetime(time_unit="us")).alias("datetime_us"),
        bl.col("nanoseconds").cast(bl.Datetime(time_unit="ns")).alias("datetime_ns"),
        bl.col("1d").cast(bl.List(bl.UInt16())).alias("lst"),
        bl.col("1d").cast(bl.Array(bl.UInt16(), size=2)).alias("arr_1d"),
        bl.col("2d").cast(bl.Array(bl.UInt16(), size=2).with_dim(2)).alias("arr_2d"),
        bl.col("blobs").cast(bl.Binary()),
        bl.col("duration").cast(bl.Duration()),
        bl.col("enumerated").cast(bl.Enum(["A", "B", "C"])),
        bl.col("enumerated").cast(bl.Enum(MyEnum)).alias("enumerated_enum"),
        bl.col("mapped").cast(bl.Map(bl.String(), bl.Int32())),
        bl.col("structured").cast(
            bl.Struct({"a": bl.Int32(), "b": bl.String(), "c": bl.Boolean()})
        ),
        bl.col("unioned").cast(bl.Union([bl.Int32(), bl.String(), bl.Float64()])),
        bl.col("bits").cast(bl.BitString()),
        bl.col("uuid_data").cast(bl.UUID()),
        bl.col("geometry").cast(bl.Geometry()),
    )


def test_geometry(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["geometry"], bl.Geometry)


def test_signed_integer(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["i8"], bl.Int8)
    assert isinstance(cast_schema["i16"], bl.Int16)
    assert isinstance(cast_schema["i32"], bl.Int32)
    assert isinstance(cast_schema["i64"], bl.Int64)
    assert isinstance(cast_schema["i128"], bl.Int128)
    assert bl.Int32().is_signed_integer()
    assert bl.Int32.is_integer()
    assert bl.Int32.is_numeric()
    assert bl.Int32().is_(bl.Int32())
    assert not bl.Int32.is_(bl.Int64())


def test_unsigned_integer(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["u8"], bl.UInt8)
    assert isinstance(cast_schema["u16"], bl.UInt16)
    assert isinstance(cast_schema["u32"], bl.UInt32)
    assert isinstance(cast_schema["u64"], bl.UInt64)
    assert isinstance(cast_schema["u128"], bl.UInt128)
    assert bl.UInt16.is_unsigned_integer()
    assert bl.UInt16.is_integer()
    assert bl.UInt16.is_numeric()


def test_float(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["f32"], bl.Float32)
    assert isinstance(cast_schema["f64"], bl.Float64)
    assert bl.Float32.is_float()
    assert bl.Float64.is_float()
    assert bl.Float64.is_numeric()


def test_bool_str(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["bool"], bl.Boolean)
    assert isinstance(cast_schema["s"], bl.String)


def test_decimal(cast_schema: Dict[str, bl.DataType]) -> None:
    dec = cast(bl.Decimal, cast_schema["dec"])
    assert isinstance(dec, bl.Decimal)
    assert dec.precision == 10
    assert dec.scale == 2
    assert bl.Decimal.is_decimal()
    assert bl.Decimal.is_numeric()


def test_temporal(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["dates"], bl.Date)
    assert isinstance(cast_schema["hours"], bl.DatetimeTZ)
    assert isinstance(cast_schema["time"], bl.Time)
    assert isinstance(cast_schema["duration"], bl.Duration)
    assert bl.Date.is_temporal()
    assert bl.Time.is_temporal()
    assert bl.Duration.is_temporal()


def test_list(cast_schema: Dict[str, bl.DataType]) -> None:
    lst = cast(bl.List, cast_schema["lst"])
    assert isinstance(lst, bl.List)
    assert isinstance(lst.inner, bl.UInt16)
    assert bl.List.is_nested()


def test_list_from_sql_dtype() -> None:
    parsed = bl.DataType.from_duckdb(duckdb.list_type("varchar"))
    assert isinstance(parsed, bl.List)
    assert isinstance(parsed.inner, bl.String)


def test_array(cast_schema: Dict[str, bl.DataType]) -> None:
    arr_1d = cast(bl.Array, cast_schema["arr_1d"])
    assert isinstance(arr_1d, bl.Array)
    assert isinstance(arr_1d.inner, bl.UInt16)
    assert arr_1d.shape == 2

    arr_2d = cast(bl.Array, cast_schema["arr_2d"])
    assert isinstance(arr_2d, bl.Array)
    assert arr_2d.shape == 2
    assert bl.Array.is_nested()


def test_binary(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["blobs"], bl.Binary)


def test_enum(cast_schema: Dict[str, bl.DataType]) -> None:
    enumerated = cast(bl.Enum, cast_schema["enumerated"])
    assert isinstance(enumerated, bl.Enum)
    assert tuple(enumerated.categories) == ("A", "B", "C")

    enumerated_enum = cast(bl.Enum, cast_schema["enumerated_enum"])
    assert isinstance(enumerated_enum, bl.Enum)
    assert tuple(enumerated_enum.categories) == ("A", "B", "C")


def test_map(cast_schema: Dict[str, bl.DataType]) -> None:
    mapped = cast(bl.Map, cast_schema["mapped"])
    assert isinstance(mapped, bl.Map)
    assert isinstance(mapped.key, bl.String)
    assert isinstance(mapped.value, bl.Int32)


def test_struct(cast_schema: Dict[str, bl.DataType]) -> None:
    struct = cast(bl.Struct, cast_schema["structured"])
    assert isinstance(struct, bl.Struct)
    assert isinstance(struct.fields["a"], bl.Int32)
    assert isinstance(struct.fields["b"], bl.String)
    assert isinstance(struct.fields["c"], bl.Boolean)
    assert bl.Struct.is_nested()


def test_union(cast_schema: Dict[str, bl.DataType]) -> None:
    unioned = cast(bl.Union, cast_schema["unioned"])
    assert isinstance(unioned, bl.Union)
    assert isinstance(unioned.fields[0], bl.Int32)
    assert isinstance(unioned.fields[1], bl.String)
    assert isinstance(unioned.fields[2], bl.Float64)


def test_time_tz(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["time_tz"], bl.TimeTZ)


def test_datetime_all_time_units(cast_schema: Dict[str, bl.DataType]) -> None:
    datetime_s = cast(bl.Datetime, cast_schema["datetime_s"])
    assert isinstance(datetime_s, bl.Datetime)
    assert datetime_s.time_unit == "s"

    datetime_ms = cast(bl.Datetime, cast_schema["datetime_ms"])
    assert isinstance(datetime_ms, bl.Datetime)
    assert datetime_ms.time_unit == "ms"

    datetime_us = cast(bl.Datetime, cast_schema["datetime_us"])
    assert isinstance(datetime_us, bl.Datetime)
    assert datetime_us.time_unit in {"us", "ns"}

    datetime_ns = cast(bl.Datetime, cast_schema["datetime_ns"])
    assert isinstance(datetime_ns, bl.Datetime)
    assert datetime_ns.time_unit == "ns"


def test_bitstring(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["bits"], bl.BitString)


def test_uuid(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["uuid_data"], bl.UUID)


def test_number(cast_schema: Dict[str, bl.DataType]) -> None:
    assert isinstance(cast_schema["num"], bl.Number)


CANDIDATES: tuple[StrIntoPyType, ...] = (
    "bigint",
    "bit",
    "bignum",
    "blob",
    "boolean",
    "date",
    "double",
    "float",
    "geometry",
    "hugeint",
    "integer",
    "interval",
    "smallint",
    "null",
    "time with time zone",
    "time",
    "time_ns",
    "timestamp_ms",
    "timestamp_ns",
    "timestamp_s",
    "timestamp with time zone",
    "timestamp",
    "tinyint",
    "ubigint",
    "uhugeint",
    "uinteger",
    "usmallint",
    "utinyint",
    "uuid",
    "varchar",
    "variant",
    "json",
)


@pytest.mark.parametrize("dtype", CANDIDATES)
def test_from_str(dtype: StrIntoPyType) -> None:
    parsed = duckdb.type(dtype)
    from_str = bl.DataType.from_str(dtype).__class__.__name__
    from_duckdb = bl.DataType.from_duckdb(parsed).__class__.__name__
    assert from_str == from_duckdb
