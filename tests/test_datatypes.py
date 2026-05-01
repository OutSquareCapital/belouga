from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pytest

import pql

if TYPE_CHECKING:
    from pyochain import Dict


class MyEnum(Enum):
    A = "A"
    B = "B"
    C = "C"


_DATA = pql.LazyFrame({
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
def sample_data() -> pql.LazyFrame:
    return _DATA


@pytest.fixture(scope="session")
def cast_schema(sample_data: pql.LazyFrame) -> Dict[str, pql.DataType]:
    return _create_schema(sample_data)


def _create_schema(sample_data: pql.LazyFrame) -> Dict[str, pql.DataType]:
    return sample_data.select(_exprs()).schema


def _exprs() -> tuple[pql.Expr, ...]:
    numeric = pql.col("numeric")
    return (
        numeric.cast(pql.Int8()).alias("i8"),
        numeric.cast(pql.Int16()).alias("i16"),
        numeric.cast(pql.Int32()).alias("i32"),
        numeric.cast(pql.Int64()).alias("i64"),
        numeric.cast(pql.Int128()).alias("i128"),
        numeric.cast(pql.UInt8()).alias("u8"),
        numeric.cast(pql.UInt16()).alias("u16"),
        numeric.cast(pql.UInt32()).alias("u32"),
        numeric.cast(pql.UInt64()).alias("u64"),
        numeric.cast(pql.UInt128()).alias("u128"),
        numeric.cast(pql.Float32()).alias("f32"),
        numeric.cast(pql.Float64()).alias("f64"),
        numeric.cast(pql.Boolean()).alias("bool"),
        numeric.cast(pql.Decimal(10, 2)).alias("dec"),
        numeric.cast(pql.String()).alias("s"),
        numeric.cast(pql.Number()).alias("num"),
        pql.col("time").cast(pql.Time()).alias("time"),
        pql.col("time_tz").cast(pql.TimeTZ()).alias("time_tz"),
        pql.col("dates").cast(pql.Date()).alias("dates"),
        pql.col("hours").cast(pql.DatetimeTZ()).alias("hours"),
        pql.col("seconds").cast(pql.Datetime(time_unit="s")).alias("datetime_s"),
        pql.col("millis").cast(pql.Datetime(time_unit="ms")).alias("datetime_ms"),
        pql.col("micros").cast(pql.Datetime(time_unit="us")).alias("datetime_us"),
        pql.col("nanoseconds").cast(pql.Datetime(time_unit="ns")).alias("datetime_ns"),
        pql.col("1d").cast(pql.List(pql.UInt16())).alias("lst"),
        pql.col("1d").cast(pql.Array(pql.UInt16(), size=2)).alias("arr_1d"),
        pql.col("2d").cast(pql.Array(pql.UInt16(), size=2).with_dim(2)).alias("arr_2d"),
        pql.col("blobs").cast(pql.Binary()),
        pql.col("duration").cast(pql.Duration()),
        pql.col("enumerated").cast(pql.Enum(["A", "B", "C"])),
        pql.col("enumerated").cast(pql.Enum(MyEnum)).alias("enumerated_enum"),
        pql.col("mapped").cast(pql.Map(pql.String(), pql.Int32())),
        pql.col("structured").cast(
            pql.Struct({"a": pql.Int32(), "b": pql.String(), "c": pql.Boolean()})
        ),
        pql.col("unioned").cast(pql.Union([pql.Int32(), pql.String(), pql.Float64()])),
        pql.col("bits").cast(pql.BitString()),
        pql.col("uuid_data").cast(pql.UUID()),
        pql.col("geometry").cast(pql.Geometry()),
    )


def test_geometry(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["geometry"], pql.Geometry)


def test_signed_integer(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["i8"], pql.Int8)
    assert isinstance(cast_schema["i16"], pql.Int16)
    assert isinstance(cast_schema["i32"], pql.Int32)
    assert isinstance(cast_schema["i64"], pql.Int64)
    assert isinstance(cast_schema["i128"], pql.Int128)
    assert pql.Int32().is_signed_integer()
    assert pql.Int32.is_integer()
    assert pql.Int32.is_numeric()
    assert pql.Int32().is_(pql.Int32())
    assert not pql.Int32.is_(pql.Int64())


def test_unsigned_integer(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["u8"], pql.UInt8)
    assert isinstance(cast_schema["u16"], pql.UInt16)
    assert isinstance(cast_schema["u32"], pql.UInt32)
    assert isinstance(cast_schema["u64"], pql.UInt64)
    assert isinstance(cast_schema["u128"], pql.UInt128)
    assert pql.UInt16.is_unsigned_integer()
    assert pql.UInt16.is_integer()
    assert pql.UInt16.is_numeric()


def test_float(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["f32"], pql.Float32)
    assert isinstance(cast_schema["f64"], pql.Float64)
    assert pql.Float32.is_float()
    assert pql.Float64.is_float()
    assert pql.Float64.is_numeric()


def test_bool_str(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["bool"], pql.Boolean)
    assert isinstance(cast_schema["s"], pql.String)


def test_decimal(cast_schema: Dict[str, pql.DataType]) -> None:
    dec: pql.Decimal = cast_schema["dec"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(dec, pql.Decimal)
    assert dec.precision == 10
    assert dec.scale == 2
    assert pql.Decimal.is_decimal()
    assert pql.Decimal.is_numeric()


def test_temporal(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["dates"], pql.Date)
    assert isinstance(cast_schema["hours"], pql.DatetimeTZ)
    assert isinstance(cast_schema["time"], pql.Time)
    assert isinstance(cast_schema["duration"], pql.Duration)
    assert pql.Date.is_temporal()
    assert pql.Time.is_temporal()
    assert pql.Duration.is_temporal()


def test_list(cast_schema: Dict[str, pql.DataType]) -> None:
    lst: pql.List = cast_schema["lst"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(lst, pql.List)
    assert isinstance(lst.inner, pql.UInt16)
    assert pql.List.is_nested()


def test_array(cast_schema: Dict[str, pql.DataType]) -> None:
    arr_1d: pql.Array = cast_schema["arr_1d"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(arr_1d, pql.Array)
    assert isinstance(arr_1d.inner, pql.UInt16)
    assert arr_1d.shape == 2

    arr_2d: pql.Array = cast_schema["arr_2d"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(arr_2d, pql.Array)
    assert arr_2d.shape == 2
    assert pql.Array.is_nested()


def test_binary(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["blobs"], pql.Binary)


def test_enum(cast_schema: Dict[str, pql.DataType]) -> None:
    enumerated: pql.Enum = cast_schema["enumerated"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(enumerated, pql.Enum)
    assert tuple(enumerated.categories) == ("A", "B", "C")

    enumerated_enum: pql.Enum = cast_schema["enumerated_enum"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(enumerated_enum, pql.Enum)
    assert tuple(enumerated_enum.categories) == ("A", "B", "C")


def test_map(cast_schema: Dict[str, pql.DataType]) -> None:
    mapped: pql.Map = cast_schema["mapped"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(mapped, pql.Map)
    assert isinstance(mapped.key, pql.String)
    assert isinstance(mapped.value, pql.Int32)


def test_struct(cast_schema: Dict[str, pql.DataType]) -> None:
    struct: pql.Struct = cast_schema["structured"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(struct, pql.Struct)
    assert isinstance(struct.fields["a"], pql.Int32)
    assert isinstance(struct.fields["b"], pql.String)
    assert isinstance(struct.fields["c"], pql.Boolean)
    assert pql.Struct.is_nested()


def test_union(cast_schema: Dict[str, pql.DataType]) -> None:
    unioned: pql.Union = cast_schema["unioned"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(unioned, pql.Union)
    assert isinstance(unioned.fields[0], pql.Int32)
    assert isinstance(unioned.fields[1], pql.String)
    assert isinstance(unioned.fields[2], pql.Float64)


def test_time_tz(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["time_tz"], pql.TimeTZ)


def test_datetime_all_time_units(cast_schema: Dict[str, pql.DataType]) -> None:
    datetime_s: pql.Datetime = cast_schema["datetime_s"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(datetime_s, pql.Datetime)
    assert datetime_s.time_unit == "s"

    datetime_ms: pql.Datetime = cast_schema["datetime_ms"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(datetime_ms, pql.Datetime)
    assert datetime_ms.time_unit == "ms"

    datetime_us: pql.Datetime = cast_schema["datetime_us"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(datetime_us, pql.Datetime)
    assert datetime_us.time_unit in {"us", "ns"}

    datetime_ns: pql.Datetime = cast_schema["datetime_ns"]  # pyright: ignore[reportAssignmentType]
    assert isinstance(datetime_ns, pql.Datetime)
    assert datetime_ns.time_unit == "ns"


def test_bitstring(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["bits"], pql.BitString)


def test_uuid(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["uuid_data"], pql.UUID)


def test_number(cast_schema: Dict[str, pql.DataType]) -> None:
    assert isinstance(cast_schema["num"], pql.Number)
