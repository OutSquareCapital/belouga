"""Data types Mapping for PQL."""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum as PyEnum
from functools import partial
from typing import TYPE_CHECKING, Any, Concatenate, Self, TypeIs, final, overload

import pyochain as pc
from sqlglot import exp

if TYPE_CHECKING:
    from duckdb.sqltypes import DuckDBPyType

    from .sql.typing import EpochTimeUnit, IntoDict


build = partial(exp.DataType.build, dialect="duckdb")  # pyright: ignore[reportUnknownMemberType]


@dataclass(slots=True)
class ClassInstMethod[**P, R]:
    """Decorator that allows a method to be called from the class OR instance."""

    func: Callable[Concatenate[Any, P], R]  # pyright: ignore[reportExplicitAny]

    @overload
    def __get__(self, instance: None, type_: type) -> Callable[P, R]: ...
    @overload
    def __get__(self, instance: object, type_: type) -> Callable[P, R]: ...
    def __get__(self, instance: object | None, type_: type) -> Callable[..., R]:
        if instance is not None:
            return self.func.__get__(instance, type_)
        return self.func.__get__(type_, type_)


@dataclass(slots=True, init=False, unsafe_hash=True)
class DataType(ABC):
    """Base class for data types."""

    raw: exp.DataType

    @classmethod
    def from_duckdb(cls, dtype: DuckDBPyType) -> DataType:
        """Convert a DuckDBPyType to a PQL DataType."""
        return cls.from_sql(build(str(dtype)))

    @classmethod
    def from_sql(cls, dtype: exp.DataType) -> DataType:
        """Convert a sqlglot DataType to a PQL DataType."""
        dt_enum: exp.DType = dtype.this  # pyright: ignore[reportAny]
        match dt_enum:
            case exp.DType.ARRAY if not dtype.args.get("values"):
                return List.__from_raw__(dtype)
            case _:
                return (
                    NESTED_MAP
                    .get_item(dt_enum)
                    .map(lambda constructor: constructor.__from_raw__(dtype))
                    .unwrap_or_else(
                        lambda: (
                            NON_NESTED_MAP
                            .get_item(dt_enum)
                            .ok_or_else(lambda: f"Unsupported data type: {dtype}")
                            .unwrap()
                        )
                    )
                )

    @ClassInstMethod
    def is_[T: DataType](self, other: T) -> TypeIs[T]:
        """Check if this DataType is the same as another DataType."""
        return self == other and hash(self) == hash(other)

    @classmethod
    def is_numeric(cls) -> bool:
        """Check whether the data type is a numeric type."""
        return issubclass(cls, NumericType)

    @classmethod
    def is_decimal(cls) -> bool:
        """Check whether the data type is a decimal type."""
        return issubclass(cls, Decimal)

    @classmethod
    def is_integer(cls) -> bool:
        """Check whether the data type is an integer type."""
        return issubclass(cls, IntegerType)

    @classmethod
    def is_signed_integer(cls) -> bool:
        """Check whether the data type is a signed integer type."""
        return issubclass(cls, SignedIntegerType)

    @classmethod
    def is_unsigned_integer(cls) -> bool:
        """Check whether the data type is an unsigned integer type."""
        return issubclass(cls, UnsignedIntegerType)

    @classmethod
    def is_float(cls) -> bool:
        """Check whether the data type is a floating point type."""
        return issubclass(cls, FloatType)

    @classmethod
    def is_temporal(cls) -> bool:
        """Check whether the data type is a temporal type."""
        return issubclass(cls, TemporalType)

    @classmethod
    def is_nested(cls) -> bool:
        """Check whether the data type is a nested type."""
        return issubclass(cls, NestedType)


@dataclass(slots=True, init=False, unsafe_hash=True)
class StringType(DataType):
    """Base class for string data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class NumericType(DataType):
    """Base class for numeric data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class FloatType(NumericType):
    """Base class for floating-point data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class IntegerType(NumericType):
    """Base class for integer data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class SignedIntegerType(IntegerType):
    """Base class for signed integer data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class UnsignedIntegerType(IntegerType):
    """Base class for unsigned integer data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class TemporalType(DataType):
    """Base class for temporal data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class NestedType(DataType):
    """Base class for nested data types."""


@dataclass(slots=True, init=False, unsafe_hash=True)
class ComplexDataType(DataType):
    """Base class for complex data types that need reverse-construction from parsed sqlglot AST."""

    @classmethod
    def __from_raw__(cls, raw: exp.DataType) -> DataType:
        instance = cls.__new__(cls)
        instance.raw = raw
        instance._init_cache()
        return instance

    def _init_cache(self) -> None:
        """Initialize lazy caches for nested type properties. Override in subclasses."""


@final
@dataclass(slots=True, unsafe_hash=True)
class Time(TemporalType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.TIME))


@final
@dataclass(slots=True, unsafe_hash=True)
class TimeTZ(TemporalType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.TIMETZ))


@final
@dataclass(slots=True, unsafe_hash=True)
class Duration(TemporalType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.INTERVAL))


@final
@dataclass(slots=True, unsafe_hash=True)
class Date(TemporalType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.DATE))


@final
@dataclass(slots=True, unsafe_hash=True)
class DatetimeTZ(TemporalType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.TIMESTAMPTZ))


@final
@dataclass(slots=True, unsafe_hash=True)
class Datetime(TemporalType):
    raw: exp.DataType
    time_unit: EpochTimeUnit

    def __init__(self, time_unit: EpochTimeUnit = "ns") -> None:
        self.raw = PRECISION_MAP.get_item(time_unit).expect(
            f"Unsupported time unit: {time_unit}"
        )
        self.time_unit = time_unit


@final
@dataclass(slots=True, unsafe_hash=True)
class Boolean(DataType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.BOOLEAN))


@final
@dataclass(slots=True, unsafe_hash=True)
class Number(NumericType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.BIGNUM))


@final
@dataclass(slots=True, unsafe_hash=True)
class UUID(NumericType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.UUID))


@final
@dataclass(slots=True, unsafe_hash=True)
class Float32(FloatType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.FLOAT))


@final
@dataclass(slots=True, unsafe_hash=True)
class Float64(FloatType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.DOUBLE))


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Decimal(NumericType, ComplexDataType):
    def __init__(self, precision: int = 18, scale: int = 0) -> None:
        self.raw = exp.DataType(
            this=exp.DType.DECIMAL,
            expressions=[
                exp.DataTypeParam(this=exp.Literal.number(precision)),  # pyright: ignore[reportUnknownMemberType]
                exp.DataTypeParam(this=exp.Literal.number(scale)),  # pyright: ignore[reportUnknownMemberType]
            ],
        )

    @property
    def precision(self) -> int:
        return int(self.raw.expressions[0].this.this)  # pyright: ignore[reportAny]

    @property
    def scale(self) -> int:
        return int(self.raw.expressions[1].this.this)  # pyright: ignore[reportAny]


@final
@dataclass(slots=True, unsafe_hash=True)
class Int8(SignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.TINYINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class Int16(SignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.SMALLINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class Int32(SignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.INT))


@final
@dataclass(slots=True, unsafe_hash=True)
class Int64(SignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.BIGINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class Int128(SignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.INT128))


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt8(UnsignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.UTINYINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt16(UnsignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.USMALLINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt32(UnsignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.UINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt64(UnsignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.UBIGINT))


@final
@dataclass(slots=True, unsafe_hash=True)
class UInt128(UnsignedIntegerType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.UINT128))


@final
@dataclass(slots=True, unsafe_hash=True)
class Binary(DataType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.BLOB))


@final
@dataclass(slots=True, unsafe_hash=True)
class Geometry(DataType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.GEOMETRY))


@final
@dataclass(slots=True, unsafe_hash=True)
class String(StringType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.VARCHAR))


@final
@dataclass(slots=True, unsafe_hash=True)
class Json(StringType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.JSON))


@final
@dataclass(slots=True, unsafe_hash=True)
class BitString(StringType):
    raw: exp.DataType = field(init=False, default=build(exp.DType.BIT))


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Enum(StringType, ComplexDataType):
    def __init__(self, categories: Iterable[str] | type[PyEnum]) -> None:
        match categories:
            case type():
                values: pc.Iter[str] = pc.Iter(categories).map(lambda i: i.value)  # pyright: ignore[reportAny]
            case Iterable():
                values = pc.Iter(categories)

        self.raw = exp.DataType(
            this=exp.DType.ENUM,
            expressions=values.map(exp.Literal.string).collect(list),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )

    @property
    def categories(self) -> pc.Seq[str]:
        return (
            pc.Iter(self.raw.expressions).map(lambda lit: lit.this).collect()  # pyright: ignore[reportAny]
        )


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Union(NestedType, ComplexDataType):
    def __init__(self, fields: Iterable[DataType]) -> None:
        exprs = (
            pc
            .Iter(fields)
            .enumerate()
            .map_star(
                lambda i, f: exp.ColumnDef(this=exp.to_identifier(f"v{i}"), kind=f.raw)
            )
            .collect(list)
        )
        self.raw = exp.DataType(this=exp.DType.UNION, expressions=exprs, nested=True)

    @property
    def fields(self) -> pc.Seq[DataType]:
        return (
            pc
            .Iter(self.raw.expressions)
            .map(lambda col_def: self.from_sql(col_def.kind))  # pyright: ignore[reportAny]
            .collect()
        )


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Map(NestedType, ComplexDataType):
    def __init__(self, key: DataType, value: DataType) -> None:
        self.raw = exp.DataType(
            this=exp.DType.MAP, expressions=[key.raw, value.raw], nested=True
        )

    @property
    def key(self) -> DataType:
        return self.from_sql(self.raw.expressions[0])  # pyright: ignore[reportAny]

    @property
    def value(self) -> DataType:
        return self.from_sql(self.raw.expressions[1])  # pyright: ignore[reportAny]


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Struct(NestedType, ComplexDataType):
    def __init__(self, fields: IntoDict[str, DataType]) -> None:
        exprs = (
            pc
            .Dict(fields)
            .items()
            .iter()
            .map_star(
                lambda name, col: exp.ColumnDef(
                    this=exp.to_identifier(name), kind=col.raw
                )
            )
            .collect(list)
        )
        self.raw = exp.DataType(this=exp.DType.STRUCT, expressions=exprs, nested=True)

    @property
    def fields(self) -> pc.Dict[str, DataType]:
        return (
            pc
            .Iter(self.raw.expressions)
            .map(
                lambda col_def: (  # pyright: ignore[reportAny]
                    col_def.this.this,  # pyright: ignore[reportAny]
                    self.from_sql(col_def.kind),  # pyright: ignore[reportAny]
                )
            )
            .collect(pc.Dict)
        )


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class Array(NestedType, ComplexDataType):
    def __init__(self, inner: DataType, size: int = 1) -> None:
        self.raw = exp.DataType(
            this=exp.DType.ARRAY,
            expressions=[inner.raw],
            values=[exp.Literal.number(size)],  # pyright: ignore[reportUnknownMemberType]
            nested=True,
        )

    def with_dim(self, size: int) -> Self:
        """Add another level of nesting to the array."""
        return self.__class__(self, size)

    @property
    def inner(self) -> DataType:
        return self.from_sql(self.raw.expressions[0])  # pyright: ignore[reportAny]

    @property
    def shape(self) -> int:
        values = self.raw.args.get("values")
        return int(values[0].this) if values else 1  # pyright: ignore[reportAny]


@final
@dataclass(slots=True, init=False, unsafe_hash=True)
class List(NestedType, ComplexDataType):
    def __init__(self, inner: DataType) -> None:
        self.raw = exp.DataType(
            this=exp.DType.ARRAY, expressions=[inner.raw], nested=True
        )

    @property
    def inner(self) -> DataType:
        return self.from_sql(self.raw.expressions[0])  # pyright: ignore[reportAny]


PRECISION_MAP: pc.Dict[EpochTimeUnit, exp.DataType] = pc.Dict.from_ref({
    "s": build(exp.DType.TIMESTAMP_S),
    "ms": build(exp.DType.TIMESTAMP_MS),
    "us": build(exp.DType.TIMESTAMP),
    "ns": build(exp.DType.TIMESTAMP_NS),
})


NESTED_MAP: pc.Dict[exp.DType, type[ComplexDataType]] = pc.Dict.from_ref({
    exp.DType.LIST: List,
    exp.DType.ARRAY: Array,
    exp.DType.STRUCT: Struct,
    exp.DType.MAP: Map,
    exp.DType.UNION: Union,
    exp.DType.ENUM: Enum,
    exp.DType.DECIMAL: Decimal,
})

NON_NESTED_MAP: pc.Dict[exp.DType, DataType] = pc.Dict.from_ref({
    exp.DType.BIGINT: Int64(),
    exp.DType.BIT: BitString(),
    exp.DType.BIGNUM: Number(),
    exp.DType.VARBINARY: Binary(),
    exp.DType.BOOLEAN: Boolean(),
    exp.DType.DATE: Date(),
    exp.DType.DOUBLE: Float64(),
    exp.DType.FLOAT: Float32(),
    exp.DType.INT128: Int128(),
    exp.DType.GEOMETRY: Geometry(),
    exp.DType.INT: Int32(),
    exp.DType.INTERVAL: Duration(),
    exp.DType.JSON: Json(),
    exp.DType.SMALLINT: Int16(),
    exp.DType.TIMESTAMP_S: Datetime("s"),
    exp.DType.TIMESTAMP_MS: Datetime("ms"),
    exp.DType.TIMESTAMP: Datetime(),
    exp.DType.TIMESTAMPNTZ: Datetime(),
    exp.DType.TIMESTAMP_NS: Datetime("ns"),
    exp.DType.TIMESTAMPTZ: DatetimeTZ(),
    exp.DType.TIME: Time(),
    exp.DType.TIME_NS: Time(),
    exp.DType.TIMETZ: TimeTZ(),
    exp.DType.TINYINT: Int8(),
    exp.DType.UUID: UUID(),
    exp.DType.UINT128: UInt128(),
    exp.DType.UBIGINT: UInt64(),
    exp.DType.UINT: UInt32(),
    exp.DType.USMALLINT: UInt16(),
    exp.DType.UTINYINT: UInt8(),
    exp.DType.TEXT: String(),
    exp.DType.VARIANT: Number(),
})
