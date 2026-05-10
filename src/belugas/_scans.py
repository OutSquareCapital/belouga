from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from operator import itemgetter as get
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import duckdb
from duckdb import DuckDBPyConnection, DuckDBPyRelation
from pyochain import Dict, Iter, Seq
from sqlglot import exp

from .typing import (
    IntoArrowArray,
    IntoArrowStream,
    IntoPlDataFrame,
    IntoPlLazyFrame,
    LitSeq,
    NestedSeq,
    NPArrayLike,
)

if TYPE_CHECKING:
    import pandas as pd
    from _duckdb._enums import (  # pyright: ignore[reportMissingModuleSource]
        CSVLineTerminator,
    )
    from _duckdb._typing import (  # pyright: ignore[reportMissingModuleSource]
        ColumnsTypes,
        CsvCompression,
        CsvEncoding,
        HiveTypes,
        IntoFields,
        JsonCompression,
        JsonFormat,
        JsonRecordOptions,
        ParquetCompression,
        StrIntoPyType,
    )

    from ._frame import LazyFrame
    from .typing import (
        AnyArray,
        IntoArrow,
        IntoDict,
        IntoPolars,
        IntoRel,
        Orientation,
        PathOrBuffer,
        PythonLiteral,
        Schema,
        SeqIntoVals,
    )


def from_query(query: exp.Expr, **relations: IntoRel) -> LazyFrame:
    return ScanSource.from_query(query, **relations).into_frame()


def from_table(table: str) -> LazyFrame:
    return ScanSource.from_table(table).into_frame()


def from_table_function(function: str) -> LazyFrame:
    return ScanSource.from_table_function(function).into_frame()


def from_numpy(arr: AnyArray, orient: Orientation = "col") -> LazyFrame:
    return ScanSource.from_numpy(arr, orient=orient).into_frame()


def from_dict(mapping: IntoDict[str, PythonLiteral]) -> LazyFrame:
    return ScanSource.from_dict(mapping).into_frame()


def from_dicts(data: Sequence[Mapping[str, PythonLiteral]]) -> LazyFrame:
    return ScanSource.from_dicts(data).into_frame()


def from_records(data: SeqIntoVals, orient: Orientation = "col") -> LazyFrame:
    return ScanSource.from_records(data, orient=orient).into_frame()


def from_pandas(
    df: pd.DataFrame, connection: DuckDBPyConnection | None = None
) -> LazyFrame:
    return ScanSource.from_relation(
        duckdb.from_df(df, connection=connection)
    ).into_frame()


def from_polars(
    df: IntoPolars, connection: DuckDBPyConnection | None = None
) -> LazyFrame:
    return ScanSource.from_polars(df, connection=connection).into_frame()


def from_arrow(
    df: IntoArrow, connection: DuckDBPyConnection | None = None
) -> LazyFrame:
    return ScanSource.from_arrow(df, connection=connection).into_frame()


def scan_parquet(  # noqa: PLR0913
    file_glob: Path | str | Iterable[str | Path],
    /,
    *,
    binary_as_string: bool = False,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: ParquetCompression | None = None,
    connection: DuckDBPyConnection | None = None,
) -> LazyFrame:
    match file_glob:
        case Path():
            path = str(file_glob)
        case str():
            path = file_glob
        case Iterable():
            path = (
                Iter(file_glob)
                .map(lambda x: x if isinstance(x, str) else str(x))
                .collect()
            )
    rel = _get_conn(connection).from_parquet(
        path,
        binary_as_string,
        file_row_number=file_row_number,
        filename=filename,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        compression=compression,
    )

    return ScanSource.from_relation(rel).into_frame()


def scan_csv(  # noqa: PLR0913
    path_or_buffer: PathOrBuffer,
    *,
    header: bool | int | None = None,
    compression: CsvCompression | None = None,
    sep: str | None = None,
    delimiter: str | None = None,
    files_to_sniff: int | None = None,
    comment: str | None = None,
    thousands: str | None = None,
    dtype: IntoFields | None = None,
    na_values: str | list[str] | None = None,
    skiprows: int | None = None,
    quotechar: str | None = None,
    escapechar: str | None = None,
    encoding: CsvEncoding | None = None,
    parallel: bool | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    sample_size: int | None = None,
    auto_detect: bool | int | None = None,
    all_varchar: bool | None = None,
    normalize_names: bool | None = None,
    null_padding: bool | None = None,
    names: list[str] | None = None,
    lineterminator: CSVLineTerminator | None = None,
    columns: ColumnsTypes | None = None,
    auto_type_candidates: list[StrIntoPyType] | None = None,
    max_line_size: int | None = None,
    ignore_errors: bool | None = None,
    store_rejects: bool | None = None,
    rejects_table: str | None = None,
    rejects_scan: str | None = None,
    rejects_limit: int | None = None,
    force_not_null: list[str] | None = None,
    buffer_size: int | None = None,
    decimal: str | None = None,
    allow_quoted_nulls: bool | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: HiveTypes | None = None,
    hive_types_autocast: bool | None = None,
    strict_mode: bool | None = None,
    connection: DuckDBPyConnection | None = None,
) -> LazyFrame:
    rel = _get_conn(connection).from_csv_auto(
        path_or_buffer,
        header=header,
        compression=compression,
        sep=sep,
        delimiter=delimiter,
        files_to_sniff=files_to_sniff,
        comment=comment,
        thousands=thousands,
        dtype=dtype,
        na_values=na_values,
        skiprows=skiprows,
        quotechar=quotechar,
        escapechar=escapechar,
        encoding=encoding,
        parallel=parallel,
        date_format=date_format,
        timestamp_format=timestamp_format,
        sample_size=sample_size,
        auto_detect=auto_detect,
        all_varchar=all_varchar,
        normalize_names=normalize_names,
        null_padding=null_padding,
        names=names,
        lineterminator=lineterminator,
        columns=columns,
        auto_type_candidates=auto_type_candidates,
        max_line_size=max_line_size,
        ignore_errors=ignore_errors,
        store_rejects=store_rejects,
        rejects_table=rejects_table,
        rejects_scan=rejects_scan,
        rejects_limit=rejects_limit,
        force_not_null=force_not_null,
        buffer_size=buffer_size,
        decimal=decimal,
        allow_quoted_nulls=allow_quoted_nulls,
        filename=filename,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        hive_types=hive_types,
        hive_types_autocast=hive_types_autocast,
        strict_mode=strict_mode,
    )
    return ScanSource.from_relation(rel).into_frame()


def scan_json(  # noqa: PLR0913
    path_or_buffer: PathOrBuffer,
    *,
    columns: ColumnsTypes | None = None,
    sample_size: int | None = None,
    maximum_depth: int | None = None,
    records: JsonRecordOptions | None = None,
    fmt: JsonFormat | None = None,
    date_format: str | None = None,
    timestamp_format: str | None = None,
    compression: JsonCompression | None = None,
    maximum_object_size: int | None = None,
    ignore_errors: bool | None = None,
    convert_strings_to_integers: bool | None = None,
    field_appearance_threshold: float | None = None,
    map_inference_threshold: int | None = None,
    maximum_sample_files: int | None = None,
    filename: bool | str | None = None,
    hive_partitioning: bool | None = None,
    union_by_name: bool | None = None,
    hive_types: HiveTypes | None = None,
    hive_types_autocast: bool | None = None,
    connection: DuckDBPyConnection | None = None,
) -> LazyFrame:
    rel = _get_conn(connection).read_json(
        path_or_buffer,
        columns=columns,
        sample_size=sample_size,
        maximum_depth=maximum_depth,
        records=records,
        format=fmt,
        date_format=date_format,
        timestamp_format=timestamp_format,
        compression=compression,
        maximum_object_size=maximum_object_size,
        ignore_errors=ignore_errors,
        convert_strings_to_integers=convert_strings_to_integers,
        field_appearance_threshold=field_appearance_threshold,
        map_inference_threshold=map_inference_threshold,
        maximum_sample_files=maximum_sample_files,
        filename=filename,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        hive_types=hive_types,
        hive_types_autocast=hive_types_autocast,
    )
    return ScanSource.from_relation(rel).into_frame()


def _get_conn(connection: DuckDBPyConnection | None) -> DuckDBPyConnection:
    match connection:
        case DuckDBPyConnection():
            return connection
        case None:
            return duckdb.default_connection()


COL0 = "column_0"


class BelugasConversionError(ValueError):
    """Raised when a conversion from a sqlglot expression to a DuckDB expression fails."""

    def __init__(self, e: Exception, expr: exp.Expr) -> None:
        msg = f"""
Failed to convert expression to DuckDB!
DuckDB error:
        {e}
----------------------------------
sqlglot expression:
{expr!r}
----------------------------------
SQL representation:
```sql
{expr.sql(dialect="duckdb", pretty=True, identify=True)}
```
"""
        super().__init__(msg)


@dataclass(slots=True)
class ScanSource:
    relation: DuckDBPyRelation
    schema: Schema

    @classmethod
    def from_query(cls, query: exp.Expr, **relations: IntoRel) -> Self:
        """Create a `DuckDBPyRelation` from a  `exp.Expr` node.

        Args:
            query (exp.Expr): The SQL node to execute.
            **relations (IntoRel): Relations to include in the query.

        Returns:
            DuckDBPyRelation: The resulting DuckDB relation.

        Raises:
                BelugasConversionError: If the SQL query cannot be parsed by DuckDB.
        """
        try:
            rels = (
                Iter(relations.items())
                .map_star(lambda k, v: (k, cls.build(v).relation))
                .collect(dict)
            )
            parsed = query.sql(dialect="duckdb", identify=True)
            namespace = {"duckdb": duckdb, "parsed": parsed, **rels}
            exec("relation = duckdb.from_query(parsed)", namespace)
            result = cast(DuckDBPyRelation, namespace["relation"])
            return cls.from_relation(result)
        except duckdb.Error as e:
            raise BelugasConversionError(e, query) from e

    @classmethod
    def build(cls, source: IntoRel | None, orient: Orientation = "col") -> Self:  # noqa: PLR0911
        match source:
            case None:
                return cls.from_none()
            case ScanSource():
                return cls.copy(source)  # pyright: ignore[reportArgumentType]
            case DuckDBPyRelation():
                return cls.from_relation(source)
            case Mapping():
                return cls.from_dict(source)
            case NPArrayLike():
                return cls.from_numpy(source, orient=orient)
            case IntoPlDataFrame() | IntoPlLazyFrame():
                return cls.from_polars(source)
            case IntoArrowStream() | IntoArrowArray():
                return cls.from_arrow(source)
            case Sequence():
                return cls.from_records(source, orient=orient)

    @classmethod
    def from_none(cls) -> Self:
        from ._plan import Marker

        return cls.from_dict({Marker.TEMP: ()})

    def copy(self) -> Self:
        return self.__class__(self.relation, self.schema)

    @classmethod
    def from_dict(cls, data: IntoDict[str, Any]) -> Self:  # pyright: ignore[reportExplicitAny]
        data = Dict(data)

        raw_vals = data.items().iter().map_star(_to_expr).collect(tuple)
        rel = duckdb.values(raw_vals).select(*data.iter().map(_unnest))
        return cls.from_relation(rel)

    @classmethod
    def from_numpy(cls, data: AnyArray, orient: Orientation = "col") -> Self:

        match data.ndim:
            case 1:
                rel = duckdb.values(_to_expr(COL0, data)).select(_unnest(COL0))
                return cls.from_relation(rel)
            case _:
                arr = data.T if orient == "col" else data

                def _array_strategy() -> tuple[int, Callable[[int], AnyArray]]:
                    match (arr.ndim, orient):
                        case (2, _) | (_, "row"):
                            return 1, lambda j: arr[:, j]  # pyright: ignore[reportAny]
                        case _:
                            return 0, lambda j: arr[j]  # pyright: ignore[reportAny]

                def _named_array(names: Seq[str]) -> DuckDBPyRelation:
                    vals = (
                        names
                        .iter()
                        .enumerate()
                        .map_star(lambda j, name: _to_expr(name, arr_getter(j)))
                        .collect(tuple)
                    )

                    return duckdb.values(vals).select(*names.iter().map(_unnest))

                axis, arr_getter = _array_strategy()
                names_nb: int = arr.shape[axis]  # pyright: ignore[reportAny]
                cols = Iter(range(names_nb)).map(_named).collect()
                return cls.from_relation(_named_array(cols))

    @classmethod
    def from_polars(
        cls, df: IntoPolars, connection: DuckDBPyConnection | None = None
    ) -> Self:
        """Create a ScanSource from a Polars DataFrame or LazyFrame.

        Note:
            Two big improvements here would be to:

            1)  Exploit `polars::LazyFrame::collect_batches` to avoid materializing the entire DataFrame in memory at once.
                This would require managing the lifecycle of the Iterator. If we do it naively, it will just freeze once the Iterator is empty.

            2)  Exploit `sqlglot` and the sql capabilities of polars to push down the AST into polars directly.

        Returns:
            Self
        """
        return cls.from_arrow(df.lazy().collect(), connection=connection)

    @classmethod
    def from_arrow(
        cls, df: IntoArrow, connection: DuckDBPyConnection | None = None
    ) -> Self:
        return cls.from_relation(duckdb.from_arrow(df, connection=connection))

    @classmethod
    def from_records(cls, data: SeqIntoVals, orient: Orientation = "col") -> Self:
        match data[0]:
            case Mapping():
                vals = cast(Sequence[Mapping[str, Any]], data)  # pyright: ignore[reportExplicitAny]
                return cls.from_dicts(vals)
            case Sequence() as value if not isinstance(value, str | bytes | bytearray):  # pyright: ignore[reportUnknownVariableType]
                vals = cast(NestedSeq, data)
                match orient:
                    case "col":
                        return cls.from_seq_col(vals)

                    case "row":
                        return cls.from_seq_row(vals)
            case _:
                vals = cast(LitSeq, data)
                return cls.from_seq_lit(vals)

    @classmethod
    def from_dicts(cls, data: Sequence[Mapping[str, PythonLiteral]]) -> Self:
        return (
            Iter(data[0])
            .map(lambda key: (key, _into_tup(data, key)))
            .into(cls.from_dict)
        )

    @classmethod
    def from_seq_lit(cls, data: LitSeq) -> Self:
        rel = duckdb.values(_to_expr(COL0, tuple(data))).select(_unnest(COL0))
        return cls.from_relation(rel)

    @classmethod
    def from_seq_col(cls, data: NestedSeq) -> Self:
        return (
            Iter(data)
            .enumerate()
            .map_star(lambda k, v: (_named(k), v))
            .into(cls.from_dict)
        )

    @classmethod
    def from_seq_row(cls, data: NestedSeq) -> Self:
        width = len(data[0])
        return (
            Iter(range(width))
            .map(lambda j: (_named(j), _into_tup(data, j)))
            .into(cls.from_dict)
        )

    @property
    def identity(self) -> str:
        return f"bl_scan_{id(self.relation)}"

    def set_alias(self) -> Self:
        self.relation = self.relation.set_alias(self.identity)
        return self

    def into_frame(self) -> LazyFrame:
        from ._frame import LazyFrame

        return LazyFrame(self.relation)

    @classmethod
    def from_table(cls, name: str) -> Self:
        return cls.from_relation(duckdb.table(name))

    @classmethod
    def from_table_function(cls, name: str, *args: object) -> Self:
        return cls.from_relation(duckdb.table_function(name, *args))

    @classmethod
    def from_relation(cls, relation: DuckDBPyRelation) -> Self:
        schema = (
            Iter(relation.columns)
            .zip(relation.dtypes, strict=True)
            .map_star(lambda k, d: (k, exp.DataType.from_str(str(d), dialect="duckdb")))
            .collect(Dict)
        )

        return cls(relation, schema)


def _named(j: object) -> str:
    return f"column_{j}"


def _into_tup[T](
    vals: Iterable[Sequence[T]] | Iterable[Mapping[T, object]], key: T
) -> tuple[T, ...]:
    return Iter(vals).map(get(key)).collect(tuple)


def _to_expr(k: str, v: PythonLiteral) -> duckdb.Expression:
    return duckdb.ConstantExpression(v).alias(k)


def _unnest(k: str) -> duckdb.Expression:
    return duckdb.FunctionExpression("unnest", duckdb.ColumnExpression(k)).alias(k)
