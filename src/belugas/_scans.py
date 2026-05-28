from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import DuckDBPyConnection
from pyochain import option

from ._frame import LazyFrame
from ._plan import nodes
from .typing import CSVOptions, JsonOptions, ParquetOptions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    from duckdb import DuckDBPyRelation

    from .typing import (
        AnyArray,
        FileGlob,
        IntoArrow,
        IntoDict,
        IntoPolars,
        Orientation,
        PathOrBuffer,
        PythonLiteral,
        SeqIntoVals,
    )
type Conn = DuckDBPyConnection | None
"""Type alias for a DuckDB connection that can be optionally provided to scan functions."""


def from_query(query: DuckDBPyRelation, connection: Conn = None) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), query))


def from_table(table: str, connection: Conn = None) -> LazyFrame:
    return _from_node(nodes.ScanTable(option(connection), table))


def from_table_function(function: str, connection: Conn = None) -> LazyFrame:
    return _from_node(nodes.ScanTableFunction(option(connection), function))


def from_numpy(
    arr: AnyArray, orient: Orientation = "col", connection: Conn = None
) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), arr, orient=orient))


def from_dict(
    mapping: IntoDict[str, PythonLiteral], connection: Conn = None
) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), mapping))  # pyright: ignore[reportUnknownArgumentType, reportArgumentType]


def from_dicts(
    data: Sequence[Mapping[str, PythonLiteral]], connection: Conn = None
) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), data))


def from_records(
    data: SeqIntoVals, orient: Orientation = "col", connection: Conn = None
) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), data, orient=orient))


def from_pandas(df: pd.DataFrame, connection: Conn = None) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), df))


def from_polars(df: IntoPolars, connection: Conn = None) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), df))


def from_arrow(df: IntoArrow, connection: Conn = None) -> LazyFrame:
    return _from_node(nodes.ScanInMemory(option(connection), df))


def scan_parquet(  # noqa: PLR0913
    file_glob: FileGlob,
    /,
    *,
    binary_as_string: bool = False,
    file_row_number: bool = False,
    filename: bool = False,
    hive_partitioning: bool = False,
    union_by_name: bool = False,
    compression: ParquetCompression | None = None,
    connection: Conn = None,
) -> LazyFrame:
    options = ParquetOptions(
        binary_as_string=binary_as_string,
        file_row_number=file_row_number,
        filename=filename,
        hive_partitioning=hive_partitioning,
        union_by_name=union_by_name,
        compression=compression,
    )
    return _from_node(nodes.ScanParquet(option(connection), file_glob, options))


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
    connection: Conn = None,
) -> LazyFrame:
    options = CSVOptions(
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
    return _from_node(nodes.ScanCSV(option(connection), path_or_buffer, options))


def scan_json(  # noqa: PLR0913
    path_or_buffer: PathOrBuffer,
    *,
    columns: ColumnsTypes | None = None,
    sample_size: int | None = None,
    maximum_depth: int | None = None,
    records: JsonRecordOptions | None = None,
    format: JsonFormat | None = None,  # noqa: A002
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
    connection: Conn = None,
) -> LazyFrame:
    options = JsonOptions(
        columns=columns,
        sample_size=sample_size,
        maximum_depth=maximum_depth,
        records=records,
        format=format,
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
    return _from_node(nodes.ScanJson(option(connection), path_or_buffer, options))


def _from_node(scan: nodes.Scan) -> LazyFrame:
    """Helper function to create a LazyFrame from a scan node.

    Allow us to avoid implementing a check for nodes in the `LazyFrame::__init__` method,

    and to clearly state that this is a private function used only here (as opposed to creating a new dunder/private constructor on `LazyFrame`).

    Returns:
        LazyFrame
    """
    out = LazyFrame.__new__(LazyFrame)
    out._inner = scan  # pyright: ignore[reportPrivateUsage]
    return out
