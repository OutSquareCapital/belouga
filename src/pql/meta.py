"""metadata table functions."""

from . import sql as _sql
from ._frame import LazyFrame


def approx_database_count(*args: object) -> LazyFrame:
    """SQL approx_database_count table function.

    **SQL name**: *duckdb_approx_database_count*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.approx_database_count(*args))


def columns(*args: object) -> LazyFrame:
    """SQL columns table function.

    **SQL name**: *duckdb_columns*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.columns(*args))


def connection_count(*args: object) -> LazyFrame:
    """SQL connection_count table function.

    **SQL name**: *duckdb_connection_count*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.connection_count(*args))


def constraints(*args: object) -> LazyFrame:
    """SQL constraints table function.

    **SQL name**: *duckdb_constraints*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.constraints(*args))


def coordinate_systems(*args: object) -> LazyFrame:
    """SQL coordinate_systems table function.

    **SQL name**: *duckdb_coordinate_systems*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.coordinate_systems(*args))


def databases(*args: object) -> LazyFrame:
    """SQL databases table function.

    **SQL name**: *duckdb_databases*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.databases(*args))


def dependencies(*args: object) -> LazyFrame:
    """SQL dependencies table function.

    **SQL name**: *duckdb_dependencies*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.dependencies(*args))


def extensions(*args: object) -> LazyFrame:
    """SQL extensions table function.

    **SQL name**: *duckdb_extensions*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.extensions(*args))


def external_file_cache(*args: object) -> LazyFrame:
    """SQL external_file_cache table function.

    **SQL name**: *duckdb_external_file_cache*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.external_file_cache(*args))


def functions(*args: object) -> LazyFrame:
    """SQL functions table function.

    **SQL name**: *duckdb_functions*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.functions(*args))


def indexes(*args: object) -> LazyFrame:
    """SQL indexes table function.

    **SQL name**: *duckdb_indexes*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.indexes(*args))


def keywords(*args: object) -> LazyFrame:
    """SQL keywords table function.

    **SQL name**: *duckdb_keywords*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.keywords(*args))


def log_contexts(*args: object) -> LazyFrame:
    """SQL log_contexts table function.

    **SQL name**: *duckdb_log_contexts*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.log_contexts(*args))


def logs(denormalized_table: bool | None = None, *args: object) -> LazyFrame:  # noqa: FBT001
    """SQL logs table function.

    **SQL name**: *duckdb_logs*

    Args:
        denormalized_table (bool | None): Parameter
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.logs(denormalized_table, *args))


def memory(*args: object) -> LazyFrame:
    """SQL memory table function.

    **SQL name**: *duckdb_memory*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.memory(*args))


def optimizers(*args: object) -> LazyFrame:
    """SQL optimizers table function.

    **SQL name**: *duckdb_optimizers*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.optimizers(*args))


def prepared_statements(*args: object) -> LazyFrame:
    """SQL prepared_statements table function.

    **SQL name**: *duckdb_prepared_statements*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.prepared_statements(*args))


def schemas(*args: object) -> LazyFrame:
    """SQL schemas table function.

    **SQL name**: *duckdb_schemas*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.schemas(*args))


def secret_types(*args: object) -> LazyFrame:
    """SQL secret_types table function.

    **SQL name**: *duckdb_secret_types*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.secret_types(*args))


def secrets(redact: bool | None = None, *args: object) -> LazyFrame:  # noqa: FBT001
    """SQL secrets table function.

    **SQL name**: *duckdb_secrets*

    Args:
        redact (bool | None): Parameter
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.secrets(redact, *args))


def sequences(*args: object) -> LazyFrame:
    """SQL sequences table function.

    **SQL name**: *duckdb_sequences*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.sequences(*args))


def settings(*args: object) -> LazyFrame:
    """SQL settings table function.

    **SQL name**: *duckdb_settings*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.settings(*args))


def tables(*args: object) -> LazyFrame:
    """SQL tables table function.

    **SQL name**: *duckdb_tables*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.tables(*args))


def temporary_files(*args: object) -> LazyFrame:
    """SQL temporary_files table function.

    **SQL name**: *duckdb_temporary_files*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.temporary_files(*args))


def types(*args: object) -> LazyFrame:
    """SQL types table function.

    **SQL name**: *duckdb_types*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.types(*args))


def variables(*args: object) -> LazyFrame:
    """SQL variables table function.

    **SQL name**: *duckdb_variables*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.variables(*args))


def views(*args: object) -> LazyFrame:
    """SQL views table function.

    **SQL name**: *duckdb_views*

    Args:
        *args (object): Variable arguments

    Returns:
        LazyFrame
    """
    return LazyFrame(_sql.meta.views(*args))
