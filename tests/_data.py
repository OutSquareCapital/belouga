from datetime import date, datetime, time

import duckdb
import polars as pl

import belugas as bl

nan = float("nan")

_DATA = {
    "a": [True, False, True, None, True, False],
    "b": [True, True, False, None, True, False],
    "x": [-10, 2, -3, 5, -10, 20],
    "uint": [1, 2, 3, None, 1, 2],
    "enum": ["foo", "bar", "baz", None, "foo", "bar"],
    "float_vals": [1.3652, 2.7525, 3.7314, None, 1.3685, 2.7785],
    "decimal_vals": [1.3652, 2.7525, 3.7314, None, 1.3685, 2.7785],
    "n": [None, 3, 1, None, 2, 3],
    "s": ["1", "2", "3", None, "1", "2"],
    "age": [25, 30, 35, None, 25, 30],
    "salary": [50000.0, 60000.0, 70000.0, None, 50000.0, 60000.0],
    "nested": [[1, 2], [3, 4], [5], None, [1, 2], [3, 4]],
    "nan_vals": [1.0, nan, 3.0, nan, 5.0, nan],
    "arr_str_vals": [
        ["g", "b", "c"],
        ["a", "b", "a"],
        ["c", "c", "c"],
        ["d", "e", "d"],
        ["g", "b", "c"],
        ["a", "b", "a"],
    ],
    "arr_booleans": [
        [True, False, True],
        [True, True, True],
        [False, False, False],
        [True, None, True],
        [True, False, True],
        [True, True, True],
    ],
    "arr_num": [
        [1, 2, 7, 3],
        [3, 4, 5, 5],
        [2, 5, 8, 1],
        [1, 2, 3, 4],
        [1, 2, 7, 3],
        [3, 4, 5, 5],
    ],
    "list_num": [
        [1, 2, 7, 3],
        [3, 4, 5, 5],
        [2, 5],
        [1],
        [1, 2],
        [3, 4, 5],
    ],
    "list_booleans": [
        [True, False, True],
        [True, True],
        [False],
        [True, None],
        [True, True, True, True, True, True, False],
        [False, False, False, False, False, False, True],
    ],
    "list_str_vals": [["g", "b", "c"], ["a", "b"], ["c"], [], ["hello", "world"], [""]],
    "structs": [
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 5, "b": 6, "c": 7, "d": 8},
        {"a": 5, "b": 6, "c": 7, "d": 8},
        {"a": 5, "b": 6, "c": 7, "d": 8},
        {"a": 5, "b": 6, "c": 7, "d": 8},
        {"a": 5, "b": 6, "c": 7, "d": 8},
    ],
    "d": [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
        date(2024, 1, 1),
        date(2024, 1, 2),
    ],
    "dt": [
        datetime(2024, 1, 1, 10, 30, 15, 123_456),
        datetime(2024, 1, 2, 11, 45, 30, 1),
        datetime(2024, 1, 3, 23, 59, 59, 999_001),
        datetime(2024, 1, 4, 0, 0, 0, 0),
        datetime(2024, 1, 1, 10, 30, 15, 123_456),
        datetime(2024, 1, 2, 11, 45, 30, 1),
    ],
    "binary": [b"foo", b"bar", b"baz", None, b"foo", b"bar"],
    "time": [
        time(10, 30, 15, 123_456),
        time(11, 45, 30, 1),
        time(23, 59, 59, 999_001),
        time(0, 0, 0, 0),
        time(10, 30, 15, 123_456),
        time(11, 45, 30, 1),
    ],
}
_SCHEMA = {
    "uint": pl.UInt16(),
    "decimal_vals": pl.Decimal(10, 4),
    "binary": pl.Binary(),
    "arr_booleans": pl.Array(pl.Boolean, shape=3),
    "arr_str_vals": pl.Array(pl.String, shape=3),
    "arr_num": pl.Array(pl.UInt16, shape=4),
}
_DF = pl.DataFrame(_DATA, schema_overrides=_SCHEMA).pipe(duckdb.from_arrow)
_DF_PQL = bl.LazyFrame(_DF)
_LF_PL = _DF.pl(lazy=True)
LF_TEST = pl.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "sex": ["F", "M", "M", "M", "F"],
    "age": [25, 30, 35, 28, 22],
    "salary": [50000.0, 60000.0, 75000.0, 55000.0, 45000.0],
    "department": [
        "Engineering",
        "Sales",
        "Engineering",
        "Sales",
        "Engineering",
    ],
    "is_active": [True, True, False, True, True],
    "value": [10.0, None, 30.0, None, 50.0],
    "category": ["A", "B", None, "A", "B"],
}).pipe(bl.from_arrow)


def sample_lf() -> pl.LazyFrame:
    return _LF_PL


def sample_bl() -> bl.LazyFrame:
    return _DF_PQL
