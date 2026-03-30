
# PQL, write polars syntax, create SQL queries for DuckDB

This package is **WIP** and ultimately aim to provide a polars like API for [DuckDB Python API](https://duckdb.org/docs/stable/clients/python/overview), as well as full support of the [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview).

## Differences and additions compared to polars

Altough `pql` aims to be as close as possible to polars, some differences exists.
Sometimes they are due to hard limitations of duckdb (e.g `Categorical` datatypes), sometimes they are just deliberate design choices (e.g cross join strategy).
Some of those are listed here, but for a more comprehensive list, see the [API coverage](API_COVERAGE.md) report.

### Differences

- `DataFrame` don't exist. Only `LazyFrame` is implemented, as it is the only one that can be implemented with duckdb.
- To convert to polars, you can do:

```python
import pql

lf_pql = pql.LazyFrame({...})
lf_polars = lf_pql.lazy()  # equivalent to DuckDBPyRelation.pl(lazy=True)
df_polars = lf_pql.collect()  # equivalent to DuckDBPyRelation.pl(lazy=False)
```

- `LazyFrame.join()` don't have a `"cross"` strategy. Instead, call `LazyFrame.join_cross()`. This is a deliberate choice, because:
  - `duckdb` natively have differents methods for join/cross_join
  - The internal implementation is simpler and cleaner if we don't have to handle the cross join as a special case of the regular join
  - The public API is clearer, as *on*, *left_on* and *right_on* parameters don't make sense for a cross join, and it is better to not have them in the signature of the method, rather than throwing runtime errors if they are used with a cross join strategy.
- `Categorical` datatypes are not supported (this is not representable in duckdb).

### Additions from polars

- Full support of the `GEOMETRY` datatypes and functions, as they are [natively supported in duckdb](https://duckdb.org/docs/current/sql/data_types/geometry)
- `LazyFrame.group_by_all()` method -> [see more here](https://duckdb.org/docs/stable/sql/query_syntax/groupby#group-by-all)
- columns/schema, and other methods/properties who return plain python `Iterable` return [pyochain objects](https://outsquarecapital.github.io/pyochain/). This allows you to use all the methods of those objects, whilst keeping the same method chaining style than with `Expression/LazyFrame`. For example, you can do:

```python
>>> data = {"price": [1, 2, 3], "name": ["x", "y", "z"]}
>>> lf = pql.LazyFrame(data)
# get the columns as a pyochain object
>>> cols = lf.columns.iter().filter(lambda col: col.startswith("p"))
>>> lf.select(cols).columns
Vec("price",)

```

## Comparison to other tools

### Narwhals

[narwhals](https://github.com/narwhals-dev/narwhals) aims to support more functionnality from polars AND all of those from duckdb, as pql is not limited by multiple backend compatibility.
Furthermore, `narwhals` is primarly designed for library developpers who want integration with multiple dataframe libraries, not for end users.
Narwhals support a **subset** of polars API, hence necessarily a **subset** of DuckDB API, while pql aims to support the **full** API of both.

### SQLFrame

[SQLFrame](https://github.com/eakmanrq/sqlframe) is fundamentally a PySpark oriented library API-wise.

### Ibis

Ibis has a different syntax from polars. It can be close for some operations, but totally different for others.
Also the goal isn't the same, as Ibis is more focused on providing a high level API for multiple backends, while pql is focused on providing a polars like API for DuckDB.

## Architecture

`pql` two main public classes are `LazyFrame` and `Expr`.

### Expr

Expressions are the base building blocks of the API.
an `Expr` is a wrapper around an internal `SqlExpr` class.
This responsibility separation allows to separate metadata handling (column names resolution mainly, `Selectors` implementation, etc.. ), and internal implementation of custom expressions (e.g `Expr.str.titlecase()`).
`SqlExpr` in turn wraps a `sqlglot.Expression` object, which is the AST used to generate the final SQL query.
Once needed, the `sqlglot.Expression` is converted to a native `duckdb.Expression` object, which is the one used to execute the query.

### LazyFrame

This class wraps a `duckdb.DuckDBPyRelation` object, and is the main entry point for users.
It provides methods that give context for the `Expr` objects, and also handle the final SQL query generation and execution.

## Scripts

Scripts are used for code generation and API comparison at dev time.
They are not meant to be used by end users, and are not part of the public API.

More infos with the following command:

```shell
uv run -m scripts --help
```

### Comparator

The **compare** command will create the [coverage](API_COVERAGE.md) report to compare `pql` vs `polar`s and `narwhals` API's.

### Generators

The **gen-{fns, themes}** commands will respectively generate python code for:

- [The functions from the `table_functions` DuckDB table](src/pql/sql/_code_gen/_fns.py)
- [A `Literal` for SQL display theming](src/pql/_typing.py) (see `Theme` type)

**Note** that if you never generated the `table_functions` code, you need first to run `fns-to_parquet` once to get the parquet file with the data casted and updated, and then `gen-fns` to generate the code.

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)
