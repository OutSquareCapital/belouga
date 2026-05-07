
# belugas

<p align="center">
    <img src="docs/Bélouga.jpg" alt="Belouga" width="260" />
    <br/>
</p>

`belugas` is a dataframe library with a `DuckDB` backend.

Just like his arctic cetacean cousin [narwhals](https://github.com/narwhals-dev/narwhals), `belugas` API is inspired by [polars](https://github.com/pola-rs/polars), with a fluent, lazy API, focused on building reusable expressions chains that can be executed once passed in a `LazyFrame` context.

Contrary to `narwhals` or [`Ibis`](https://github.com/ibis-project/ibis), `belugas` is a specialized tool, aiming to expose the full power of `DuckDB` possibilities, in a syntax familiar to `polars` users, without concessions on functionnality.

It is not trying to be portable across multiple backends, but rather to be the best possible interface for users who want to write expressive queries in Python and execute them on `DuckDB`.

## Features

The two main pillars of `belugas` are:

- `Expr`, the base class for all expressions, created from `col`, `lit`, selectors, and others module functions, that can be combined together to build complex expressions chains.
- `LazyFrame`, the main entry point for all data manipulation, that can be created from various data sources (arrow, python mapping/sequences, pandas dataframe, CSV, parquet, json etc...), and on which you can call all the available methods to build your query.

`belugas` currently provides:

- `LazyFrame::{select, filter, with_columns, join, join_asof, pivot, unpivot, sort, sink_csv, sink_parquet, ...}`
- Aggregations contexts with `LazyFrame::group_by::{agg, all, len, ...}`
- A **rich expressions catalog covering +700 of DuckDB's built-in functions**, with custom expressions implementations for `polars` functionnalities that `DuckDB` currently don't provide
- `Expr` namespaces for **geospatial, json, map, regex functions, and more**
- A `when/then/otherwise` expression builder for complex conditional logic
- A complete family of Datatypes, including `Enums`, `List`, `Array`, `Struct`, `Map`, and `Geometry`, with the same ergonomics of the polars library
- selectors by dtypes, by name, regex, and more, just like `polars`
- Conversions to `LazyFrame` from python Mapping and Sequence, numpy arrays, pandas and polars dataframes, and more.
- Various module level functions, like `unnest`, `scan_csv`, `coalesce`, `all`, and more.
- Query introspections, with syntax highlighted SQL in your terminal with +20 color themes, sql formatting, AST inspection, query plan inspection, and more.
- A `meta` module with all the table and metadata functions provided by `DuckDB` to inspect your database schema, list functions, and more.

## Quick Start

### Installation

```shell
uv add https://github.com/OutSquareCapital/belugas.git
```

### Example

```python
import belugas as bl

data = {
    "city": ["Paris", "Paris", "Berlin", "Berlin"],
    "price": [100, 120, 80, 90],
    "qty": [1, 2, 3, 4],
    "is_promo": [False, True, False, True],
}
query = (
    bl
    .from_dict(data)
    .filter(bl.col("price").ge(90))
    .with_columns(revenue=bl.col("price").mul("qty"))
    .group_by("city")
    .agg(
        total_revenue=bl.col("revenue").sum(),
        avg_price=bl.col("price").mean(),
        promo_rows=bl.col("is_promo").sum(),
    )
    .sort("total_revenue", descending=True)
)
```

Output:

```shell
 ┌─────────┬───────────────┬───────────┬────────────┐
 │  city   │ total_revenue │ avg_price │ promo_rows │
 │ varchar │    int128     │  double   │   int128   │
 ├─────────┼───────────────┼───────────┼────────────┤
 │ Berlin  │           360 │      90.0 │          1 │
 │ Paris   │           340 │     110.0 │          1 │
 └─────────┴───────────────┴───────────┴────────────┘
```

You can inspect the generated SQL query directly, format it, with syntax highlighting and various available themes:

```python
import belugas as bl

query = bl.LazyFrame({"x": [1, 2, 3]}).filter(bl.col("x").gt(1))
sql = query.sql_query()

sql.show("friendly")
print("---" * 10)
sql.show(pretty=True)
```

![alt text](docs/sql_highlight.png)

You can also inspect the DuckDB plan:

```python
print(query.explain())
```

Output:

```shell
┌───────────────────────────┐
│         PROJECTION        │
│    ────────────────────   │
│             #1            │
│                           │
│          ~0 rows          │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│           FILTER          │
│    ────────────────────   │
│          (x > 1)          │
│                           │
│          ~0 rows          │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│           UNNEST          │
└─────────────┬─────────────┘
┌─────────────┴─────────────┐
│      COLUMN_DATA_SCAN     │
│    ────────────────────   │
│           ~1 row          │
└───────────────────────────┘
```

## Dependencies

### DuckDB

`belugas` uses [DuckDB](https://duckdb.org/) as the execution engine.

### sqlglot

[`sqlglot`](https://github.com/tobymao/sqlglot) is used to build and manipulate SQL ASTs for the IR between `LazyFrame`/`Expr` operations and the generated SQL queries.

### Pyochain

[`pyochain`](https://github.com/OutSquareCapital/pyochain) is used for iterable-returning methods, and internal implementations.

This means that column lists and schema views stay chainable:

```python
import belugas as bl

lf = bl.LazyFrame({"price": [1, 2, 3], "name": ["x", "y", "z"]})

cols = lf.columns.iter().filter(lambda col: col.startswith("p"))
result = lf.select(cols).columns
print(result)

# PyoKeysView(Dict('price': DataType(this=DType.INT, nested=False)))
```

## Comparison with other tools

**DuckDB Relational API** is the native way to interact with DuckDB in Python, but as it currently stands, `belugas` offers much more possibilities. For example, `LazyFrame::{unpivot, join_asof, pivot}` are not available in the relational API, and the function catalog is only available trough raw str passed to functions, without hover documentation nor type safety on the argument types.

### Narwhals

**Narwhals** is a compatibility layer aimed at library authors who want to write dataframe-agnostic code that runs across Polars, pandas, and other backends. The API is Polars-inspired but intentionally limited to what can be expressed portably — it is not trying to expose deep DuckDB surface. End users doing data work are not the primary audience.

### Ibis

**Ibis** targets portability across 20+ backends (DuckDB, BigQuery, Snowflake, Spark, ...) under a single Ibis-native API. It also uses `sqlglot` internally and can use DuckDB as a local backend. The tradeoff is that the API stays generic enough to compile to all those targets, so DuckDB-specific functionality is not exposed. If you need the same query graph to run on multiple engines, or don't wan't to purely use `DuckDB` and polars, Ibis is the right tool.
It's also closer to polars than `SQLFrame` in terms of API design (take this with a grain of salt, I haven't used any of those libraries extensively, just browsed their docs and codebase).

### SQLFrame

**SQLFrame** implements the PySpark DataFrame API on top of SQL engines. The syntax is PySpark-first — `withColumn`, `F.col`, `SparkSession` — not Polars-like. It is designed for teams who want to run PySpark transformation pipelines on DuckDB, BigQuery, or Snowflake without an actual Spark cluster.

## Contributing

If you want to contribute, start with [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

- `DuckDB` for building an amazing analytical database engine.
- `sqlglot` for the amazing work on SQL parsing and AST manipulation, that made it possible to build a powerful IR for `belugas` without having to write a SQL parser from scratch. Also hats off to the `sqlglot` team who was very reactive on my contributions and feature requests.
- `narwhals` for the inspiration on the API design and the idea of building a dataframe library on top of `DuckDB`.
- [@MarcoGorelli](https://github.com/MarcoGorelli) for the library name suggestion!
