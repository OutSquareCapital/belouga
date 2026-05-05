
# pql

`pql` is a lazy dataframe library for DuckDB with a Polars-like API. It compiles expression trees to DuckDB SQL via `sqlglot` and returns native Polars objects on collect.

It targets the case where DuckDB is the execution engine, and you want a fluent dataframe API close to Polars rather than handwritten SQL or a generic multi-backend abstraction.

## Quick Start

### Installation

```shell
uv add https://github.com/OutSquareCapital/pql.git
```

### Example

```python
import pql

data = {
    "city": ["Paris", "Paris", "Berlin", "Berlin"],
    "price": [100, 120, 80, 90],
    "qty": [1, 2, 3, 4],
    "is_promo": [False, True, False, True],
}
query = (
    pql
    .from_dict(data)
    .filter(pql.col("price").ge(90))
    .with_columns(revenue=pql.col("price").mul("qty"))
    .group_by("city")
    .agg(
        total_revenue=pql.col("revenue").sum(),
        avg_price=pql.col("price").mean(),
        promo_rows=pql.col("is_promo").sum(),
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
import pql

query = pql.LazyFrame({"x": [1, 2, 3]}).filter(pql.col("x").gt(1))
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

## API

The two core types are `LazyFrame` (relational operations) and `Expr` (expression trees). Both are used in the same way as their Polars counterparts.

`LazyFrame` can be built from Python objects, NumPy arrays, any Narwhals-compatible frame, DuckDB relations, named tables, and DuckDB table functions. It supports `select`, `with_columns`, `filter`, `sort`, `join`, `group_by`, `pivot`, `unpivot`, `sink_*`, and more.

`Expr` exposes 700+ DuckDB-backed methods, including namespaces for strings (`.str`), lists (`.list`), arrays (`.arr`), structs (`.struct`), datetimes (`.dt`), JSON (`.json`), regex (`.re`), maps (`.map`), enums (`.enum`), geospatial (`.geo`), and naming (`.name`).

Module-level helpers cover the usual entry points: `col`, `lit`, `when`, `coalesce`, scalar and horizontal aggregations.

`pql.selectors` mirrors the Polars selectors API. `pql.datatypes` exposes DuckDB-aligned type objects used for casts and schema work, including `Geometry`.

## Notable Features

### DuckDB function coverage

`pql` exposes 700+ DuckDB-backed expression methods, covering most of what DuckDB's function catalog offers in a fluent chainable style.

### Native geometry support

The `Geometry` datatype and `.geo` namespace expose DuckDB's spatial functions directly. This is not something Polars targets.

### `group_by_all()`

`LazyFrame.group_by_all()` maps to DuckDB's `GROUP BY ALL`, which is convenient when the grouping columns are all non-aggregated ones.

### Pyochain integration

Iterable-returning methods return `pyochain` objects, so column lists and schema views stay chainable:

```python
import pql

lf = pql.LazyFrame({"price": [1, 2, 3], "name": ["x", "y", "z"]})

cols = lf.columns.iter().filter(lambda col: col.startswith("p"))
result = lf.select(cols).columns
print(result)

# PyoKeysView(Dict('price': DataType(this=DType.INT, nested=False)))
```

## Differences vs Polars

`pql` follows Polars conventions where they translate cleanly to DuckDB, and deviates where they don't. See [API_COVERAGE.md](API_COVERAGE.md) for the full method matrix.

**Structural:** lazy-only — no eager `DataFrame`. `.collect()` and `.lazy()` return native Polars objects. Cross joins use `join_cross()` instead of `join(how="cross")`.

**Semantics:** null handling, ordering, and some aggregation behavior follow DuckDB. Logical operators follow SQL semantics. `Categorical` is not supported.

**Signatures:** some methods (`collect`, `explain`, `group_by`, `join`, `join_asof`, `pivot`, `unique`, `with_row_index`) have different signatures because they expose DuckDB-specific options rather than mimicking Polars exactly.

**Gaps:** async sinks, several serialization helpers, and some expression methods are not yet implemented. Coverage is tracked in [API_COVERAGE.md](API_COVERAGE.md).

## DuckDB catalog access

`pql` can start from DuckDB tables and table functions directly, which makes catalog introspection straightforward:

```python
import pql

(
    pql.meta
    .functions()
    .filter(pql.col("function_name").str.contains("json"))
    .select("function_name", "parameter_types", "return_type")
    .sort("function_name")
    .limit(3)
    .show()
)

# ┌───────────────┬────────────────────┬─────────────┐
# │ function_name │  parameter_types   │ return_type │
# │    varchar    │     varchar[]      │   varchar   │
# ├───────────────┼────────────────────┼─────────────┤
# │ array_to_json │ []                 │ JSON        │
# │ from_json     │ [VARCHAR, VARCHAR] │ ANY         │
# │ from_json     │ [JSON, VARCHAR]    │ ANY         │
# └───────────────┴────────────────────┴─────────────┘
```

## Comparison with other tools

**Narwhals** is a compatibility layer aimed at library authors who want to write dataframe-agnostic code that runs across Polars, pandas, and other backends. The API is Polars-inspired but intentionally limited to what can be expressed portably — it is not trying to expose deep DuckDB surface. End users doing data work are not the primary audience.

**Ibis** targets portability across 20+ backends (DuckDB, BigQuery, Snowflake, Spark, ...) under a single Ibis-native API. It also uses `sqlglot` internally and can use DuckDB as a local backend. The tradeoff is that the API stays generic enough to compile to all those targets, so DuckDB-specific functionality is not exposed. If you need the same query graph to run on multiple engines, Ibis is the right tool.

**SQLFrame** implements the PySpark DataFrame API on top of SQL engines. The syntax is PySpark-first — `withColumn`, `F.col`, `SparkSession` — not Polars-like. It is designed for teams who want to run PySpark transformation pipelines on DuckDB, BigQuery, or Snowflake without an actual Spark cluster.

`pql` sits in a different spot: Polars-like syntax, DuckDB as the fixed target, and access to the full DuckDB function surface (700+ methods, geospatial, `GROUP BY ALL`, catalog introspection) that generalist multi-backend tools do not expose.

## How It Works

`pql` compiles `Expr` and `LazyFrame` operations into a `sqlglot` AST, then materializes queries through `ScanSource` against a DuckDB relation. Generated code in `src/pql/_fns.py` covers most of the DuckDB function catalog.

## Contributing

If you want to contribute, start with [CONTRIBUTING.md](CONTRIBUTING.md).
