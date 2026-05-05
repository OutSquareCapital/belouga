
# Contributing to `pql`

Thank you for your interest in contributing to `pql`!

Contributions are always welcome, whether it's a bug fix, a new feature, or just improving the documentation.

## Testing

The project heavily compares `pql` behavior against reference Polars chains where parity is expected.

We want to periodically check the coverage. To do so, run:

```shell
uv run pytest tests/ --cov=src/ --cov-report=term-missing
```

## Architecture

`pql` exposes a Polars-like lazy API on top of DuckDB, with `sqlglot` used as the SQL AST layer.

The two main public objects are `LazyFrame` and `Expr`, but the project is organized in a few distinct layers.

### Public API layer

The public surface lives mostly in:

- `src/pql/_frame.py` for `LazyFrame`
- `src/pql/_expr.py` for `Expr`
- `src/pql/_funcs.py` for module-level expression helpers such as `col`, `lit`, `when`, `coalesce`, and aggregations
- `src/pql/_scans.py` for constructors such as `from_df`, `from_dict`, `from_query`, `from_table`, and `from_table_function`
- `src/pql/__init__.py` for package-level re-exports

This is the user-facing layer and should remain Polars-like in ergonomics.

### Core wrappers and SQL building

`src/pql/_core.py` contains the low-level wrappers shared across the project:

- `CoreHandler[T]` is the generic wrapper base used by the main fluent objects
- `ExprHandler` specializes `CoreHandler` for `sqlglot.exp.Expr`
- `NameSpaceHandler[T]` is the shared base for expression namespaces that return the parent expression type
- `into_expr`, `into_expr_list`, `anon`, `anon_agg`, and `func` are the main SQL expression builders and coercion helpers

This layer is responsible for normalizing Python values into `sqlglot` nodes and keeping the fluent API compact.

### Expression layer

`Expr` lives in `src/pql/_expr.py`.

It wraps a `sqlglot.exp.Expr` and extends the generated `Fns` mixin from `src/pql/_fns.py`, which provides DuckDB function wrappers.

`Expr` also carries `ExprMeta` from `src/pql/_meta.py`. This metadata is important for aliasing, naming resolution, selectors support, and context-sensitive behavior when the expression is used inside a frame operation.

The namespace entry points exposed on `Expr` are implemented in `src/pql/namespaces.py`, including `.str`, `.list`, `.struct`, `.dt`, `.arr`, `.json`, `.re`, `.map`, `.enum`, `.geo`, and `.name`.

### Frame and query layer

`LazyFrame` lives in `src/pql/_frame.py` and is the main query builder.

It inherits from `CoreHandler[sqlglot.exp.Query]` and stores:

- `_inner`: the current `sqlglot` query AST
- `_sources`: the underlying scan sources used to materialize the query
- `_schema`: the inferred schema tracked across transformations

Most relational operations are implemented here, including `select`, `with_columns`, `filter`, `group_by`, `join`, `pivot`, `sort`, and execution helpers such as `collect()` and `lazy()`.

`LazyFrame` does not directly hold a single long-lived `DuckDBPyRelation`. Instead, it builds query ASTs and materializes them through scan sources when execution is needed.

### Relation and input normalization

`src/pql/_scans.py` contains `ScanSource`, which is the bridge between the query AST world and executable DuckDB relations.

`ScanSource` wraps a `duckdb.DuckDBPyRelation` together with schema metadata and is responsible for normalizing the different supported inputs:

- DuckDB relations
- Polars and Narwhals frames
- dictionaries and sequences
- NumPy arrays
- SQL queries, tables, and table functions

`ScanSource.from_query(...)` is the main execution boundary: it turns a `sqlglot` query into an executable DuckDB relation, which is then converted back to Polars objects by `LazyFrame.collect()` and `LazyFrame.lazy()`.

### Supporting modules

Some important supporting modules are:

- `src/pql/_when.py` for the fluent `when(...).then(...).otherwise(...)` builder
- `src/pql/_window.py` for window specification and rolling/window logic
- `src/pql/_groupby.py` for grouped-frame operations
- `src/pql/_joins.py` for join key normalization and join construction helpers
- `src/pql/_parser.py` for parsing SQL strings into query objects
- `src/pql/selectors.py` for selectors
- `src/pql/datatypes.py` for the public datatype objects and conversions

### Generated code

Two important files are generated and should not be edited by hand:

- `src/pql/_fns.py` for DuckDB function wrappers and generated namespace mixins
- `src/pql/meta.py` for DuckDB meta table-function helpers

If a generated API needs to change, update the generator logic in `scripts/` and regenerate the file instead of patching the generated output directly.

## Scripts

Scripts are used for code generation and API comparison at dev time.
They are not meant to be used by end users, and are not part of the public API.

More infos with the following command:

```shell
uv run -m scripts --help
```

### Comparator

The **compare** command creates the [coverage](API_COVERAGE.md) report used to compare the `pql`, `polars`, and `narwhals` APIs.

### Generators

The generators are driven from `scripts/` and produce source files used by the public API.

The main outputs are:

- [DuckDB function wrappers and namespace mixins](src/pql/_fns.py)
- [DuckDB meta table-function helpers](src/pql/meta.py)
- [The SQL display theme literal](src/pql/typing.py)

**Note** that if you never generated the DuckDB function wrappers before, you need to run `fns-to-parquet` once to build the cached metadata file, and then `gen-fns` to generate the wrappers.

## References

- [DuckDB functions](https://duckdb.org/docs/stable/sql/functions/overview)

## Known bug: `DuckDB` → `polars.LazyFrame` panic on `dynamic_predicate`

> **Versions**: Polars 1.39.3, DuckDB 1.5.2.dev40

### Summary

`pql.LazyFrame.lazy()` produces a Polars `LazyFrame` backed by a **`PYTHON SCAN`** (via `duckdb/polars_io.py`).
Certain Polars operations that internally generate a `dynamic_predicate` optimization node cause a **panic** when collected.

**Affected operations:** `.sort().limit()`, `.sort().head()`, `.top_k()`, `.bottom_k()`

**Workaround:** `.collect().lazy()` works — it materializes to an in-memory `DataFrame` first, so the plan uses a native `DF [...]` scan instead of `PYTHON SCAN`.

### Mechanism

1. Polars optimizes `sort + limit` into a single node with a `dynamic_predicate` — an internal filter that pre-screens rows before the full sort.
2. This predicate gets pushed down to the DuckDB IO source plugin as the `predicate` callback argument.
3. `_predicate_to_expression` in `polars_io.py` fails to convert the `dynamic_predicate` node to a DuckDB expression (correctly suppressed via `contextlib.suppress`).
4. The fallback path (`polars_io.py:307`) calls `pl.from_arrow(batch).filter(predicate)`, which internally does `.lazy().filter(predicate).collect()`.
5. The `dynamic_predicate` expression is an optimizer-internal node — Polars' own `expr_to_ir` converter doesn't handle it → **panic** at `expr_to_ir.rs:627`.

### Responsibility

This is a **DuckDB `polars_io` plugin bug**: the fallback filter path doesn't account for optimizer-internal predicate nodes that cannot be evaluated as user-level expressions.
