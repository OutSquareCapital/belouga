---
description: Instructions for the PQL project agents.
applyTo: '*'
---
# AGENTS Instructions for `belugas`

## Project mission

`belugas` exposes DuckDB through a Polars-like lazy API built on `sqlglot`.

Primary objective:

- Provide a high-level public API (`belugas.Expr`, `belugas.LazyFrame`) that compiles to efficient DuckDB-native expressions/relations.

Secondary objective:

- Keep parity visibility against Narwhals/Polars via generated coverage reports.

---

## Current public surface

`belugas` currently exposes:

- `LazyFrame` in `src/belugas/_frame.py`.
- `Expr` in `src/belugas/_expr.py`, re-exported from `src/belugas/__init__.py`.
- Public scan constructors in `src/belugas/_scans.py` and at package root (`from_df`, `from_dict`, `from_query`, `from_table`, `from_table_function`, ...).
- Module-level expression helpers in `src/belugas/_funcs.py` (`col`, `lit`, `when`, `coalesce`, scalar aggs, horizontal aggs, `element`, `row_number`, ...).
- `selectors`, `meta`, and datatype objects re-exported from the package root.
- `Expr` lives in `src/belugas/_expr.py` and the public frame type is `LazyFrame`.
- Relation handling is organized around `LazyFrame` plus `ScanSource`.

---

## Architecture (must understand before changing code)

### 1) Public API layer

- `src/belugas/_frame.py`: public `LazyFrame` API.
- `src/belugas/_scans.py`: public constructors for turning Python/native data into `LazyFrame`.
- `src/belugas/__init__.py`: exported user-facing symbols.
- `src/belugas/_expr.py`: public `Expr` API, re-exported at the package root.

This layer should remain Polars-like and user ergonomic.

### 2) Frame/query layer

#### `LazyFrame` (`src/belugas/_frame.py`)

`LazyFrame` is the main query builder.

- Inherits from `CoreHandler[sqlglot.exp.Query]`.
- Stores `_inner: sqlglot.exp.Query`, `_sources`, and `_schema`.
- Builds query context for expressions (`select`, `with_columns`, `filter`, `group_by`, `join`, `pivot`, `sort`, ...).
- Executes through `ScanSource.from_query(...)`.
- Converts back to Polars through DuckDB relation interop in `lazy()` and `collect()`.

`LazyFrame` plus `ScanSource` form the relation/query abstraction.

### 3) Expression layer

#### `Expr` (`src/belugas/_expr.py`)

`Expr` is the public expression object.

- Extends the generated `Fns` mixin from `src/belugas/_fns.py`.
- Wraps a `sqlglot.exp.Expr` through `ExprHandler`/`CoreHandler`.
- Carries `aliaser: AliasMapper` to preserve naming, aliasing, and context-sensitive behavior.
- Uses `Expr.new(...)` as the normal coercion entrypoint.
- Exposes namespaces such as `.str`, `.list`, `.struct`, `.dt`, `.arr`, `.json`, `.re`, `.map`, `.enum`, `.geo`, and `.name`.

The expression layer is where most feature work happens.

Important sharp edge:

- Reverse literal operators rely on `Marker.LITERAL` aliasing to preserve Polars-like output names. Be careful when touching reverse arithmetic/logical operators or alias/meta handling.

### 4) SQL core layer

#### Core abstractions (`src/belugas/_core.py`)

The current SQL core layer is centered on `sqlglot` AST composition plus a DuckDB conversion boundary.

- **`CoreHandler[T]`**
  - Generic wrapper base shared by `Expr`, `LazyFrame`, and namespace handlers.
  - Provides `.pipe(...)`, `._cls(...)`, and `.inner`.

- **`ExprHandler(CoreHandler[sqlglot.exp.Expr])`**
  - Specialization for `sqlglot` expressions.
  - Base for expression-side fluent behavior.

- **`NameSpaceHandler[T: ExprHandler]`**
  - Wraps a parent expression and returns the parent type from namespace methods.

- **`anon(name, *args)` / `anon_agg(name, *args)` / `func(name, *args)`**
  - Low-level expression builders used throughout the expression layer.

#### Conversion helpers (`src/belugas/_core.py`)

- **`into_expr(value, as_col=True) -> sqlglot.exp.Expr`**
  - Normalizes `Expr`, strings, and Python literals into `sqlglot` expressions.

- **`into_expr_list(args, as_col=False) -> list[sqlglot.exp.Expr]`**
  - Bulk conversion helper for function arguments.

- **`PQLConversionError`**
  - Raised from `src/belugas/_scans.py` when SQL generation cannot be parsed by DuckDB.

#### Relation/input wrapper (`src/belugas/_scans.py`)

`ScanSource` is the current wrapper around `duckdb.DuckDBPyRelation` plus column metadata.

- Holds `relation: duckdb.DuckDBPyRelation` and `schema`.
- Normalizes input sources through `build(...)`.
- Supports relation construction from `LazyFrame`, DuckDB relations, mappings, sequences, NumPy arrays, Narwhals/Polars frames, SQL queries, tables, and table functions.
- `from_query(...)` is the main bridge from AST query nodes to executable DuckDB relations.

### 5) Supporting modules

- `src/belugas/_funcs.py`: public module-level expression helpers (`col`, `lit`, `reduce`, `coalesce`, horizontal aggs, `unnest`, ...).
- `src/belugas/_when.py`: fluent `when(...).then(...).otherwise(...)` builder.
- `src/belugas/_window.py`: window specification helpers and rolling/window plumbing.
- `src/belugas/_meta.py`: expression metadata, markers, and planning helpers used by context methods.
- `src/belugas/namespaces.py`: handwritten namespace behavior and Polars-compat shims.
- `src/belugas/selectors.py`: selectors API.
- `src/belugas/datatypes.py`: `belugas` datatype objects and conversions.

### 6) Auto-generated code (do not edit manually)

- `src/belugas/_fns.py`: generated DuckDB function wrappers and generated namespace mixins.
- `src/belugas/meta.py`: generated `duckdb_*` module-level meta helpers.

Generated from scripts in `scripts/`.

Edit generator pipelines and regenerate instead of patching generated files by hand.

### 7) Code generation and analysis scripts

- `scripts/fn_generator/*`: generate DuckDB SQL function wrappers.
- `scripts/meta_generator/*`: generate DuckDB meta table functions.
- `scripts/comparator/*`: build `API_COVERAGE.md`.
- `scripts/_theme_generator.py`: generate SQL theme literals.
- `scripts/_check_missing_sqlglot.py`: compare DuckDB function coverage with sqlglot parser support.
- `scripts/__main__.py`: Typer CLI entrypoint.

---

## Non-negotiable implementation rules

1. Prefer `Expr`, `sqlglot`, and `ScanSource` over raw SQL strings
- Raw SQL strings are not needed AT ALL, since sqlglot can express any SQL construct we need. If you think you need raw SQL, check if sqlglot can do it first, and if it can't, either create an anonymous Expr, or patch it in the `_sqlglot_patch.py` module. Note that this should only be the case for SQL functions. Other relational nodes are all supported by sqlglot, and if you don't know how to do it, it's a documentation fetching issue from your part.

2. Do not patch generated files directly:

- Never hand-edit `src/belugas/_fns.py` or `src/belugas/meta.py`.
- Modify generator logic in `scripts/fn_generator/*` or `scripts/meta_generator/*` and regenerate.
- Note that 90% of the time, a few modifications in the `_rules.py` module are all of what is needed to fix a generator issue or add an exception. Always check the rules before considering a generator code patch.

3. Preserve DuckDB semantics:

- Do not “hack” DuckDB behavior to mimic Polars exactly when semantics differ.
- Null ordering/handling differences are acceptable if explicit and consistent.
- Note that this is the tricky part of this library. Handling tests and documenting the behavior is a human-level decision. DON'T hack your way out of this if you find yourself in a situation like this. Instead, acknowledge it, explain it in the chat, and wait for feedback.

4. Preserve expression metadata and naming behavior:

- Changes around `AliasMapper`, `Marker`, aliasing, reverse operators, and output names can easily break `select()` and `with_columns()` parity.
- Treat naming regressions as real behavior regressions.

5. Keep generated SQL/relations efficient:

- Avoid unnecessary projections/materialization.
- Keep expression composition compact.

6. Maintain fluent style:

- Prefer method chaining.
- Reuse existing helpers (`Expr.new`, `into_expr`, `into_expr_list`, `func`, `when`, `ScanSource.build`).

7. Don't hack the arguments:
- Avoid adding arguments who are not used or raise NotImplementedErrors for "API compatibility". If it don't work, then it don't exist.

8. Stay within the current abstractions:

- Build features around `Expr`, `LazyFrame`, `ScanSource`, namespace classes, and the active generator/comparator pipelines.
- When in doubt, verify the current code before introducing a new abstraction layer.

---

## Required coding style

### General Python style

- Python version target: `>=3.13`.
- Full typing is required (params, returns, key variables, generics).
- Use `match` where it improves branch clarity.
- Avoid broad/naked exceptions. `pyochain.Result` is ALWAYS preferred. Even if we want to raise immediatly, use an helper, and then unwrap it at call site.
- Don't introduce useless helpers that are used once. IF an helper is needed, but only for one call site (e.g code duplication in one method that can have a few logical branches depending on input), prefer closures rather than module-level private functions/class-level private methods. 
This often allow to reuse the arguments already in-scope, and improve "code locality" (`LazyFrame` methods are a good example of this pattern).

### Pyochain style (mandatory in this repo)

- imperative loops are forbidden. Keep iterable transformations chain-based (`map/filter/fold/filter_map/map_star/...`).
- Avoid ad-hoc Python container churn when `pc.Iter/Seq/Vec/Dict/Set` fits.
- Prefer `Option`/`Result`-oriented handling over manual `None` and ad-hoc checks. NOTE that this don't apply when we are at the public level, as we expect users to prefer passing arguments as it is rather than `Some(x)`. However, inside the implementation, for closure helpers, etc... we want to convert those ASAP to pyochain constructs.

---

## Testing protocol (critical)

Goal:

- 100% coverage target for public API behavior.

Current helpers and conventions:

- `tests/_utils.py` provides `assert_eq`, `assert_lf_eq`, and `FnsCat`.
- `assert_eq` validates expression behavior through both `select()` and `with_columns()` by default.
- Tests heavily use parametrized belugas/polars function pairs and identical call chains.

Rules for any new/updated tests:

1. Comparison-first strategy:

- Prefer comparison helpers (`assert_frame_equal`-based helpers) for behavior checks.
- Avoid naked `assert` for dataframe behavior when helper-based comparison is feasible.

1. Identical call chains:

- belugas and reference backend (Narwhals/Polars) chains must be structurally identical.
- No parameter/method-call divergence unless impossible.

1. If identical chains are impossible:

- Do not silently force a divergent implementation.
- Document why parity cannot hold (semantic/API gap), with concrete examples and options.

1. If you notice pre-existing violations while editing nearby tests:

- Fix them immediately as part of the same change scope.

1. If you change expression naming/alias behavior:

- Cover both expression-level and frame-context behavior.
- Regressions often only appear once the expression is run through `select()` or `with_columns()`.

---

## API parity workflow

Use `API_COVERAGE.md` as tracking input, not as a strict blocker.

When implementing a missing/mismatched method:

1. Check if the capability already exists in `Expr`, `LazyFrame`, a namespace class, module-level helpers, selectors, or generated mixins.
2. Validate naming and signature alignment against project intent (Polars-like + DuckDB-centric).
3. Add/adjust tests with identical belugas vs reference chains.
4. Check `scripts/comparator/_rules.py` before deciding a mismatch is a bug.
5. Regenerate coverage report if API surface changed.

---

## Generator workflow

Use `uv` commands:

- Fetch DuckDB function metadata cache:
  - `uv run -m scripts fns-to-parquet`
- Generate function wrappers:
  - `uv run -m scripts gen-fns`
- Generate DuckDB meta helpers:
  - `uv run -m scripts gen-meta`
- Generate SQL theme literal:
  - `uv run -m scripts gen-themes`
- Rebuild API coverage:
  - `uv run -m scripts compare`
- Analyze cached function metadata:
  - `uv run -m scripts analyze-funcs`
- Check sqlglot DuckDB function coverage:
  - `uv run -m scripts check-sqlglot`
- `gen-fns` depends on the parquet cache produced by `fns-to-parquet`.

After generation, run Ruff on touched files.

---

## Validation checklist before opening/merging changes

1. Did you avoid editing `_code_gen` manually?
2. Did you implement the change in the correct layer (`LazyFrame`, `Expr`, namespace, `ScanSource`, generator, comparator)?
3. Did you preserve DuckDB and `sqlglot` semantics (especially null/order behavior and expression naming)?
4. Are tests using comparison helpers and identical call chains where required?
5. If API changed, did you refresh/report coverage implications in `API_COVERAGE.md`?
6. Did you run Ruff and the relevant tests for the touched area?

---

## Inspirations and reference points

### Narwhals

Installed Narwhals implementation in `.venv` (notably `narwhals/sql.py`, `narwhals/_sql/*`, `narwhals/_duckdb/*`).
<https://narwhals-dev.github.io/narwhals/generating_sql/>
<https://narwhals-dev.github.io/narwhals/api-completeness/>

### DuckDB

DuckDB Python API and DuckDB SQL functions are the execution targets.
<https://duckdb.org/docs/stable/clients/python/overview>
<https://duckdb.org/docs/stable/sql/functions/overview>

### sqlglot

`sqlglot` is the AST layer used to model queries and expressions before conversion to DuckDB.
DuckDB dialect behavior matters when changing expression generation.
<https://github.com/tobymao/sqlglot>

### Polars API

Polars API is the ergonomics and parity reference.
Keep `belugas` decisions aligned with DuckDB semantics and the current repository architecture.
