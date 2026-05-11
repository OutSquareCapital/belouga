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
- Public scan constructors in `src/belugas/_scans.py` and at package root (`from_arrow`, `from_dict`, `from_dicts`, `from_numpy`, `from_pandas`, `from_polars`, `from_query`, `from_records`, `from_table`, `from_table_function`, `scan_csv`, `scan_json`, `scan_parquet`).
- Module-level expression helpers in `src/belugas/_funcs.py` (`col`, `lit`, `when`, `coalesce`, scalar aggs, horizontal aggs, `element`, `count`, `len`, `all`, `any`, ...).
- `selectors`, `meta`, and datatype objects re-exported from the package root.
- Query inspection helpers exposed through `LazyFrame.sql_query()` and `LazyFrame.explain()`.
- Grouped operations through `LazyGroupBy` in `src/belugas/_groupby.py`.
- Relation handling organized around `LazyFrame`, `ScanSource`, and the logical plan in `src/belugas/_plan/`.

---

## Architecture (must understand before changing code)

### 1) Public API layer

The user-facing API is centered on `LazyFrame` in `src/belugas/_frame.py` and `Expr` in `src/belugas/_expr.py`, re-exported from `src/belugas/__init__.py`. Constructors and scan entrypoints live in `src/belugas/_scans.py`, and the overall surface should stay Polars-like and ergonomic.

### 2) Frame/query layer

#### `LazyFrame` (`src/belugas/_frame.py`)

`LazyFrame` is the main query builder. It inherits from `CoreHandler[ScanSource]`, keeps the input relation in `_inner`, and accumulates deferred operations in `_nodes`. It does not keep a final SQL query around. Compilation happens on demand in `_compile()`, and terminal methods such as `collect`, `lazy`, `fetch_all`, `sql_query`, `explain`, and the sink methods all execute from that compiled plan. In practice, relation behavior in `belugas` is driven by `LazyFrame`, `ScanSource`, and the plan compiler together.

### 3) Plan layer

#### `src/belugas/_plan/`

The plan layer is the center of relation compilation. `nodes.py` defines the logical operations, `_resolve.py` turns a source plus node sequence into a `CompiledPlan`, and the other modules in `src/belugas/_plan/` compile or optimize specific families of operations. Most frame changes therefore belong either in the public method that appends the operation or in the plan layer that resolves and emits it.

### 4) Expression layer

#### `Expr` (`src/belugas/_expr.py`)

`Expr` is the public expression object. It extends the generated `Fns` mixin, wraps a `sqlglot` expression through the core handler stack, and carries alias metadata so names survive context changes correctly. `Expr.new(...)` is the normal coercion entrypoint, and most expression feature work happens here or in the namespace layer. Reverse literal operators are a sharp edge because they rely on `Marker.LITERAL` aliasing to preserve Polars-like output names.

### 5) SQL core layer

#### Core abstractions (`src/belugas/_core.py`)

The SQL core layer is the shared `sqlglot` boundary. `CoreHandler`, `ExprHandler`, and `NameSpaceHandler` provide the common wrapper behavior used across expressions, frames, and namespaces, while helpers such as `anon`, `anon_agg`, `func`, `into_expr`, and `into_expr_list` normalize Python inputs into `sqlglot` expressions.

#### Conversion helpers (`src/belugas/_core.py`)

`PQLConversionError` is raised from `src/belugas/_scans.py` when generated SQL cannot be parsed by DuckDB.

#### Relation/input wrapper (`src/belugas/_scans.py`)

`ScanSource` wraps a `duckdb.DuckDBPyRelation` together with schema metadata. It is the execution boundary for relation inputs and the bridge from compiled `sqlglot` ASTs back to executable DuckDB relations.

### 6) Query inspection and supporting modules

#### `src/belugas/_parser.py`

`ParsedQuery` is the public query-inspection object returned by `LazyFrame.sql_query()`. It handles SQL rendering, tokenization, and pretty display, so SQL-inspection changes should usually be made here.

### 7) Supporting modules

The main handwritten support code is split between module-level expression helpers in `src/belugas/_funcs.py`, conditional and window helpers in `src/belugas/_when.py` and `src/belugas/_window.py`, DuckDB-specific `sqlglot` patching in `src/belugas/_sqlglot_patch.py`, and the public namespace, selector, and datatype layers. Expression metadata and alias-planning helpers such as `Resolver`, `AliasMapper`, and `MultiAliasMapper` live directly in `src/belugas/_expr.py`.

### 8) Auto-generated code (do not edit manually)

Generated outputs include `src/belugas/_fns.py`, `src/belugas/meta.py`, `src/belugas/_plan/nodes.py`, and generated parts of `src/belugas/typing.py`. Update the generator pipeline in `scripts/` and regenerate instead of patching those files by hand.

### 9) Code generation and analysis scripts

Dev tooling lives under `scripts/`. It covers function and meta generation, plan-node and theme generation, coverage comparison, missing-sqlglot analysis, and benchmarks, with `scripts/__main__.py` as the CLI entrypoint.

---

## Non-negotiable implementation rules

1. Prefer `Expr`, `sqlglot`, `ScanSource`, and plan nodes over raw SQL strings
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

- Build features around `Expr`, `LazyFrame`, `ScanSource`, `src/belugas/_plan/`, namespace classes, and the active generator/comparator pipelines.
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
- Prefer narrow targeted test runs while iterating, then broaden only once the touched area is stable.

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

- Generate function wrappers:
  - `uv run -m scripts gen-fns`
- Regenerate function metadata and wrappers from DuckDB introspection:
  - `uv run -m scripts gen-fns --r`
- Generate DuckDB meta helpers:
  - `uv run -m scripts gen-meta`
- Generate SQL theme literal:
  - `uv run -m scripts gen-themes`
- Generate logical plan node declarations:
  - `uv run -m scripts gen-nodes`
- Rebuild API coverage:
  - `uv run -m scripts compare`
- Analyze cached function metadata:
  - `uv run -m scripts analyze-funcs`
- Check sqlglot DuckDB function coverage:
  - `uv run -m scripts check-sqlglot`
- Run benchmarks:
  - `uv run -m scripts bench`

After generation or handwritten code changes, run Ruff and type-checking on the touched scope.

---

## Validation checklist before opening/merging changes

1. Did you avoid editing generated outputs manually?
2. Did you implement the change in the correct layer (`LazyFrame`, `Expr`, namespace, `ScanSource`, generator, comparator)?
3. If the change affects relation behavior, did you check whether it belongs in a plan node push, a plan optimizer rule, or a plan compiler handler?
4. Did you preserve DuckDB and `sqlglot` semantics (especially null/order behavior and expression naming)?
5. Are tests using comparison helpers and identical call chains where required?
6. If API changed, did you refresh/report coverage implications in `API_COVERAGE.md`?
7. Did you run Ruff and the relevant tests for the touched area?

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
