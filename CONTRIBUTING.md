
# Contributing to `belugas`

Thank you for your interest in contributing to `belugas`!

Contributions are always welcome, whether it's a bug fix, a new feature, or just improving the documentation.

## Initial setup

1. Fork the repository and clone it locally.
2. Install the development dependencies using `uv sync --dev`.

## Testing

The project heavily compares `belugas` behavior against reference Polars chains where parity is expected.

To run the tests, use:

```shell
uv run pytest tests/
uv run pytest tests/ --cov=src/ --cov-report=term-missing # If we want to check the uncovered lines
```

## Type checking and formatting

Before submitting a PR, ensure that the code is properly formatted and type-checked.

To do so, run at the repo root:

```shell
uv run ruff check . --fix --unsafe-fixes; uv run ruff format .; uv run basedpyright .
```

Note that the repo rules are VERY pedantic. We use all rules for basedpyright and Ruff (including experimental ones), only desactivating a few ones.

Sometimes ignoring rules can be necessary for valid reasons.

In that case, motivate the decision in a review comment, and prefer `basedpyright: ignore <reason>` over `type: ignore <reason>`, as the former can be checked for staleness.

## Code style

The code style can be surprizing for pure python users, as ALL iterations are handled with [`pyochain`](https://outsquarecapital.github.io/pyochain/).

If you are not familiar with it, take some time to get used to the style and patterns, by consulting the pyochain documentation linked above, or read this excellent series of articles from [jetbrains](https://blog.jetbrains.com/rust/2024/03/12/rust-iterators-beyond-the-basics-part-i-building-blocks/).

The same applies for nullable values and error handling, which are also handled with pyochain patterns, with the [`Option`](https://doc.rust-lang.org/rust-by-example/std/option.html) and [`Result`](https://doc.rust-lang.org/rust-by-example/std/result.html) constructs.

There's a friction at this level tough: `belugas` is a Python library, and we don't expect, nor want to, end users to input Option and Results values when calling the API.

So we expect in entry points values such as `int | None`, but we handle them internally as `Option[int]`, especially if we then pass this value to internal helpers that aren't part of the public API.
Note that those must ALWAYS use `Option[T]` instead of `T | None`.

As for errors, we always unwrap them before returning to the end user.

However we carry the `Result` type internally for as long as necessary, because it allows us to handle errors in a more robust way, without losing context, and without having to litter the code with try/except blocks, with implicit failure paths.

## Architecture

`belugas` exposes a Polars-like lazy API on top of DuckDB, with `sqlglot` used as the SQL AST layer.

The runtime code lives under `src/belugas`, dev tooling under `scripts`, feature-oriented tests under `tests`, and the root files mainly track packaging, roadmap, coverage, and SQL-glot gaps.

The `docs/` folder currently only contains assets used by the README.

### Public surface

The public API is centered on `LazyFrame` in `src/belugas/_frame.py` and `Expr` in `src/belugas/_expr.py`, both re-exported from `src/belugas/__init__.py`.

Module-level helpers such as `col`, `lit`, `when`, aggregations, and horizontal aggregations live in `src/belugas/_funcs.py`, while data-loading constructors and scans live in `src/belugas/_scans.py`.

### Core query pipeline

`src/belugas/_core.py` contains the shared wrappers and coercion helpers used everywhere: `CoreHandler`, `ExprHandler`, `NameSpaceHandler`, `into_expr`, `into_expr_list`, `anon`, `anon_agg`, and `func`.

This is the boundary where Python values are normalized into `sqlglot` expressions and where the fluent API keeps a consistent internal shape.

`LazyFrame` is the relational builder.

It wraps a `sqlglot` selectable, tracks sources and schema, and implements the main query operations such as `select`, `with_columns`, `filter`, `group_by`, `join`, `pivot`, `sort`, `collect`, and `lazy`.

Grouped operations are split into `src/belugas/_groupby.py`, and join normalization lives in `src/belugas/_joins.py`.

### Expression system

`Expr` wraps a `sqlglot` expression and extends the generated mixins from `src/belugas/_fns.py`.

Expression metadata lives in `src/belugas/_meta.py`; it is responsible for naming, aliasing, markers, and frame-context behavior, so regressions there usually surface through `select` and `with_columns`.

Namespaces such as `.str`, `.list`, `.struct`, `.dt`, `.arr`, `.json`, `.re`, `.map`, `.enum`, `.geo`, and `.name` are implemented in `src/belugas/namespaces.py`.

### Execution boundary and supporting modules

`src/belugas/_scans.py` contains `ScanSource`, the bridge between query ASTs and executable DuckDB relations.

It normalizes supported inputs such as DuckDB relations, Python mappings and sequences, NumPy arrays, pandas and Polars objects, SQL queries, tables, and table functions, then materializes queries through `ScanSource.from_query(...)`.

The rest of the handwritten support code is organized by concern:

- `src/belugas/_when.py` for conditional builders
- `src/belugas/_window.py` for window logic
- `src/belugas/_parser.py` for SQL parsing and query inspection
- `src/belugas/_sqlglot_patch.py` for DuckDB-specific `sqlglot` extensions
- `src/belugas/selectors.py` for selectors
- `src/belugas/datatypes.py` for public datatypes
- `src/belugas/utils.py` plus `src/belugas/typing.py` for internal support types and generated SQL-display assets.

### Generated files

`src/belugas/_fns.py` and `src/belugas/meta.py` (and a few lines from `src/belugas/typing.py`) are generated outputs.

If one of them needs to change, update the generator logic in `scripts/` and regenerate instead of editing the generated file by hand.

## Scripts

Scripts are dev-time tooling, not part of the public API.

- `scripts/fn_generator/` generates the DuckDB function wrappers
- `scripts/meta_generator/` generates the DuckDB meta helpers
- `scripts/comparator/` produces `API_COVERAGE.md`.

The remaining top-level script modules support metadata extraction, sqlglot coverage checks, and SQL theme generation.

More infos with the following command:

```shell
uv run -m scripts --help
```

The main generated outputs are:

- [DuckDB function wrappers and namespace mixins](src/belugas/_fns.py)
- [DuckDB meta table-function helpers](src/belugas/meta.py)
- [The SQL display theme literal](src/belugas/typing.py)
