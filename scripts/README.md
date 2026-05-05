# Scripts

This directory contains the development-time tooling used to maintain `pql`.
These scripts are not part of the public API and are intended for contributors working on code generation, metadata extraction, and API coverage tracking.

The main entrypoint is:

```shell
uv run -m scripts --help
```

## What This Folder Is For

The `scripts` package is used for four main tasks:

1. Generate typed wrappers around DuckDB functions.
2. Generate typed wrappers around DuckDB core relation and expression APIs.
3. Generate the SQL theme `Literal` used by pretty-printing.
4. Compare `pql` against Polars and Narwhals to produce the API coverage report.

Generated files live in `src/pql/sql/_code_gen/` and must not be edited manually. If generation output is wrong, fix the generator logic in this directory and regenerate.

## CLI Commands

### `gen-fns`

Generate typed wrappers for DuckDB SQL functions.

- Source data: `scripts/fn_generator/functions.parquet`
- Default output: `src/pql/_fns.py`
- Implementation: `scripts/fn_generator/`

Examples:

```shell
uv run -m scripts fns-to-parquet
uv run -m scripts gen-fns
```

Useful options:

- `--input-path` / `-ip`: override the parquet metadata source.
- `--output-path` / `-op`: override the generated file destination.
- `--c`: run Ruff in check mode instead of applying fixes.
- `--p`: enable profiling of the generation pipeline.

### `gen-core`

Generate typed wrappers around DuckDB core Python objects from the installed DuckDB stubs.

- Default stub input: `.venv/Lib/site-packages/_duckdb-stubs/__init__.pyi`
- Default output: `src/pql/sql/_code_gen/_core.py`
- Implementation: `scripts/core_generator/`

Example:

```shell
uv run -m scripts gen-core
```

Useful options:

- `--input-path` / `-ip`: override the stub path.
- `--output-path` / `-op`: override the generated file destination.
- `--c`: run Ruff in check mode instead of applying fixes.

### `gen-themes`

Generate the `Themes = Literal[...]` declaration used by SQL pretty-printing.

- Default output: `src/pql/_typing.py`
- Implementation: `scripts/_theme_generator.py`

Example:

```shell
uv run -m scripts gen-themes
```

### `fns-to-parquet`

Fetch DuckDB function metadata and store it as parquet for later reuse by `gen-fns` and `analyze-funcs`.

- Default output: `scripts/fn_generator/functions.parquet`
- Data source: `duckdb_functions()`

Example:

```shell
uv run -m scripts fns-to-parquet
```

### `compare`

Build the API parity report between `pql`, Polars, and Narwhals.

- Default output: `API_COVERAGE.md`
- Implementation: `scripts/comparator/`

Example:

```shell
uv run -m scripts compare
```

### `analyze-funcs`

Inspect the cached DuckDB function metadata and print summary tables in the console.

- Default input: `scripts/fn_generator/functions.parquet`
- Implementation: `scripts/_func_table_analysis.py`

Example:

```shell
uv run -m scripts analyze-funcs
```

## Recommended Workflows

### Refresh function wrappers

Use this after DuckDB changes, metadata rule changes, or generator updates in `scripts/fn_generator/`.

```shell
uv run -m scripts fns-to-parquet
uv run -m scripts gen-fns
```

### Refresh core wrappers

Use this after updating DuckDB or changing the parsing and generation logic in `scripts/core_generator/`.

```shell
uv run -m scripts gen-core
```

### Refresh API coverage

Use this after changing the public API or comparison rules.

```shell
uv run -m scripts compare
```

### Refresh theme literals

Use this if the SQL rendering theme support changes.

```shell
uv run -m scripts gen-themes
```

## Directory Layout

### `comparator/`

Builds the API coverage report.

- `_main.py`: assembles the comparison report.
- `_parse.py`, `_infos.py`: inspect compared APIs.
- `_rules.py`: ignore rules and comparison-specific logic.
- `_array_builder.py`, `_text.py`: format markdown output.

### `core_generator/`

Generates typed wrappers from DuckDB stub files.

- `_parse.py`: extracts methods from stubs.
- `_rules.py`: generation rules and special cases.
- `_sections.py`: file/class rendering helpers.
- `_structs.py`: generator data structures.
- `_target.py`: generation targets and wrapper metadata.
- `_main.py`: high-level generation entrypoint.

### `fn_generator/`

Generates typed wrappers for DuckDB SQL functions.

- `_query.py`: DuckDB metadata query and preprocessing.
- `_schemas.py`: schema declarations for the metadata dataset.
- `_rules.py`: naming and generation rules.
- `_sections.py`, `_format.py`, `_str_builder.py`: code rendering.
- `_dtypes.py`: dtype handling.
- `_main.py`: high-level pipeline entrypoint.

### Top-level helpers

- `_func_table_analysis.py`: console analysis of the function metadata parquet.
- `_theme_generator.py`: updates the `Themes` literal in `src/pql/_typing.py`.
- `_utils.py`: shared enums and helpers used by multiple generators.
- `__main__.py`: Typer CLI entrypoint.

## Notes

- Most generation commands automatically run Ruff on the generated file.
- `gen-fns` depends on the parquet cache created by `fns-to-parquet`.
- `compare` writes directly to the repository root.
- If a generated file looks wrong, change the generator code here instead of patching the generated file.
- Git is your friend. If a script code change generate broken code, pql could potentially crash, and since scripts depend on pql, revert the generated code before trying again.
