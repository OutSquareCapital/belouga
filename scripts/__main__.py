"""script entry point.

Run with: `uv run -m scripts`
"""

from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

SELF_PATH = Path(__file__).relative_to(Path().cwd())
PQL = Path("src", "pql")
CODE_GEN = PQL.joinpath("sql", "_code_gen")

FNS_OUTPUT = CODE_GEN.joinpath("_fns.py")
REL_OUTPUT = CODE_GEN.joinpath("_core.py")
META_OUTPUT = CODE_GEN.joinpath("meta.py")

DATA_PATH = Path("scripts", "data", "functions.parquet")
STUB_PATH = Path(".venv", "Lib", "site-packages", "_duckdb-stubs", "__init__.pyi")
TYPING_PATH = PQL.joinpath("_typing.py")

InputPath = Annotated[Path, typer.Option("--input-path", "-ip")]
OutputPath = Annotated[Path, typer.Option("--output-path", "-op")]
CheckArg = Annotated[
    bool, typer.Option("--c", help="Check output without Ruff applying fixes")
]
app = typer.Typer(pretty_exceptions_show_locals=True)

console = Console()


@app.command()
def gen_fns(
    data_path: InputPath = DATA_PATH,
    output: OutputPath = FNS_OUTPUT,
    *,
    check_only: CheckArg = False,
    profile: Annotated[
        bool, typer.Option("--p", help="Enable profiling of the pipeline")
    ] = False,
) -> None:
    """Generate typed DuckDB function wrappers from the database."""
    from .fn_generator import run_pipeline

    console.print("Fetching functions from DuckDB...")
    content = run_pipeline(SELF_PATH, data_path, profile=profile)

    output.parent.mkdir(parents=True, exist_ok=True)
    res = output.write_text(content, encoding="utf-8")
    console.print(Text("Generated file at ").append(output.as_posix(), style="cyan"))
    _run_ruff(check_only=check_only, dest=output)
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def gen_themes(path: InputPath = TYPING_PATH) -> None:
    """Generate a `Literal` of all available styles for pretty-printing of the `LazyFrame.sql_query` method."""
    from ._theme_generator import generate_themes

    res = generate_themes(SELF_PATH, path)
    _run_ruff(check_only=False, dest=path)
    console.print(
        Text("Generated themes Literal in ").append(path.as_posix(), style="cyan")
    )
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def fns_to_parquet(path: InputPath = DATA_PATH) -> None:
    """Fetch function metadata from DuckDB and store as parquet at `scripts/generator/functions.parquet`."""
    from .fn_generator import get_data

    get_data(path)

    console.print(f"Fetched function metadata and stored at {path}")


@app.command()
def gen_meta(
    data_path: InputPath = DATA_PATH,
    output: OutputPath = META_OUTPUT,
    *,
    check_only: CheckArg = False,
) -> None:
    """Generate DuckDB meta table functions (duckdb_* module-level functions)."""
    from .meta_generator import run_pipeline

    console.print("Generating meta table functions...")
    content = run_pipeline(SELF_PATH, data_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    res = output.write_text(content, encoding="utf-8")
    console.print(Text("Generated file at ").append(output.as_posix(), style="cyan"))
    _run_ruff(check_only=check_only, dest=output)
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def compare() -> int:
    """Run the comparison between polars/narwhals and pql and generate markdown report at the repo root.

    Returns:
        int
    """
    from .comparator import get_comparisons

    return Path("API_COVERAGE.md").write_text(get_comparisons(), encoding="utf-8")


@app.command()
def analyze_funcs(path: InputPath = DATA_PATH) -> None:
    """Run analysis of the functions metadata and print results in console."""
    from ._func_table_analysis import analyze

    analyze(path)


@app.command()
def check_sqlglot() -> None:
    """Check for missing functions in the sqlglot `DuckDB` parser `FUNCTIONS` mapping."""
    from ._check_missing_sqlglot import check_missing_sqlglot

    res = check_missing_sqlglot(Path("MISSING_SQLGLOT.md"))
    console.print(f"Done with exit code {res}!", style="bold green")


def _run_ruff(*, check_only: bool, dest: Path) -> None:

    import subprocess

    console.print("Running Ruff checks and format...")
    uv_args = ("uv", "run", "ruff")
    run_ruff = partial(subprocess.run, check=False)

    _ = run_ruff((*uv_args, "format", str(dest)))
    _ = run_ruff((*uv_args, *_check_args(check_only=check_only), str(dest)))


def _check_args(*, check_only: bool) -> Iterable[str]:
    if check_only:
        return ("check", "--unsafe-fixes", "--diff")

    return ("check", "--fix", "--unsafe-fixes")


if __name__ == "__main__":
    app()
