"""script entry point.

Run with: `uv run -m scripts`
"""

from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import Annotated, final

import typer
from rich.console import Console
from rich.text import Text


@final
class _Paths:
    SELF = Path(__file__).relative_to(Path().cwd())
    PQL = Path("src", "belugas")
    FNS = PQL.joinpath("_fns.py")
    META = PQL.joinpath("meta.py")

    DATA = Path("scripts", "data", "functions.parquet")
    STUB = Path(".venv", "Lib", "site-packages", "_duckdb-stubs", "__init__.pyi")


InputPath = Annotated[Path, typer.Option("--input-path", "-ip")]  # pyright: ignore[reportUnknownMemberType]
OutputPath = Annotated[Path, typer.Option("--output-path", "-op")]  # pyright: ignore[reportUnknownMemberType]
CheckArg = Annotated[
    bool, typer.Option("--c", help="Check output without Ruff applying fixes")  # pyright: ignore[reportUnknownMemberType]
]
app = typer.Typer(pretty_exceptions_show_locals=True)

console = Console()


@app.command()
def gen_fns(
    data_path: InputPath = _Paths.DATA,
    output: OutputPath = _Paths.FNS,
    *,
    check_only: CheckArg = False,
    profile: Annotated[
        bool, typer.Option("--p", help="Enable profiling of the pipeline")  # pyright: ignore[reportUnknownMemberType]
    ] = False,
    regenerate: Annotated[
        bool,
        typer.Option(  # pyright: ignore[reportUnknownMemberType]
            "--r", help="Regenerate the source parquet file from DuckDB introspection"
        ),
    ] = False,
) -> None:
    """Generate typed DuckDB function wrappers from the database."""
    from .fn_generator import run_pipeline

    console.print("Fetching functions from DuckDB...")
    content = run_pipeline(
        _Paths.SELF, data_path, profile=profile, regenerate=regenerate
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    res = output.write_text(content, encoding="utf-8")
    console.print(Text("Generated file at ").append(output.as_posix(), style="cyan"))
    _run_ruff(check_only=check_only, dest=output)
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def gen_themes() -> None:
    """Generate a `Literal` of all available styles for pretty-printing of the `LazyFrame.sql_query` method."""
    from ._literal_generator import generate_themes

    res, path = generate_themes(_Paths.SELF)
    _run_ruff(check_only=False, dest=path)
    console.print(
        Text("Generated themes Literal in ").append(path.as_posix(), style="cyan")
    )
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def gen_nodes() -> None:
    """Generate a `Type` of all available styles for pretty-printing of the `LazyFrame.sql_query` method."""
    from ._literal_generator import generate_nodes

    res, path = generate_nodes(_Paths.SELF)
    _run_ruff(check_only=False, dest=path)
    console.print(
        Text("Generated nodes Type in ").append(path.as_posix(), style="cyan")
    )
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def gen_meta(
    data_path: InputPath = _Paths.DATA,
    output: OutputPath = _Paths.META,
    *,
    check_only: CheckArg = False,
) -> None:
    """Generate DuckDB meta table functions (duckdb_* module-level functions)."""
    from .meta_generator import run_pipeline

    console.print("Generating meta table functions...")
    content = run_pipeline(_Paths.SELF, data_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    res = output.write_text(content, encoding="utf-8")
    console.print(Text("Generated file at ").append(output.as_posix(), style="cyan"))
    _run_ruff(check_only=check_only, dest=output)
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def compare() -> int:
    """Run the comparison between polars/narwhals and belugas and generate markdown report at the repo root.

    Returns:
        int
    """
    from .comparator import get_comparisons

    return Path("API_COVERAGE.md").write_text(get_comparisons(), encoding="utf-8")


@app.command()
def analyze_funcs(path: InputPath = _Paths.DATA) -> None:
    """Run analysis of the functions metadata and print results in console."""
    from ._func_table_analysis import analyze

    analyze(path)


@app.command()
def check_sqlglot() -> None:
    """Check for missing functions in the sqlglot `DuckDB` parser `FUNCTIONS` mapping."""
    from ._check_missing_sqlglot import check_missing_sqlglot

    res = check_missing_sqlglot(Path("MISSING_SQLGLOT.md"))
    console.print(f"Done with exit code {res}!", style="bold green")


@app.command()
def bench(
    runs: Annotated[
        int, typer.Option("--runs", "-r", help="Number of runs per benchmark")  # pyright: ignore[reportUnknownMemberType]
    ] = 250,
    names: Annotated[
        list[str] | None,
        typer.Option(  # pyright: ignore[reportUnknownMemberType]
            "--names",
            "-n",
            help="Names of benchmarks to run (default: all)",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Run the benchmarks comparing belugas and polars implementations."""
    from pyochain import option

    from ._benchmark import run_benchmark

    run_benchmark(runs, option(names))


@app.command()
def bench_dtypes(
    runs: Annotated[
        int, typer.Option("--runs", "-r", help="Number of runs per benchmark")  # pyright: ignore[reportUnknownMemberType]
    ] = 250,
) -> None:
    """Run the benchmarks comparing belugas and polars implementations."""
    from ._benchmark_dtypes import main

    main(runs)


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
