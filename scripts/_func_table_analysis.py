"""Analyse approfondie de la classification des fonctions DuckDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pyochain as pc
from rich.console import Console
from rich.table import Table

from .fn_generator._query import _filters  # pyright: ignore[reportPrivateUsage]
from .fn_generator._schemas import DuckCols

CONSOLE = Console()


def _df_to_iter(df: pl.DataFrame) -> pc.Iter[tuple[Any, ...]]:  # pyright: ignore[reportExplicitAny]
    return pc.Iter(df.iter_rows())


def _analyze_by_categories(lf: pl.LazyFrame) -> None:
    """Analyse par categories (exploded)."""
    CONSOLE.print(
        "\n═══ Analyse par categories (1ère catégorie) ═══\n", style="bold cyan"
    )

    df = (
        lf
        .explode("categories")
        .select(pl.col("categories").value_counts(parallel=True).struct.unnest())
        .sort("count", descending=True)
        .collect()
    )

    table = Table(title="Distribution par catégorie")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("%", justify="right", style="green")

    total = df.get_column("count").sum()
    df.head(20).pipe(_df_to_iter).for_each_star(
        lambda cat, count: table.add_row(  # pyright: ignore[reportAny]
            cat or "NULL",
            str(count),  # pyright: ignore[reportAny]
            f"{count / total * 100:.1f}%",
        )
    )

    CONSOLE.print(table)
    CONSOLE.print(f"\nTotal: {total} fonctions")


def _analyze_multi_category(lf: pl.LazyFrame) -> None:
    """Analyse les fonctions avec plusieurs catégories."""
    CONSOLE.print("\n═══ Fonctions multi-catégories ═══\n", style="bold cyan")

    df = (
        lf
        .with_columns(pl.col("categories").list.len().alias("n_categories"))
        .filter(pl.col("n_categories").gt(1))
        .select("function_name", "categories", "n_categories")
        .sort("n_categories", descending=True)
        .collect()
    )

    if df.height == 0:
        CONSOLE.print("Aucune fonction avec plusieurs catégories", style="yellow")
        return

    table = Table(title="Fonctions avec plusieurs catégories")
    table.add_column("Function", style="cyan")
    table.add_column("Categories", style="magenta")
    table.add_column("Count", justify="right", style="green")

    df.pipe(_df_to_iter).for_each_star(
        lambda name, cats, n: table.add_row(name, ", ".join(cats), str(n))  # pyright: ignore[reportAny]
    )

    CONSOLE.print(table)
    CONSOLE.print(f"\nTotal: {df.height} fonctions multi-catégories")


def analyze(path: Path) -> None:
    lf = pl.scan_parquet(path).pipe(_filters, DuckCols())
    _analyze_by_categories(lf)
    _analyze_multi_category(lf)
