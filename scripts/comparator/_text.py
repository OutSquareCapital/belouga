from dataclasses import dataclass, field
from enum import StrEnum
from types import ModuleType

import polars as pl
from pyochain import Iter, Seq, Set, Vec

from .._utils import Dunders, Pql, get_attr
from ._infos import ComparisonResult
from ._rules import Status


class _BarColor(StrEnum):
    RED = "#e74c3c"
    ORANGE = "#f39c12"
    GREEN = "#27ae60"

    @classmethod
    def on_pct(cls, percentage: float) -> str:
        match percentage:
            case p if p < 30:
                return cls.RED
            case p if p < 60:
                return cls.ORANGE
            case _:
                return cls.GREEN


@dataclass(slots=True)
class ComparisonReport:
    """Report for a class comparison."""

    name: str
    results: Vec[ComparisonResult]

    def to_section(self) -> Seq[str]:
        """Format detailed sections for a class comparison.

        Returns:
            Seq[str]: A sequence of formatted strings representing the detailed sections of the report.
        """
        return Seq((
            f"\n## {self.name}\n",
            _format(self.results, "[x] Missing Methods", status=Status.MISSING),
            _format(
                self.results,
                "[!] Signature Mismatches",
                status=Status.SIGNATURE_MISMATCH,
            ),
            _format(
                self.results, "[+] Extra Methods (belugas-only)", status=Status.EXTRA
            ),
        ))


def header() -> Iter[str]:
    txt = """
# belugas vs Polars API Comparison Report.

This report shows the API coverage of belugas compared to polars.

## Summary

Each summary cell is relative to Polars.
"""
    return Iter.once(txt)


@dataclass(slots=True)
class ClassComparison:
    """Converter between entry arguments and ComparisonReport."""

    polars_cls: object
    belugas_cls: object
    name: Pql
    ignored_names: Set[str] = field(default_factory=Set[str].new)

    def to_report(self) -> ComparisonReport:
        """Compare two classes and return comparison results.

        Returns:
            ComparisonReport: A report containing the results of the comparison.
        """
        polars_methods = self._get_public_methods(self.polars_cls)
        belugas_methods = self._get_public_methods(self.belugas_cls)

        return ComparisonReport(
            self.name,
            polars_methods
            .union(belugas_methods)
            .iter()
            .map(
                lambda name: ComparisonResult(
                    self.polars_cls, self.belugas_cls, name, self.name
                )
            )
            .sort(key=lambda r: r.method_name),
        )

    def _get_public_methods(self, cls: object) -> Set[str]:
        def _module_public_names() -> Set[str]:
            match cls:
                case ModuleType() as mod:
                    return Set(mod.__all__)
                case _:
                    return Set(dir(cls))

        def _predicate(name: str) -> bool:
            return (
                not name.startswith("_")
                and not self.ignored_names.contains(name)
                and not (
                    get_attr(cls, name)
                    .and_then(lambda attr: get_attr(attr, Dunders.DEPRECATED))
                    .map(bool)
                    .unwrap_or(default=False)
                )
                and (
                    get_attr(cls, name)
                    .map(lambda attr: callable(attr) or isinstance(attr, property))
                    .unwrap_or(default=False)
                )
            )

        return _module_public_names().iter().filter(_predicate).collect(Set)


def render_summary_table(comps: Seq[ComparisonReport]) -> Iter[str]:
    from .._utils import set_pl_config

    set_pl_config()
    class_name = pl.col("class_name")
    has_reference = pl.col("has_reference")
    has_belugas = pl.col("has_belugas")
    classification = pl.col("classification")
    polars_total = pl.col("Polars Total")
    matched = pl.col("Matched")

    def _coverage_cell(pct: float) -> str:
        filled = int(pct / 100 * 10)
        color = _BarColor.on_pct(pct)
        filled_bar = f'<span style="color: {color};">{"█" * filled}</span>'
        empty_bar = f'<span style="color: #bdc3c7;">{"░" * (10 - filled)}</span>'
        return f"{filled_bar}{empty_bar} ({pct:.1f}%)"

    return (
        comps
        .iter()
        .flat_map(
            lambda comp: comp.results.iter().map(
                lambda r: (
                    comp.name,
                    r.infos.has_reference(),
                    r.infos.belugas_info.is_some(),
                    str(r.classification),
                )
            )
        )
        .collect()
        .into(
            lambda rows: pl.LazyFrame(
                rows,
                schema={
                    "class_name": pl.String,
                    "has_reference": pl.Boolean,
                    "has_belugas": pl.Boolean,
                    "classification": pl.String,
                },
                orient="row",
            )
        )
        .group_by(class_name, maintain_order=True)
        .agg(
            has_reference.sum().alias("Polars Total"),
            has_belugas.sum().alias("Belugas Total"),
            has_reference.and_(has_belugas).sum().alias("Compared"),
            classification.eq(Status.MATCH).sum().alias("Matched"),
            classification.eq(Status.MISSING).sum().alias("Missing"),
            classification.eq(Status.SIGNATURE_MISMATCH).sum().alias("Mismatched"),
            classification.eq(Status.EXTRA).sum().alias("Belugas Only"),
        )
        .select(
            class_name.alias("Class"),
            pl
            .when(polars_total.gt(0))
            .then(matched.cast(pl.Float64).truediv(polars_total).mul(100.0))
            .otherwise(pl.lit(100.0))
            .map_elements(_coverage_cell, return_dtype=pl.String)
            .alias("Coverage"),
            "Belugas Total",
            "Compared",
            "Matched",
            "Missing",
            "Mismatched",
            "Belugas Only",
        )
        .collect()
        .pipe(lambda df: Iter.once(repr(df)))
    )


def _format(results: Vec[ComparisonResult], title: str, *, status: Status) -> str:
    """Format a section of the report.

    Args:
        results (Vec[ComparisonResult]): The comparison results to format.
        title (str): The title of the section.
        status (Status): The status to filter the results by.

    Returns:
        str: The formatted section of the report.
    """
    return (
        results
        .into(_by_status, status)
        .then(
            lambda items: (
                Iter((f"\n### {title} ({items.length()})\n",))
                .chain(items.iter().flat_map(lambda r: r.to_format(status=status)))
                .join("\n")
            )
        )
        .unwrap_or("")
    )


def _by_status(results: Vec[ComparisonResult], status: Status) -> Seq[ComparisonResult]:
    return results.iter().filter(lambda r: r.classification == status).collect()
