from dataclasses import dataclass, field
from types import ModuleType

import pyochain as pc

from .._utils import Dunders, Pql, get_attr
from ._array_builder import ArrayBuilder
from ._infos import ComparisonResult
from ._rules import Status


@dataclass(slots=True)
class ComparisonReport:
    """Report for a class comparison."""

    name: str
    results: pc.Vec[ComparisonResult]

    def to_section(self) -> pc.Seq[str]:
        """Format detailed sections for a class comparison."""
        return pc.Seq((
            f"\n## {self.name}\n",
            _format(self.results, "[x] Missing Methods", status=Status.MISSING),
            _format(
                self.results,
                "[!] Signature Mismatches",
                status=Status.SIGNATURE_MISMATCH,
            ),
            _format(self.results, "[+] Extra Methods (pql-only)", status=Status.EXTRA),
        ))

    def to_row(self) -> pc.Vec[str]:
        """Return a row of summary data as columns."""
        return (
            ArrayBuilder(self.results)
            .with_name(self.name)
            .coverage_cell()
            .count_cell()
            .status_cell(Status.MATCH)
            .status_cell(Status.MISSING)
            .status_cell(Status.SIGNATURE_MISMATCH)
            .status_cell(Status.EXTRA)
            .build()
        )


def header() -> pc.Iter[str]:
    txt = """
# pql vs Polars API Comparison Report.

This report shows the API coverage of pql compared to other libraries.

## Summary

Each summary cell is relative to Polars.
"""
    return pc.Iter.once(txt)


def _summary_header() -> pc.Seq[str]:
    return pc.Seq((
        "Class",
        "Coverage",
        "Implemented",
        "Matched",
        "Missing",
        "Mismatched",
        "Extra",
    ))


@dataclass(slots=True)
class ClassComparison:
    """Converter between entry arguments and ComparisonReport."""

    polars_cls: object
    pql_cls: object
    name: Pql
    ignored_names: pc.Set[str] = field(default_factory=pc.Set[str].new)

    def to_report(self) -> ComparisonReport:
        """Compare two classes and return comparison results."""
        polars_methods = self._get_public_methods(self.polars_cls)
        pql_methods = self._get_public_methods(self.pql_cls)

        return ComparisonReport(
            self.name,
            polars_methods
            .union(pql_methods)
            .iter()
            .map(
                lambda name: ComparisonResult(
                    self.polars_cls, self.pql_cls, name, self.name
                )
            )
            .sort(key=lambda r: r.method_name),
        )

    def _get_public_methods(self, cls: object) -> pc.Set[str]:
        def _module_public_names() -> pc.Set[str]:
            match cls:
                case ModuleType() as mod:
                    return pc.Set(mod.__all__)  # pyright: ignore[reportAny]
                case _:
                    return pc.Set(dir(cls))

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

        return _module_public_names().iter().filter(_predicate).collect(pc.Set)


def render_summary_table(comps: pc.Seq[ComparisonReport]) -> pc.Iter[str]:
    data_rows = _summary_rows(comps)

    return (
        pc.Iter
        .once(_summary_header())
        .chain(data_rows)
        .collect()
        .into(_summary_widths)
        .collect()
        .into(
            lambda widths: pc.Iter.once(_format_row(_summary_header(), widths)).chain(
                pc.Iter.once(_format_separator(widths)),
                data_rows.iter().map(lambda row: _format_row(row, widths)),
            )
        )
    )


def _summary_rows(comps: pc.Seq[ComparisonReport]) -> pc.Seq[pc.Vec[str]]:
    return comps.iter().map(lambda comp: comp.to_row()).collect()


def _summary_widths(rows: pc.Seq[pc.Seq[str]]) -> pc.Iter[int]:
    return pc.Iter(range(_summary_header().length())).map(
        lambda idx: (
            rows
            .iter()
            .map(lambda row: len(row[idx]))
            .fold(0, lambda acc, length: max(length, acc))
        )
    )


def _format_separator(widths: pc.Seq[int]) -> str:
    cells = widths.iter().map(lambda width: "-" * width).join(" | ")
    return f"| {cells} |"


def _format(results: pc.Vec[ComparisonResult], title: str, *, status: Status) -> str:
    """Format a section of the report."""
    return (
        results
        .into(_by_status, status)
        .then(
            lambda items: (
                pc
                .Iter((f"\n### {title} ({items.length()})\n",))
                .chain(items.iter().flat_map(lambda r: r.to_format(status=status)))
                .join("\n")
            )
        )
        .unwrap_or("")
    )


def _format_row(row: pc.Seq[str], widths: pc.Seq[int]) -> str:
    cells = (
        pc
        .Iter(range(widths.length()))
        .map(lambda idx: row[idx].ljust(widths[idx]))
        .join(" | ")
    )
    return f"| {cells} |"


def _by_status(
    results: pc.Vec[ComparisonResult], status: Status
) -> pc.Seq[ComparisonResult]:
    return results.iter().filter(lambda r: r.classification == status).collect()
