from dataclasses import dataclass, field
from types import ModuleType

from pyochain import Iter, Seq, Set, Vec

from .._utils import Dunders, Pql, get_attr
from ._array_builder import ArrayBuilder
from ._infos import ComparisonResult
from ._rules import Status


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
            _format(self.results, "[+] Extra Methods (pql-only)", status=Status.EXTRA),
        ))

    def to_row(self) -> Vec[str]:
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


def header() -> Iter[str]:
    txt = """
# pql vs Polars API Comparison Report.

This report shows the API coverage of pql compared to other libraries.

## Summary

Each summary cell is relative to Polars.
"""
    return Iter.once(txt)


@dataclass(slots=True)
class ClassComparison:
    """Converter between entry arguments and ComparisonReport."""

    polars_cls: object
    pql_cls: object
    name: Pql
    ignored_names: Set[str] = field(default_factory=Set[str].new)

    def to_report(self) -> ComparisonReport:
        """Compare two classes and return comparison results.

        Returns:
            ComparisonReport: A report containing the results of the comparison.
        """
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

    def _get_public_methods(self, cls: object) -> Set[str]:
        def _module_public_names() -> Set[str]:
            match cls:
                case ModuleType() as mod:
                    return Set(mod.__all__)  # pyright: ignore[reportAny]
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
    data_rows = _summary_rows(comps)

    return (
        Iter
        .once(_summary_header())
        .chain(data_rows)
        .collect()
        .into(_summary_widths)
        .collect()
        .into(
            lambda widths: Iter.once(_format_row(_summary_header(), widths)).chain(
                Iter.once(_format_separator(widths)),
                data_rows.iter().map(lambda row: _format_row(row, widths)),
            )
        )
    )


def _summary_rows(comps: Seq[ComparisonReport]) -> Seq[Vec[str]]:
    return comps.iter().map(lambda comp: comp.to_row()).collect()


def _summary_widths(rows: Seq[Seq[str]]) -> Iter[int]:
    return Iter(range(_summary_header().length())).map(
        lambda idx: (
            rows
            .iter()
            .map(lambda row: len(row[idx]))
            .fold(0, lambda acc, length: max(length, acc))
        )
    )


def _summary_header() -> Seq[str]:
    return Seq((
        "Class",
        "Coverage",
        "Implemented",
        "Matched",
        "Missing",
        "Mismatched",
        "Extra",
    ))


def _format_separator(widths: Seq[int]) -> str:
    cells = widths.iter().map(lambda width: "-" * width).join(" | ")
    return f"| {cells} |"


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


def _format_row(row: Seq[str], widths: Seq[int]) -> str:
    cells = (
        Iter(range(widths.length()))
        .map(lambda idx: row[idx].ljust(widths[idx]))
        .join(" | ")
    )
    return f"| {cells} |"


def _by_status(results: Vec[ComparisonResult], status: Status) -> Seq[ComparisonResult]:
    return results.iter().filter(lambda r: r.classification == status).collect()
