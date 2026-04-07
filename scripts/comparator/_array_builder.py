from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

import pyochain as pc

from ._infos import ComparisonResult
from ._rules import Status


class BarColor(StrEnum):
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
class ArrayBuilder:
    results: pc.Vec[ComparisonResult]
    _row: pc.Vec[str] = field(default_factory=pc.Vec.new)

    def build(self) -> pc.Vec[str]:
        return self._row

    def count_cell(self) -> Self:
        return self._add(str(self._count_with()))

    def status_cell(self, status: Status) -> Self:
        return self._add(str(self._count_for_status(status)))

    def coverage_cell(self) -> Self:
        global_pct = self._coverage_percent()
        bar = _generate_progress_bar(global_pct)
        return self._add(f"{bar} ({global_pct:.1f}%)")

    def with_name(self, name: str) -> Self:
        return self._add(name)

    def _add(self, element: str) -> Self:
        self._row.append(element)
        return self

    def _coverage_percent(self) -> float:
        total = self._count_with()
        matched = self._count_for_status(Status.MATCH)
        match total:
            case 0:
                return 100.0
            case _:
                return (matched / total) * 100

    def _count_for_status(self, status: Status) -> int:
        return (
            self.results
            .iter()
            .filter(
                lambda result: (
                    result.infos
                    .status()
                    .map(lambda current: current == status)
                    .unwrap_or(default=False)
                )
            )
            .length()
        )

    def _count_with(self) -> int:
        return (
            self.results
            .iter()
            .filter(lambda result: result.infos.has_reference())
            .length()
        )


def _generate_progress_bar(percentage: float, width: int = 10) -> str:
    """Generate a Unicode progress bar with color based on percentage.

    Args:
        percentage (float): The percentage to represent.
        width (int): The width of the progress bar.

    Returns:
        str: The formatted progress bar as a string.
    """
    filled = int(percentage / 100 * width)
    empty = width - filled
    color = BarColor.on_pct(percentage)

    filled_bar = f'<span style="color: {color};">{"█" * filled}</span>'
    empty_bar = f'<span style="color: #bdc3c7;">{"░" * empty}</span>'
    return filled_bar + empty_bar
