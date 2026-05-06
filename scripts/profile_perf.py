"""Profile pql planning hotspots with reproducible synthetic scenarios."""

from __future__ import annotations

import re
from cProfile import Profile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from pstats import Stats
from timeit import Timer
from typing import TYPE_CHECKING, Literal

from pyochain import Dict, Iter, Option, Seq, Some

import pql

if TYPE_CHECKING:
    from collections.abc import Callable

    from pql._frame import LazyFrame

type ScenarioFn = Callable[[], object]

ROOT = Path(__file__).resolve().parents[1]
STAT_LINE = re.compile(
    r"^\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.+):(\d+)\((.+)\)$"
)
TARGET_FUNCTIONS = Seq((
    "_compute_schema",
    "_extract_root_name",
    "_find_all",
    "into_resolved",
    "__init__",
    "qualify",
    "annotate_types",
    "find_all",
    "output_name",
))


@dataclass(slots=True, frozen=True)
class _Scenario:
    """Synthetic profiling scenario definition."""

    name: str
    repeat: int
    number: int
    profile_runs: int
    fn: ScenarioFn


@dataclass(slots=True, frozen=True)
class _TimingRow:
    """Timing summary for one scenario."""

    name: str
    best_ms: float
    avg_ms: float


@dataclass(slots=True, frozen=True)
class _StatRow:
    """Parsed profile row for a targeted function."""

    name: str
    primitive_calls: int
    total_calls: int
    total_time_ms: float
    cumulative_time_ms: float
    file: str
    line: int


def _wide_frame(width: int = 250, height: int = 6) -> LazyFrame:
    def _values() -> tuple[int, ...]:
        return tuple(range(height))

    return pql.LazyFrame(
        Iter(range(width)).map(lambda idx: (f"c{idx}", _values())).collect(Dict)
    )


def _prefixed_exprs(width: int) -> Seq[pql.Expr]:
    return (
        Iter(range(width))
        .map(lambda idx: f"c{idx}")
        .map(lambda name: pql.col(name).add(1).name.prefix("x_"))
        .collect()
    )


def _replacement_exprs(width: int) -> Seq[pql.Expr]:
    return (
        Iter(range(width))
        .map(lambda idx: f"c{idx}")
        .map(lambda name: pql.col(name).mul(2).alias(name))
        .collect()
    )


def _chain_with_columns(width: int, steps: int) -> LazyFrame:
    def _step(acc: LazyFrame, pair: tuple[int, str]) -> LazyFrame:
        idx, name = pair
        return acc.with_columns(pql.col(name).add(1).alias(f"n{idx}"))

    return (
        Iter(range(width))
        .map(lambda idx: f"c{idx}")
        .take(steps)
        .enumerate()
        .fold(_wide_frame(width=width), _step)
    )


def _scenarios() -> Seq[_Scenario]:
    frame = _wide_frame()
    columns = frame.columns
    prefixed = _prefixed_exprs(width=columns.length())
    replaced = _replacement_exprs(width=columns.length())

    return Seq((
        _Scenario(
            name="select_strings",
            repeat=5,
            number=40,
            profile_runs=35,
            fn=lambda: frame.select(columns),
        ),
        _Scenario(
            name="select_prefixed_exprs",
            repeat=5,
            number=20,
            profile_runs=20,
            fn=lambda: frame.select(prefixed),
        ),
        _Scenario(
            name="with_columns_replace",
            repeat=5,
            number=15,
            profile_runs=15,
            fn=lambda: frame.with_columns(replaced),
        ),
        _Scenario(
            name="select_numeric_selector",
            repeat=5,
            number=30,
            profile_runs=30,
            fn=lambda: frame.select(pql.selectors.numeric().name.prefix("num_")),
        ),
        _Scenario(
            name="chain_with_columns",
            repeat=4,
            number=3,
            profile_runs=4,
            fn=lambda: _chain_with_columns(width=80, steps=30),
        ),
    ))


def _time_scenario(scenario: _Scenario) -> _TimingRow:
    runs = Timer(scenario.fn).repeat(repeat=scenario.repeat, number=scenario.number)
    return _TimingRow(
        name=scenario.name,
        best_ms=min(runs) * 1000 / scenario.number,
        avg_ms=sum(runs) * 1000 / scenario.number / len(runs),
    )


def _parse_ncalls(raw: str) -> tuple[int, int]:
    match raw.split("/"):
        case [total, primitive]:
            return int(total), int(primitive)
        case [single]:
            calls = int(single)
            return calls, calls
        case _:
            msg = f"Unexpected call count format: {raw}"
            raise ValueError(msg)


def _targeted_rows(output: str) -> Seq[_StatRow]:
    def _into_row(line: str) -> _StatRow | None:
        match Option(STAT_LINE.match(line)):
            case Some(groups):
                total_calls, primitive_calls = _parse_ncalls(groups.group(1))
                file = Path(groups.group(4))
                name = groups.group(6)
                match str(ROOT) in str(file) and TARGET_FUNCTIONS.any(
                    lambda target: target == name
                ):
                    case True:
                        return _StatRow(
                            name=name,
                            primitive_calls=primitive_calls,
                            total_calls=total_calls,
                            total_time_ms=float(groups.group(2)) * 1000,
                            cumulative_time_ms=float(groups.group(3)) * 1000,
                            file=file.relative_to(ROOT).as_posix(),
                            line=int(groups.group(5)),
                        )
                    case False:
                        return None
            case _:
                return None

    rows = (
        Iter(output.splitlines())
        .filter_map(lambda line: Option(_into_row(line)))
        .sort(key=lambda row: row.cumulative_time_ms, reverse=True)
    )
    return Seq(tuple(rows[:12]))


def _profile_scenario(scenario: _Scenario) -> Seq[_StatRow]:
    profile = Profile()
    stream = StringIO()
    profile.enable()
    Iter(range(scenario.profile_runs)).for_each(lambda _idx: scenario.fn())
    profile.disable()
    _ = Stats(profile, stream=stream).sort_stats("cumulative").print_stats()
    return _targeted_rows(stream.getvalue())


def _render_timing(row: _TimingRow) -> str:
    return f"{row.name}: best={row.best_ms:.3f}ms/op avg={row.avg_ms:.3f}ms/op"


def _render_stat(row: _StatRow) -> str:
    return (
        f"  - {row.name}: cum={row.cumulative_time_ms:.3f}ms "
        f"tot={row.total_time_ms:.3f}ms calls={row.total_calls} "
        f"loc={row.file}:{row.line}"
    )


def _render_block(name: str, rows: Seq[_StatRow]) -> str:
    lines = rows.iter().map(_render_stat).join("\n")
    return f"{name}\n{lines}"


def _main(mode: Literal["bench", "profile", "all"] = "all") -> None:
    """Run timing and profiling views for synthetic planning workloads."""
    scenarios = _scenarios()

    match mode:
        case "bench":
            print(scenarios.iter().map(_time_scenario).map(_render_timing).join("\n"))
        case "profile":
            print(
                scenarios
                .iter()
                .map(
                    lambda scenario: _render_block(
                        scenario.name, _profile_scenario(scenario)
                    )
                )
                .join("\n\n")
            )
        case _:
            print("== timings ==")
            print(scenarios.iter().map(_time_scenario).map(_render_timing).join("\n"))
            print()
            print("== targeted profile ==")
            print(
                scenarios
                .iter()
                .map(
                    lambda scenario: _render_block(
                        scenario.name, _profile_scenario(scenario)
                    )
                )
                .join("\n\n")
            )


if __name__ == "__main__":
    _main()
