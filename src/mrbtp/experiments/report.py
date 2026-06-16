"""Metrics aggregation and table exporters (CSV / Markdown / JSON)."""

from __future__ import annotations

import csv
import io
import math
from typing import Any, Iterable

from .runner import TrialRecord


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def aggregate(trials: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    """Per-scenario aggregate metrics over its trials."""
    by_scenario: dict[str, list[TrialRecord]] = {}
    for trial in trials:
        by_scenario.setdefault(trial.scenario, []).append(trial)

    rows: list[dict[str, Any]] = []
    for scenario, records in by_scenario.items():
        n = len(records)
        rounds = [float(r.correction_rounds) for r in records]
        seconds = [r.wall_seconds for r in records]
        rows.append(
            {
                "scenario": scenario,
                "trials": n,
                "validity_rate": round(sum(r.valid for r in records) / n, 3),
                "success_rate": round(sum(r.success for r in records) / n, 3),
                "goal_success_rate": round(sum(r.goal_success for r in records) / n, 3),
                "mean_correction_rounds": round(_mean(rounds), 3),
                "std_correction_rounds": round(_stddev(rounds), 3),
                "mean_wall_seconds": round(_mean(seconds), 3),
            }
        )
    return rows


_COLUMNS = [
    "scenario",
    "trials",
    "validity_rate",
    "success_rate",
    "goal_success_rate",
    "mean_correction_rounds",
    "std_correction_rounds",
    "mean_wall_seconds",
]


def to_csv(trials: Iterable[TrialRecord]) -> str:
    rows = aggregate(trials)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def to_markdown_table(trials: Iterable[TrialRecord]) -> str:
    rows = aggregate(trials)
    header = "| " + " | ".join(_COLUMNS) + " |"
    divider = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row[column]) for column in _COLUMNS) + " |")
    return "\n".join(lines)
