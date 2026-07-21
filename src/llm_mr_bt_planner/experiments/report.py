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


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for a Bernoulli rate."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def aggregate(trials: Iterable[TrialRecord]) -> list[dict[str, Any]]:
    """Per-scenario aggregate metrics over its trials."""
    by_scenario: dict[tuple[str, str, str, str], list[TrialRecord]] = {}
    for trial in trials:
        key = (trial.method, trial.condition, trial.scenario, trial.metric_scope)
        by_scenario.setdefault(key, []).append(trial)

    rows: list[dict[str, Any]] = []
    for (method, condition, scenario, metric_scope), records in by_scenario.items():
        n = len(records)
        rounds = [float(r.correction_rounds) for r in records]
        seconds = [r.wall_seconds for r in records]
        goal_count = sum(r.goal_success for r in records)
        valid_count = sum(r.valid for r in records)
        goal_ci = _wilson(goal_count, n)
        valid_ci = _wilson(valid_count, n)
        rows.append(
            {
                "method": method,
                "condition": condition,
                "scenario": scenario,
                "metric_scope": metric_scope,
                "trials": n,
                "validity_rate": round(valid_count / n, 3),
                "validity_ci95": f"[{valid_ci[0]}, {valid_ci[1]}]",
                "success_rate": round(sum(r.success for r in records) / n, 3),
                "goal_success_rate": round(goal_count / n, 3),
                "goal_success_ci95": f"[{goal_ci[0]}, {goal_ci[1]}]",
                "synchronization_error_rate": round(sum(r.synchronization_errors > 0 for r in records) / n, 3),
                "capability_error_rate": round(sum(r.capability_errors > 0 for r in records) / n, 3),
                "causal_error_rate": round(sum(r.causal_errors > 0 for r in records) / n, 3),
                "structural_error_rate": round(sum(r.structural_errors > 0 for r in records) / n, 3),
                "deadlock_rate": round(sum(r.deadlock for r in records) / n, 3),
                "timeout_rate": round(sum(r.timeout for r in records) / n, 3),
                "mean_correction_rounds": round(_mean(rounds), 3),
                "std_correction_rounds": round(_stddev(rounds), 3),
                "mean_wall_seconds": round(_mean(seconds), 3),
                "mean_plan_nodes": round(_mean([float(r.plan_nodes) for r in records]), 3),
                "mean_action_nodes": round(_mean([float(r.action_nodes) for r in records]), 3),
                "mean_condition_nodes": round(_mean([float(r.condition_nodes) for r in records]), 3),
                "mean_sync_edges": round(_mean([float(r.synchronization_edges) for r in records]), 3),
                "mean_executed_actions": round(_mean([float(r.executed_actions) for r in records]), 3),
                "mean_makespan_ticks": round(_mean([float(r.makespan_ticks) for r in records]), 3),
            }
        )
    return rows


_COLUMNS = [
    "method",
    "condition",
    "scenario",
    "metric_scope",
    "trials",
    "validity_rate",
    "validity_ci95",
    "success_rate",
    "goal_success_rate",
    "goal_success_ci95",
    "synchronization_error_rate",
    "capability_error_rate",
    "causal_error_rate",
    "structural_error_rate",
    "deadlock_rate",
    "timeout_rate",
    "mean_correction_rounds",
    "std_correction_rounds",
    "mean_wall_seconds",
    "mean_plan_nodes",
    "mean_action_nodes",
    "mean_condition_nodes",
    "mean_sync_edges",
    "mean_executed_actions",
    "mean_makespan_ticks",
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
        lines.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in _COLUMNS) + " |")
    return "\n".join(lines)


def _latex_escape(value: Any) -> str:
    text = str(value)
    for source, target in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&")):
        text = text.replace(source, target)
    return text


def _pct(value: Any) -> str:
    return f"{100 * float(value):.1f}"


def to_latex_tables(trials: Iterable[TrialRecord]) -> str:
    """Render paper-ready main and diagnostic comparison tables."""
    rows = aggregate(trials)
    lines = [
        "% AUTO-GENERATED by the experiment snapshot pipeline. Do not edit by hand.",
        r"\begin{table*}[!t]",
        r"\caption{Main comparison. Rates are percentages; MRBTP rows marked native use its native execution semantics.}",
        r"\label{tab:results}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lllrrrrrrrr}",
        r"\toprule",
        r"Method & Condition & Scenario & $N$ & Valid & Goal & Sync err. & Deadlock & Nodes & Ticks & Time (s) \\",
        r"\midrule",
    ]
    for row in rows:
        method = _latex_escape(row["method"])
        if row["metric_scope"] != "shared_validator_simulator":
            method += r"$^{\dagger}$"
        lines.append(
            f"{method} & {_latex_escape(row['condition'])} & {_latex_escape(row['scenario'])} & "
            f"{row['trials']} & {_pct(row['validity_rate'])} & {_pct(row['goal_success_rate'])} & "
            f"{_pct(row['synchronization_error_rate'])} & {_pct(row['deadlock_rate'])} & "
            f"{row['mean_plan_nodes']:.1f} & {row['mean_makespan_ticks']:.1f} & "
            f"{row['mean_wall_seconds']:.2f} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{0.2em}\parbox{0.98\textwidth}{\scriptsize $^{\dagger}$Native MRBTP protocol; not re-simulated with blocking-guard semantics. Validity and goal success require a grounded condition and extractable trees.}",
        r"\end{table*}",
        "",
        r"\begin{table*}[!t]",
        r"\caption{Diagnostic and complexity metrics. Error and timeout columns are percentages.}",
        r"\label{tab:diagnostics}",
        r"\centering\scriptsize",
        r"\begin{tabular}{lllrrrrrrrrr}",
        r"\toprule",
        r"Method & Condition & Scenario & Cap. err. & Causal err. & Struct. err. & Timeout & Actions & Conditions & Sync edges & Rounds \\",
        r"\midrule",
    ])
    for row in rows:
        lines.append(
            f"{_latex_escape(row['method'])} & {_latex_escape(row['condition'])} & "
            f"{_latex_escape(row['scenario'])} & {_pct(row['capability_error_rate'])} & "
            f"{_pct(row['causal_error_rate'])} & {_pct(row['structural_error_rate'])} & "
            f"{_pct(row['timeout_rate'])} & {row['mean_action_nodes']:.1f} & "
            f"{row['mean_condition_nodes']:.1f} & {row['mean_sync_edges']:.1f} & "
            f"{row['mean_correction_rounds']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)
