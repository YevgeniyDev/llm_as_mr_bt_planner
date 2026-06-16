"""Multi-trial experiment runner for reproducible evaluation.

LLM planning is stochastic, so a single run is not evidence. This runner sweeps
``scenarios x trials`` (optionally across providers, by calling it once per
client), records every trial, and aggregates per-scenario metrics with mean and
sample standard deviation. The output is plain data, ready for CSV/Markdown
export and for tabulation in a paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..domain import Scenario
from ..llm.base import LLMClient
from ..planner import PlannerResult, run_planner


@dataclass
class TrialRecord:
    scenario: str
    provider: str
    model: str
    trial: int
    valid: bool
    success: bool
    goal_success: bool
    correction_rounds: int
    wall_seconds: float
    num_validation_errors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "provider": self.provider,
            "model": self.model,
            "trial": self.trial,
            "valid": self.valid,
            "success": self.success,
            "goal_success": self.goal_success,
            "correction_rounds": self.correction_rounds,
            "wall_seconds": round(self.wall_seconds, 3),
            "num_validation_errors": self.num_validation_errors,
        }


@dataclass
class ExperimentReport:
    config: dict[str, Any]
    trials: list[TrialRecord] = field(default_factory=list)
    results: list[PlannerResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from .report import aggregate

        return {
            "config": self.config,
            "trials": [t.to_dict() for t in self.trials],
            "aggregates": aggregate(self.trials),
        }


def run_experiment(
    scenarios: list[Scenario],
    client: LLMClient,
    trials: int = 1,
    max_corrections: int = 4,
    max_ticks: int = 80,
    include_hints: bool = False,
    suggest_producers: bool = False,
    samples: int = 1,
    two_stage: bool = False,
    on_trial: Callable[[TrialRecord], None] | None = None,
) -> ExperimentReport:
    report = ExperimentReport(
        config={
            "provider": getattr(client, "name", "unknown"),
            "model": getattr(client, "model", "unknown"),
            "trials": trials,
            "max_corrections": max_corrections,
            "max_ticks": max_ticks,
            "include_hints": include_hints,
            "suggest_producers": suggest_producers,
            "samples": samples,
            "two_stage": two_stage,
            "mode": "assisted" if (include_hints or suggest_producers) else "pure",
            "scenarios": [s.task_id for s in scenarios],
        }
    )
    for scenario in scenarios:
        for trial in range(1, trials + 1):
            result = run_planner(
                scenario, client,
                max_corrections=max_corrections, max_ticks=max_ticks,
                include_hints=include_hints, suggest_producers=suggest_producers,
                samples=samples, two_stage=two_stage,
            )
            record = TrialRecord(
                scenario=scenario.task_id,
                provider=result.provider,
                model=result.model,
                trial=trial,
                valid=result.valid,
                success=result.success,
                goal_success=result.goal_success,
                correction_rounds=result.correction_rounds,
                wall_seconds=result.wall_seconds,
                num_validation_errors=len(result.validation_errors),
            )
            report.trials.append(record)
            report.results.append(result)
            if on_trial is not None:
                on_trial(record)
    return report
