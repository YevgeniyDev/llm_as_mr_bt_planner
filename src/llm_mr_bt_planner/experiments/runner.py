"""Multi-trial experiment runner for reproducible evaluation.

LLM planning is stochastic, so a single run is not evidence. This runner sweeps
``scenarios x trials`` (optionally across providers, by calling it once per
client), records every trial, and aggregates per-scenario metrics with mean and
sample standard deviation. The output is plain data, ready for CSV/Markdown
export and for tabulation in a paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..domain import Scenario
from ..llm.base import LLMClient
from ..planner import PlannerResult, run_planner
from ..skills import skills_section_for


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
    method: str = "proposed"
    condition: str = "default"
    seed: int | None = None
    temperature: float | None = None
    metric_scope: str = "shared_validator_simulator"
    validation_error_types: tuple[str, ...] = ()
    simulation_error_types: tuple[str, ...] = ()
    synchronization_errors: int = 0
    capability_errors: int = 0
    causal_errors: int = 0
    structural_errors: int = 0
    deadlock: bool = False
    timeout: bool = False
    plan_nodes: int = 0
    action_nodes: int = 0
    condition_nodes: int = 0
    synchronization_edges: int = 0
    executed_actions: int = 0
    makespan_ticks: int = 0

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
            "method": self.method,
            "condition": self.condition,
            "seed": self.seed,
            "temperature": self.temperature,
            "metric_scope": self.metric_scope,
            "validation_error_types": list(self.validation_error_types),
            "simulation_error_types": list(self.simulation_error_types),
            "synchronization_errors": self.synchronization_errors,
            "capability_errors": self.capability_errors,
            "causal_errors": self.causal_errors,
            "structural_errors": self.structural_errors,
            "deadlock": self.deadlock,
            "timeout": self.timeout,
            "plan_nodes": self.plan_nodes,
            "action_nodes": self.action_nodes,
            "condition_nodes": self.condition_nodes,
            "synchronization_edges": self.synchronization_edges,
            "executed_actions": self.executed_actions,
            "makespan_ticks": self.makespan_ticks,
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
            "results": [result.to_dict() for result in self.results],
        }


_SYNC_ERRORS = {
    "invalid_synchronization", "missing_sync_condition", "missing_sync_producer",
    "condition_before_producer",
}
_CAPABILITY_ERRORS = {"invalid_capability", "unknown_robot"}
_CAUSAL_ERRORS = {
    "unsupported_goal", "unsupported_precondition", "unsupported_condition",
    "condition_before_producer", "missing_sync_producer",
}


def _walk_bt(node: Any):
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("children", []):
        yield from _walk_bt(child)


def _record_from_result(
    result: PlannerResult,
    *,
    scenario: str,
    trial: int,
    method: str,
    condition: str,
    seed: int | None,
    temperature: float | None,
) -> TrialRecord:
    validation_types = tuple(error.get("type", "unknown") for error in result.validation_errors)
    simulation_errors = result.simulation.get("errors", [])
    simulation_types = tuple(error.get("type", "unknown") for error in simulation_errors)
    nodes = [
        node
        for tree in result.plan.get("behavior_trees", {}).values()
        for node in _walk_bt(tree)
    ]
    trace = result.simulation.get("trace", [])
    action_events = [event for event in trace if event.get("event") == "action"]
    return TrialRecord(
        scenario=scenario,
        provider=result.provider,
        model=result.model,
        trial=trial,
        valid=result.valid,
        success=result.success,
        goal_success=result.goal_success,
        correction_rounds=result.correction_rounds,
        wall_seconds=result.wall_seconds,
        num_validation_errors=len(result.validation_errors),
        method=method,
        condition=condition,
        seed=seed,
        temperature=temperature,
        metric_scope=result.metric_scope,
        validation_error_types=validation_types,
        simulation_error_types=simulation_types,
        synchronization_errors=sum(kind in _SYNC_ERRORS for kind in validation_types),
        capability_errors=sum(kind in _CAPABILITY_ERRORS for kind in validation_types),
        causal_errors=sum(kind in _CAUSAL_ERRORS for kind in validation_types),
        structural_errors=sum(
            kind not in (_SYNC_ERRORS | _CAPABILITY_ERRORS | _CAUSAL_ERRORS)
            for kind in validation_types
        ),
        deadlock="deadlock" in simulation_types,
        timeout="timeout" in simulation_types,
        plan_nodes=len(nodes),
        action_nodes=sum(node.get("type") == "Action" for node in nodes),
        condition_nodes=sum(node.get("type") == "Condition" for node in nodes),
        synchronization_edges=len(result.plan.get("synchronization", [])),
        executed_actions=len(action_events),
        makespan_ticks=max((int(event.get("tick", 0)) for event in trace), default=0),
    )


def run_experiment(
    scenarios: list[Scenario],
    client: LLMClient | None,
    trials: int = 1,
    max_corrections: int = 4,
    max_ticks: int = 80,
    include_hints: bool = False,
    suggest_producers: bool = False,
    samples: int = 1,
    two_stage: bool = False,
    skills_dir: str | Path | None = None,
    on_trial: Callable[[TrialRecord], None] | None = None,
    runner: Callable[..., PlannerResult] = run_planner,
    method: str = "proposed",
    condition: str = "default",
    seeds: list[int] | None = None,
) -> ExperimentReport:
    """Sweep ``scenarios x trials`` with ``runner`` (the proposed method by default).

    Pass a baseline runner (see :mod:`llm_mr_bt_planner.baselines`) to evaluate a
    competing method under the identical validator+simulator and metrics. Baseline
    runners ignore the proposed-only keywords (``include_hints``/``suggest_producers``/
    ``two_stage``); they are still forwarded so the call site stays uniform.
    """
    report = ExperimentReport(
        config={
            "method": method,
            "condition": condition,
            "provider": getattr(client, "name", "unknown"),
            "model": getattr(client, "model", "unknown"),
            "temperature": getattr(client, "temperature", None),
            "seeds": list(seeds or []),
            "trials": trials,
            "max_corrections": max_corrections,
            "max_ticks": max_ticks,
            "include_hints": include_hints,
            "suggest_producers": suggest_producers,
            "samples": samples,
            "two_stage": two_stage,
            "skills": skills_dir is not None,
            "mode": "assisted" if (include_hints or suggest_producers) else "pure",
            "scenarios": [s.task_id for s in scenarios],
        }
    )
    for scenario in scenarios:
        # Skills are selected per-scenario (guidance relevant to its capabilities);
        # only the proposed method's prompts use it - baseline runners ignore it.
        skills_section = skills_section_for(scenario, skills_dir) if skills_dir else ""
        for trial in range(1, trials + 1):
            seed = seeds[trial - 1] if seeds and trial <= len(seeds) else None
            if client is not None and hasattr(client, "set_seed"):
                client.set_seed(seed)
            result = runner(
                scenario, client,
                max_corrections=max_corrections, max_ticks=max_ticks,
                include_hints=include_hints, suggest_producers=suggest_producers,
                samples=samples, two_stage=two_stage, skills_section=skills_section,
            )
            record = _record_from_result(
                result,
                scenario=scenario.task_id,
                trial=trial,
                method=method,
                condition=condition,
                seed=seed,
                temperature=getattr(client, "temperature", None),
            )
            report.trials.append(record)
            report.results.append(result)
            if on_trial is not None:
                on_trial(record)
    return report
