"""Shared scaffolding for baseline plan generators.

Every baseline produces a :class:`~llm_mr_bt_planner.plan.Plan` and is scored by the
*same* task-agnostic validator and tick simulator as the proposed method, so all
methods yield identical metric columns (validity, goal success, sync error,
correction rounds, time). A baseline differs only in *how* it produces the plan -
which prompt, which assists, whether it has a self-correction loop. This module
holds the pieces common to all of them so each baseline file stays small.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..domain import Scenario
from ..llm.base import LLMClient
from ..plan import Plan, parse_plan
from ..planner import PlannerResult
from ..prompts import SYSTEM_PROMPT, extract_json
from ..simulation import SimulationReport, simulate, skipped_simulation
from ..validation import ValidationReport, validate_plan


def score_plan(
    plan: Plan, scenario: Scenario, max_ticks: int = 80
) -> tuple[ValidationReport, SimulationReport]:
    """Validate then (if structurally valid) simulate a plan - the shared instrument.

    Baselines are evaluated in *pure* mode: ``suggest_producers`` is never enabled,
    so the validator reports what is wrong without naming candidate producers.
    """
    validation = validate_plan(plan, scenario)
    simulation = (
        simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
    )
    return validation, simulation


def candidate_score(validation: ValidationReport, simulation: SimulationReport) -> tuple:
    """Lexicographic best-of-N ranking, identical to the proposed planner's."""
    return (validation.valid and simulation.success, validation.valid, -len(validation.errors))


def query_plan(client: LLMClient, prompt: str, scenario: Scenario) -> Plan:
    """One LLM call -> parsed plan. Mirrors ``planner._query_plan``."""
    raw = extract_json(client.complete(SYSTEM_PROMPT, prompt))
    raw.setdefault("task_id", scenario.task_id)
    return parse_plan(raw)


def generate_best(
    client: LLMClient,
    prompt: str,
    scenario: Scenario,
    samples: int,
    max_ticks: int,
) -> tuple[Plan, ValidationReport, SimulationReport]:
    """Sample up to ``samples`` plans from one prompt; keep the best (pure mode)."""
    best: tuple[Plan, ValidationReport, SimulationReport, tuple] | None = None
    for _ in range(max(1, samples)):
        plan = query_plan(client, prompt, scenario)
        validation, simulation = score_plan(plan, scenario, max_ticks=max_ticks)
        score = candidate_score(validation, simulation)
        if best is None or score > best[3]:
            best = (plan, validation, simulation, score)
        if validation.valid and simulation.success:
            break
    assert best is not None
    return best[0], best[1], best[2]


def make_result(
    scenario: Scenario,
    client: LLMClient,
    plan: Plan,
    validation: ValidationReport,
    simulation: SimulationReport,
    rounds: int,
    wall_seconds: float,
) -> PlannerResult:
    """Pack a scored plan into the same ``PlannerResult`` the proposed method returns."""
    return PlannerResult(
        task_id=scenario.task_id,
        provider=getattr(client, "name", "unknown"),
        model=getattr(client, "model", "unknown"),
        valid=validation.valid,
        success=simulation.success,
        goal_success=simulation.goal_success,
        correction_rounds=rounds,
        plan=plan.to_dict(),
        validation_errors=validation.to_dicts(),
        simulation={
            "final_state": simulation.final_state,
            "trace": simulation.trace,
            "errors": simulation.errors,
        },
        wall_seconds=wall_seconds,
    )


class BaselineRunner(Protocol):
    """Uniform call signature so a baseline is drop-in for ``run_planner``.

    Proposed-only knobs (``include_hints``, ``suggest_producers``, ``two_stage``) are
    accepted and ignored by baselines, so the experiment runner can call any method
    with the same keyword arguments.
    """

    def __call__(
        self,
        scenario: Scenario,
        client: LLMClient,
        *,
        max_corrections: int = 4,
        max_ticks: int = 80,
        samples: int = 1,
        **kwargs: Any,
    ) -> PlannerResult: ...
