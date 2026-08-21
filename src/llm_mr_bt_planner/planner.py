"""Real LLM generation, deterministic evaluation, and bounded correction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .domain import Scenario
from .llm.base import LLMClient
from .plan import Plan, parse_plan
from .prompts import SYSTEM_PROMPT, build_correction_prompt, build_prompt, extract_json
from .simulation import SimulationReport, simulate, skipped_simulation
from .validation import ValidationReport, validate_plan

ProgressCallback = Callable[[str, float], None]
CancellationCheck = Callable[[], bool]


class PlanningCancelled(RuntimeError):
    """Raised after a user cancellation and before any final artifact is published."""


@dataclass
class PlannerResult:
    task_id: str
    provider: str
    model: str
    valid: bool
    success: bool
    goal_success: bool
    correction_rounds: int
    plan: dict[str, Any]
    validation_errors: list[dict[str, str]]
    simulation: dict[str, Any]
    wall_seconds: float = 0.0
    provider_responses: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "provider": self.provider,
            "model": self.model,
            "valid": self.valid,
            "success": self.success,
            "goal_success": self.goal_success,
            "correction_rounds": self.correction_rounds,
            "wall_seconds": round(self.wall_seconds, 3),
            "provider_responses": list(self.provider_responses),
            "plan": self.plan,
            "validation_errors": self.validation_errors,
            "simulation": self.simulation,
        }


def run_planner(
    scenario: Scenario,
    client: LLMClient,
    max_corrections: int = 4,
    max_ticks: int = 100,
    *,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCheck | None = None,
) -> PlannerResult:
    """Run one initial request followed by at most ``max_corrections`` repairs."""
    start = time.monotonic()
    _check_cancelled(cancelled)
    _notify(progress, "Building the initial planning prompt", 0.02)
    prompt = build_prompt(scenario)

    plan, validation, simulation = _request_and_evaluate(
        client,
        prompt,
        scenario,
        max_ticks=max_ticks,
        label="Initial candidate",
        start_fraction=0.06,
        end_fraction=0.42,
        progress=progress,
        cancelled=cancelled,
    )
    rounds = 0

    while rounds < max_corrections and not (validation.valid and simulation.success):
        rounds += 1
        _check_cancelled(cancelled)
        error_types = sorted({error.type for error in validation.errors})
        simulation_types = sorted({str(error.get("type", "simulation_error")) for error in simulation.errors})
        reasons = ", ".join([*error_types, *simulation_types]) or "goal not reached"
        _notify(progress, f"Correction round {rounds}/{max_corrections} started: {reasons}", 0.44)
        correction = build_correction_prompt(
            scenario,
            validation.to_dicts(),
            simulation,
            previous_plan=plan.raw or plan.to_dict(),
        )
        span_start = 0.44 + (rounds - 1) * (0.46 / max(1, max_corrections))
        span_end = 0.44 + rounds * (0.46 / max(1, max_corrections))
        plan, validation, simulation = _request_and_evaluate(
            client,
            correction,
            scenario,
            max_ticks=max_ticks,
            label=f"Correction round {rounds}",
            start_fraction=span_start,
            end_fraction=span_end,
            progress=progress,
            cancelled=cancelled,
        )

    accepted = validation.valid and simulation.success
    if accepted:
        _notify(progress, "Candidate accepted: validation passed and all symbolic goals were reached", 0.94)
    else:
        _notify(
            progress,
            f"Candidate rejected after {rounds} correction round(s); no final BT will be published",
            0.94,
        )
    return PlannerResult(
        task_id=scenario.task_id,
        provider=getattr(client, "name", "unknown"),
        model=getattr(client, "model", "unknown"),
        valid=validation.valid,
        success=simulation.success,
        goal_success=simulation.goal_success,
        correction_rounds=rounds,
        # Keep the exact extracted candidate for the independent service-level
        # recheck. Invalid/unknown fields must never disappear through parsing.
        plan=plan.raw or plan.to_dict(),
        validation_errors=validation.to_dicts(),
        simulation=simulation.to_dict(),
        wall_seconds=time.monotonic() - start,
        provider_responses=tuple(getattr(client, "response_metadata", ())),
    )


def _request_and_evaluate(
    client: LLMClient,
    prompt: str,
    scenario: Scenario,
    *,
    max_ticks: int,
    label: str,
    start_fraction: float,
    end_fraction: float,
    progress: ProgressCallback | None,
    cancelled: CancellationCheck | None,
) -> tuple[Plan, ValidationReport, SimulationReport]:
    span = end_fraction - start_fraction
    _check_cancelled(cancelled)
    _notify(progress, f"{label}: sending request to {getattr(client, 'name', 'provider')}", start_fraction)
    raw_text = client.complete(SYSTEM_PROMPT, prompt)
    _check_cancelled(cancelled)
    _notify(progress, f"{label}: provider response received", start_fraction + span * 0.35)

    try:
        _notify(progress, f"{label}: parsing the complete LLM-generated BT", start_fraction + span * 0.45)
        candidate = extract_json(raw_text)
        plan = parse_plan(candidate)
    except (TypeError, ValueError) as error:
        plan = parse_plan({})
        validation = ValidationReport()
        validation.add("invalid_llm_json", str(error))
        _notify(progress, f"{label}: response rejected during JSON parsing: {error}", end_fraction)
        return plan, validation, skipped_simulation()

    _check_cancelled(cancelled)
    _notify(progress, f"{label}: running static validation", start_fraction + span * 0.72)
    validation = validate_plan(plan, scenario, suggest_producers=True)
    if not validation.valid:
        kinds = ", ".join(sorted({error.type for error in validation.errors}))
        _notify(
            progress,
            f"{label}: static validation failed with {len(validation.errors)} error(s): {kinds}",
            end_fraction,
        )
        return plan, validation, skipped_simulation()

    _notify(progress, f"{label}: static validation passed", start_fraction + span * 0.82)
    _check_cancelled(cancelled)
    _notify(progress, f"{label}: running deterministic contract simulation", start_fraction + span * 0.88)
    simulation = simulate(plan, scenario, max_ticks=max_ticks)
    if simulation.success:
        _notify(progress, f"{label}: contract simulation passed", end_fraction)
    else:
        kinds = ", ".join(sorted({str(error.get("type", "simulation_error")) for error in simulation.errors}))
        _notify(progress, f"{label}: contract simulation failed: {kinds}", end_fraction)
    return plan, validation, simulation


def _notify(callback: ProgressCallback | None, message: str, fraction: float) -> None:
    if callback is not None:
        callback(message, max(0.0, min(1.0, fraction)))


def _check_cancelled(cancelled: CancellationCheck | None) -> None:
    if cancelled is not None and cancelled():
        raise PlanningCancelled("Pipeline cancelled by the user; no final BT was published.")
