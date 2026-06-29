"""The generate -> validate -> simulate -> self-correct loop (Algorithm 1)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .domain import Scenario
from .llm.base import LLMClient
from .plan import Plan, parse_plan
from .prompts import (
    SYSTEM_PROMPT,
    build_action_plan_correction_prompt,
    build_action_plan_prompt,
    build_bt_encoding_correction_prompt,
    build_bt_encoding_prompt,
    build_correction_prompt,
    build_prompt,
    extract_json,
)
from .simulation import SimulationReport, simulate, skipped_simulation
from .validation import ValidationReport, validate_plan


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
            "plan": self.plan,
            "validation_errors": self.validation_errors,
            "simulation": self.simulation,
        }


def run_planner(
    scenario: Scenario,
    client: LLMClient,
    max_corrections: int = 4,
    max_ticks: int = 80,
    include_hints: bool = False,
    suggest_producers: bool = False,
    samples: int = 1,
    two_stage: bool = False,
) -> PlannerResult:
    """Run Algorithm 1.

    Default mode is *pure*: the LLM gets only the prompt + initial state +
    output schema (plus a general, task-agnostic planning method), and the
    validator reports problems without naming specific producer actions.
    ``include_hints`` / ``suggest_producers`` enable the assisted (ablation)
    condition. ``samples`` > 1 enables best-of-N. ``two_stage`` decomposes
    generation: the LLM first emits an ordered per-robot action plan (validated
    on its own by running it as condition-free sequences), then encodes that
    fixed action plan into behavior trees with explicit synchronization. Set
    ``max_corrections=0`` for single-shot generation.
    """
    start = time.monotonic()
    generate = _two_stage_generate if two_stage else _generate_evaluated
    plan, validation, simulation, rounds = generate(
        scenario, client,
        max_corrections=max_corrections, max_ticks=max_ticks,
        include_hints=include_hints, suggest_producers=suggest_producers, samples=samples,
    )
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
        wall_seconds=time.monotonic() - start,
    )


def _generate_evaluated(
    scenario: Scenario,
    client: LLMClient,
    max_corrections: int,
    max_ticks: int,
    include_hints: bool,
    suggest_producers: bool,
    samples: int,
) -> tuple[Plan, ValidationReport, SimulationReport, int]:
    prompt = build_prompt(scenario, include_hints=include_hints)
    plan, validation, simulation = _generate_best(
        client, prompt, scenario, samples, max_ticks, suggest_producers
    )
    rounds = 0

    while rounds < max_corrections and not (validation.valid and simulation.success):
        rounds += 1
        correction = build_correction_prompt(
            scenario, validation.to_dicts(), simulation,
            previous_plan=plan.to_dict(), include_hints=include_hints,
        )
        plan, validation, simulation = _generate_best(
            client, correction, scenario, samples, max_ticks, suggest_producers
        )

    return plan, validation, simulation, rounds


def _generate_best(
    client: LLMClient,
    prompt: str,
    scenario: Scenario,
    samples: int,
    max_ticks: int,
    suggest_producers: bool,
) -> tuple[Plan, ValidationReport, SimulationReport]:
    """Generate up to ``samples`` candidate plans, returning the best one.

    Stops early on the first valid+successful candidate. With ``samples == 1``
    this is a single generation. Ranking: (valid and success) > valid >
    fewer validation errors.
    """
    best: tuple[Plan, ValidationReport, SimulationReport, tuple] | None = None
    for _ in range(max(1, samples)):
        plan = _query_plan(client, prompt, scenario)
        validation = validate_plan(plan, scenario, suggest_producers=suggest_producers)
        simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
        score = (validation.valid and simulation.success, validation.valid, -len(validation.errors))
        if best is None or score > best[3]:
            best = (plan, validation, simulation, score)
        if validation.valid and simulation.success:
            break
    assert best is not None
    return best[0], best[1], best[2]


def _query_plan(client: LLMClient, prompt: str, scenario: Scenario) -> Plan:
    raw_text = client.complete(SYSTEM_PROMPT, prompt)
    raw = extract_json(raw_text)
    raw.setdefault("task_id", scenario.task_id)
    return parse_plan(raw)


# --------------------------------------------------------------------------- #
# Two-stage generation: action plan -> behavior trees
# --------------------------------------------------------------------------- #


def _two_stage_generate(
    scenario: Scenario,
    client: LLMClient,
    max_corrections: int,
    max_ticks: int,
    include_hints: bool,
    suggest_producers: bool,
    samples: int,
) -> tuple[Plan, ValidationReport, SimulationReport, int]:
    # Stage 1: a feasible, ordered per-robot action plan (no conditions/sync).
    action_plan, validation, simulation = _generate_best_actions(
        client, build_action_plan_prompt(scenario, include_hints=include_hints),
        scenario, samples, max_ticks, suggest_producers,
    )
    rounds = 0
    while rounds < max_corrections and not (validation.valid and simulation.success):
        rounds += 1
        prompt = build_action_plan_correction_prompt(
            scenario, validation.to_dicts(), simulation, action_plan, include_hints=include_hints
        )
        action_plan, validation, simulation = _generate_best_actions(
            client, prompt, scenario, samples, max_ticks, suggest_producers
        )

    # Stage 2: encode the fixed action plan as behavior trees with synchronization.
    plan, validation, simulation = _generate_best(
        client, build_bt_encoding_prompt(scenario, action_plan, include_hints=include_hints),
        scenario, samples, max_ticks, suggest_producers,
    )
    while rounds < 2 * max_corrections and not (validation.valid and simulation.success):
        rounds += 1
        prompt = build_bt_encoding_correction_prompt(
            scenario, validation.to_dicts(), simulation, plan.to_dict(), action_plan, include_hints=include_hints
        )
        plan, validation, simulation = _generate_best(
            client, prompt, scenario, samples, max_ticks, suggest_producers
        )

    return plan, validation, simulation, rounds


def _generate_best_actions(
    client: LLMClient,
    prompt: str,
    scenario: Scenario,
    samples: int,
    max_ticks: int,
    suggest_producers: bool,
) -> tuple[dict[str, Any], ValidationReport, SimulationReport]:
    """Sample action plans; rank by feasibility of the condition-free sequences."""
    best: tuple[dict[str, Any], ValidationReport, SimulationReport, tuple] | None = None
    for _ in range(max(1, samples)):
        raw = extract_json(client.complete(SYSTEM_PROMPT, prompt))
        action_plan = raw.get("action_plan", raw) if isinstance(raw, dict) else {}
        plan = _synthesize_plan(action_plan)
        validation = validate_plan(plan, scenario, suggest_producers=suggest_producers)
        simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
        score = (validation.valid and simulation.success, validation.valid, -len(validation.errors))
        if best is None or score > best[3]:
            best = (action_plan, validation, simulation, score)
        if validation.valid and simulation.success:
            break
    assert best is not None
    return best[0], best[1], best[2]


def _synthesize_plan(action_plan: dict[str, Any]) -> Plan:
    """Turn an action plan into a condition-free Plan: each robot is a Sequence of
    its actions, with a matching task graph and assignments. Inter-robot ordering
    is checked by the simulator (actions block until their preconditions hold), so
    a feasible action plan simulates to success even without explicit conditions.
    """
    task_graph: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    behavior_trees: dict[str, Any] = {}
    for robot, actions in (action_plan or {}).items():
        if not isinstance(actions, list):
            continue
        children = []
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                continue
            name = action.get("action")
            parameters = action.get("parameters", [])
            children.append({"type": "Action", "name": name, "parameters": parameters})
            task_id = f"{robot}__{index}"
            task_graph.append({"id": task_id, "action": name, "parameters": parameters, "depends_on": []})
            assignments.append({"task_id": task_id, "robot": robot})
        behavior_trees[robot] = {"type": "Sequence", "children": children}
    return parse_plan(
        {
            "task_graph": task_graph,
            "assignments": assignments,
            "synchronization": [],
            "behavior_trees": behavior_trees,
        }
    )
