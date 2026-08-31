"""Failure-aware five-agent recovery for a dropped inspection tool."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable

from .bt import iter_nodes
from .domain import Capability, Effects, Entity, Resource, Scenario
from .plan import Plan, parse_plan
from .predicates import canonical_predicate
from .prompts import build_prompt
from .recovery import OpenAIResponsesRecoveryClient
from .simulation import SimulationReport, simulate
from .validation import ValidationReport, validate_plan

INSPECTION_ROBOTS = (
    "b2_base",
    "z1_thermal_arm",
    "husky_base",
    "husky_franka",
    "static_franka",
)

INSPECTION_RECOVERY_SYSTEM_PROMPT = (
    "You are the online recovery planner for a five-controller heterogeneous robot team. "
    "The only inspection tool fell after the nominal plan was generated. Return a complete "
    "continuation Behavior Tree for every declared robot. First use B2 and its mounted Z1 "
    "camera to search for and localize the same fallen tool. Then send Husky to the measured "
    "recovery dock and use Husky's mounted Franka to pick up and secure that tool. Continue "
    "the original solar/pipe inspection from the exact current state without a reset. Use "
    "only declared capabilities, predicates, resources, and entities; do not invent a spare "
    "tool, successful observations, or motor commands. Use only Sequence and Fallback composites."
)

Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class InspectionRecoveryPlanningResult:
    plan: Plan
    runtime_scenario: Scenario
    validation: ValidationReport
    simulation: SimulationReport
    attempts: tuple[dict[str, Any], ...]
    provider: str
    model: str
    reasoning_effort: str | None


def plan_inspection_tool_recovery(
    client: OpenAIResponsesRecoveryClient,
    scenario: Scenario,
    *,
    measured_initial_state: tuple[str, ...],
    failure_observation: dict[str, Any],
    nominal_plan: Plan,
    max_corrections: int = 3,
    max_ticks: int = 400,
    progress: Progress | None = None,
) -> InspectionRecoveryPlanningResult:
    runtime = build_inspection_tool_recovery_scenario(
        scenario,
        measured_initial_state=measured_initial_state,
        failure_observation=failure_observation,
    )
    attempts: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] | None = None
    _progress(progress, "Prepared the measured five-agent post-failure state", 0.03)
    for round_index in range(max_corrections + 1):
        label = "Initial recovery candidate" if round_index == 0 else f"Recovery correction {round_index}/{max_corrections}"
        prompt = build_inspection_recovery_prompt(
            runtime,
            failure_observation=failure_observation,
            nominal_plan=nominal_plan,
            previous_diagnostics=diagnostics,
        )
        _progress(progress, f"{label}: sending failure snapshot and failed BT to {client.provider}/{client.model}", 0.06 + 0.70 * round_index / (max_corrections + 1))
        document, provenance = client.complete(INSPECTION_RECOVERY_SYSTEM_PROMPT, prompt)
        plan = parse_plan(document)
        _progress(progress, f"{label}: validating the complete five-agent continuation", 0.18 + 0.70 * round_index / (max_corrections + 1))
        validation = validate_plan(plan, runtime, suggest_producers=True)
        simulation = (
            simulate(plan, runtime, max_ticks=max_ticks)
            if validation.valid
            else SimulationReport(
                success=False,
                goal_success=False,
                final_state=list(runtime.initial_state),
                trace=[],
                errors=[{"type": "static_validation_failed"}],
            )
        )
        semantic_errors = _semantic_errors(plan)
        accepted = validation.valid and simulation.success and not semantic_errors
        attempts.append(
            {
                "round": round_index,
                "accepted": accepted,
                "prompt": {"system": INSPECTION_RECOVERY_SYSTEM_PROMPT, "user": prompt},
                "provenance": provenance,
                "candidate": document,
                "validation": {"valid": validation.valid, "errors": validation.to_dicts()},
                "contract_simulation": simulation.to_dict(),
                "recovery_semantic_errors": semantic_errors,
            }
        )
        if accepted:
            _progress(progress, f"{label}: accepted; B2 localization and Husky pickup are present", 0.94)
            _progress(progress, "Validated five-agent continuation is ready for MuJoCo", 1.0)
            return InspectionRecoveryPlanningResult(
                plan=plan,
                runtime_scenario=runtime,
                validation=validation,
                simulation=simulation,
                attempts=tuple(attempts),
                provider=client.provider,
                model=client.model,
                reasoning_effort=client.reasoning_effort,
            )
        diagnostics = {
            "validation_errors": validation.to_dicts(),
            "simulation_errors": simulation.errors,
            "simulation_final_state": simulation.final_state,
            "recovery_semantic_errors": semantic_errors,
            "rejected_candidate": document,
        }
        reasons = sorted(
            {
                *(error.type for error in validation.errors),
                *(str(error.get("type", "simulation_error")) for error in simulation.errors),
                *semantic_errors,
            }
        )
        _progress(progress, f"{label}: rejected ({', '.join(reasons) or 'goals not reached'})", 0.22 + 0.70 * (round_index + 1) / (max_corrections + 1))
    raise RuntimeError(
        f"Five-agent recovery failed validation/simulation after {max_corrections + 1} attempt(s)."
    )


def build_inspection_recovery_prompt(
    runtime_scenario: Scenario,
    *,
    failure_observation: dict[str, Any],
    nominal_plan: Plan,
    previous_diagnostics: dict[str, Any] | None = None,
) -> str:
    prompt_observation = {
        key: value
        for key, value in failure_observation.items()
        if key != "measured_position_m_for_audit"
    }
    sections = [
        "Return a complete continuation BT for all five robot controllers.",
        "MuJoCo remains frozen at the exact measured failure state; execution resumes without reset.",
        (
            "The single inspection_kit is intact but its floor position was not known when the failure "
            "was raised. The continuation must use navigate_b2_tool_search, deploy_camera_tool_search, "
            "localize_fallen_tool, and stow_z1_after_tool_search. Then it must use "
            "navigate_husky_tool_recovery and recover_localized_tool before proceeding through "
            "navigate_husky_recovery_to_reference and navigate_b2_search_to_solar. Do not use the "
            "failed handoff load action and do not invent a replacement tool."
        ),
        f"Failure observation:\n{json.dumps(prompt_observation, indent=2)}",
        f"Failed nominal BT:\n{json.dumps(nominal_plan.to_dict(), indent=2)}",
        build_prompt(runtime_scenario),
    ]
    if previous_diagnostics is not None:
        sections.append(
            "The previous continuation was rejected. Return a complete corrected replacement.\n"
            f"Diagnostics:\n{json.dumps(previous_diagnostics, indent=2)}"
        )
    return "\n\n".join(sections)


def build_inspection_tool_recovery_scenario(
    scenario: Scenario,
    *,
    measured_initial_state: tuple[str, ...],
    failure_observation: dict[str, Any],
) -> Scenario:
    expected = {
        "classification": "tool_dropped_and_location_unknown",
        "object": "inspection_kit",
        "object_usable": True,
        "requires_localization": True,
    }
    mismatches = [key for key, value in expected.items() if failure_observation.get(key) != value]
    if mismatches:
        raise ValueError(f"Dropped-tool recovery observation mismatch: {', '.join(mismatches)}.")
    if scenario.task_id != "five_agent_solar_pipe_inspection":
        raise ValueError("Five-agent dropped-tool recovery requires the solar/pipe inspection mission.")
    if any(scenario.capability(robot, "recover_localized_tool") for robot in INSPECTION_ROBOTS):
        raise ValueError("Nominal scenario already exposes fault recovery; fault blindness is violated.")

    additions: dict[str, tuple[Capability, ...]] = {
        "b2_base": (
            _cap(
                "navigate_b2_tool_search",
                (), (), ("b2_route",), "navigation", 5, 40,
                (
                    "system_ready()", "robot_ready(b2_base)",
                    "stationary(b2_base)", "stowed(z1_thermal_arm)",
                ),
                ("docked(b2_base,tool_search_view)", "stationary(b2_base)"),
                ("docked(b2_base)",),
            ),
            _cap(
                "navigate_b2_search_to_solar",
                (), (), ("b2_route",), "navigation", 5, 40,
                (
                    "system_ready()", "robot_ready(b2_base)", "docked(b2_base,tool_search_view)",
                    "stationary(b2_base)", "stowed(z1_thermal_arm)",
                    "thermal_reference_installed(energy_rig)",
                ),
                ("docked(b2_base,solar_view)", "stationary(b2_base)"),
                ("docked(b2_base)",),
            ),
        ),
        "z1_thermal_arm": (
            _cap(
                "deploy_camera_tool_search",
                (), (), ("tool_search_zone",), "manipulation", 2, 24,
                (
                    "system_ready()", "robot_ready(z1_thermal_arm)",
                    "docked(b2_base,tool_search_view)", "stationary(b2_base)", "stowed(z1_thermal_arm)",
                ),
                ("camera_deployed(z1_thermal_arm,tool_search_view)",),
                ("stowed(z1_thermal_arm)",),
            ),
            _cap(
                "localize_fallen_tool",
                ("tool",), ("tool",), ("tool_search_zone",), "verification", 3, 30,
                (
                    "system_ready()", "robot_ready(z1_thermal_arm)",
                    "camera_deployed(z1_thermal_arm,tool_search_view)", "fallen_tool_unlocalized(tool)",
                ),
                ("tool_localized(tool)", "at(tool,search_floor)"),
                ("fallen_tool_unlocalized(tool)",),
            ),
            _cap(
                "stow_z1_after_tool_search",
                (), (), ("tool_search_zone",), "manipulation", 2, 24,
                (
                    "system_ready()", "robot_ready(z1_thermal_arm)",
                    "camera_deployed(z1_thermal_arm,tool_search_view)", "tool_localized(inspection_kit)",
                ),
                ("stowed(z1_thermal_arm)",),
                ("camera_deployed(z1_thermal_arm,tool_search_view)",),
            ),
        ),
        "husky_base": (
            _cap(
                "navigate_husky_tool_recovery",
                (), (), ("husky_route",), "navigation", 5, 45,
                (
                    "system_ready()", "robot_ready(husky_base)", "docked(husky_base,husky_home)",
                    "stationary(husky_base)", "stowed(husky_franka)", "tool_localized(inspection_kit)",
                ),
                ("docked(husky_base,tool_recovery_dock)", "stationary(husky_base)"),
                ("docked(husky_base)",),
            ),
            _cap(
                "navigate_husky_recovery_to_reference",
                (), (), ("husky_route",), "navigation", 5, 45,
                (
                    "system_ready()", "robot_ready(husky_base)",
                    "docked(husky_base,tool_recovery_dock)", "stationary(husky_base)",
                    "stowed(husky_franka)", "kit_secured(husky_franka,inspection_kit)",
                ),
                ("docked(husky_base,reference_dock)", "stationary(husky_base)"),
                ("docked(husky_base)",),
            ),
        ),
        "husky_franka": (
            _cap(
                "recover_localized_tool",
                ("tool", "location"), ("tool", "location"), ("tool_recovery_zone",),
                "manipulation", 4, 60,
                (
                    "system_ready()", "robot_ready(husky_franka)",
                    "docked(husky_base,tool_recovery_dock)", "stationary(husky_base)",
                    "stowed(husky_franka)", "gripper_empty(husky_franka)",
                    "tool_localized(tool)", "at(tool,location)",
                ),
                (
                    "holding(husky_franka,tool)", "kit_secured(husky_franka,tool)",
                    "tool_recovered(tool)", "stowed(husky_franka)",
                ),
                ("at(tool)", "gripper_empty(husky_franka)"),
            ),
        ),
    }
    robots = tuple(
        replace(robot, capabilities=(*robot.capabilities, *additions.get(robot.id, ())))
        for robot in scenario.robots
    )
    return replace(
        scenario,
        instruction=(
            f"{scenario.instruction} Continue from the measured dropped-tool state without reset. "
            "Use B2/Z1 to localize the same inspection_kit, then Husky and its mounted Franka to "
            "recover it before completing the original inspection."
        ),
        initial_state=tuple(canonical_predicate(fact) for fact in measured_initial_state),
        entities=(
            *scenario.entities,
            Entity("tool_search_view", "dock"),
            Entity("tool_recovery_dock", "dock"),
            Entity("search_floor", "location"),
            Entity("tool_search_zone", "zone"),
            Entity("tool_recovery_zone", "zone"),
        ),
        resources=(
            *scenario.resources,
            Resource("tool_search_zone", 1),
            Resource("tool_recovery_zone", 1),
        ),
        robots=robots,
    )


def _cap(
    name: str,
    parameters: tuple[str, ...],
    parameter_types: tuple[str, ...],
    resources: tuple[str, ...],
    action_type: str,
    duration_ticks: int,
    timeout_ticks: int,
    preconditions: tuple[str, ...],
    adds: tuple[str, ...],
    deletes: tuple[str, ...],
) -> Capability:
    return Capability(
        name=name,
        parameters=parameters,
        parameter_types=parameter_types,
        resources=resources,
        action_type=action_type,
        duration_ticks=duration_ticks,
        timeout_ticks=timeout_ticks,
        preconditions=tuple(canonical_predicate(item) for item in preconditions),
        effects=Effects(
            add=tuple(canonical_predicate(item) for item in adds),
            delete=tuple(canonical_predicate(item) for item in deletes),
        ),
    )


def _semantic_errors(plan: Plan) -> list[str]:
    required = {
        ("b2_base", "navigate_b2_tool_search"),
        ("z1_thermal_arm", "localize_fallen_tool"),
        ("husky_base", "navigate_husky_tool_recovery"),
        ("husky_franka", "recover_localized_tool"),
    }
    actual = {
        (robot, node.name or "")
        for robot, root in plan.behavior_trees.items()
        for node in iter_nodes(root)
        if node.type == "Action"
    }
    errors = [f"Recovery BT omits required action {robot}/{action}." for robot, action in sorted(required - actual)]
    if any(node.type == "Action" and "spare" in " ".join(node.parameters) for root in plan.behavior_trees.values() for node in iter_nodes(root)):
        errors.append("Recovery BT invents a spare tool instead of recovering inspection_kit.")
    unsupported = sorted({node.type for root in plan.behavior_trees.values() for node in iter_nodes(root) if node.children and node.type not in {"Sequence", "Fallback"}})
    if unsupported:
        errors.append("Recovery BT uses unsupported composites: " + ", ".join(unsupported))
    return errors


def _progress(callback: Progress | None, message: str, fraction: float) -> None:
    if callback is not None:
        callback(message, max(0.0, min(1.0, fraction)))
