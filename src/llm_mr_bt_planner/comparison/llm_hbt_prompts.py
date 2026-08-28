"""Clean-room prompts for the published LLM-HBT decision boundaries.

The paper specifies the information passed between the task initializer,
virtual allocator (Alex), and selected robot, but it does not publish prompt
text or a response grammar.  These prompts therefore use a strict JSON
grammar that is recorded as a reproduction choice in every run manifest.
"""

from __future__ import annotations

import json
from typing import Any

from ..domain import Scenario
from .llm_bt_native import GroundAction

INITIALIZATION_SYSTEM_PROMPT = """You are the task-initialization module of LLM-HBT.
Translate the human instruction into an ordered list of condition nodes selected only
from the supplied condition-node library. Return one JSON object and no prose:
{"conditions":["predicate(arguments)", ...]}
The order must support executable progress and preserve required final conditions.
Do not invent conditions, actions, robots, objects, or locations."""

ALEX_SYSTEM_PROMPT = """You are Alex, the centralized virtual allocator in LLM-HBT.
Given one failed condition, observations, and heterogeneous robot action libraries, assign
exactly one capable robot. Return one JSON object and no prose:
{"robot":"robot_id","mode":"local|delegated","task":"short assignment"}
Use local only when the requesting robot can resolve the condition; otherwise delegate.
Do not select an action in this stage and do not invent identifiers."""

ROBOT_SYSTEM_PROMPT = """You are the selected robot's LLM-HBT action selector.
Choose exactly one supplied grounded action whose postconditions establish the failed
condition. Return one JSON object and no prose:
{"action":"capability(parameter_1,parameter_2)"}
Do not create a multi-action plan, alter parameters, or invent capabilities."""


def build_initialization_prompt(
    scenario: Scenario,
    conditions: list[str],
) -> str:
    return "\n".join(
        [
            "Human instruction:",
            scenario.instruction,
            "",
            "Observed initial state:",
            _lines(scenario.initial_state),
            "",
            "Pre-defined condition-node library:",
            _lines(conditions),
            "",
            "Construct the initial ordered condition tree using only that library.",
        ]
    )


def build_assignment_prompt(
    scenario: Scenario,
    actions: list[GroundAction],
    *,
    failed_condition: str,
    requester: str | None,
    observed_state: set[str],
    failure_observation: dict[str, Any] | None,
) -> str:
    robot_sections: list[str] = []
    for robot in scenario.robots:
        available = [action for action in actions if action.robot == robot.id]
        rendered = "\n".join(f"  - {_action_line(action)}" for action in available)
        robot_sections.append(
            f"Robot {robot.id} ({robot.type}):\n{rendered or '  - no grounded actions'}"
        )
    lines = [
        f"Failed condition node: {failed_condition}",
        f"Requesting robot: {requester or 'unassigned initial condition'}",
        "",
        "Shared observed state:",
        _lines(sorted(observed_state)),
        "",
    ]
    if failure_observation is not None:
        lines.extend(
            [
                "Post-failure observation (available only after runtime failure):",
                json.dumps(failure_observation, sort_keys=True),
                "",
            ]
        )
    lines.extend(["Robot action libraries:", "\n\n".join(robot_sections)])
    return "\n".join(lines)


def build_action_prompt(
    scenario: Scenario,
    actions: list[GroundAction],
    *,
    failed_condition: str,
    selected_robot: str,
    task: str,
    observed_state: set[str],
    failure_observation: dict[str, Any] | None,
) -> str:
    available = [action for action in actions if action.robot == selected_robot]
    lines = [
        f"Selected robot: {selected_robot}",
        f"Assigned task: {task}",
        f"Condition that must be established: {failed_condition}",
        "",
        "Robot's current observation:",
        _lines(sorted(observed_state)),
        "",
    ]
    if failure_observation is not None:
        lines.extend(
            [
                "Post-failure observation:",
                json.dumps(failure_observation, sort_keys=True),
                "",
            ]
        )
    lines.extend(
        [
            f"Grounded action nodes for {selected_robot}:",
            "\n".join(f"- {_action_line(action)}" for action in available),
            "",
            "Choose one action whose listed postconditions establish the failed condition.",
        ]
    )
    return "\n".join(lines)


def _action_line(action: GroundAction) -> str:
    expression = f"{action.name}({','.join(action.parameters)})"
    return (
        f"{expression}; pre={list(action.preconditions)}; "
        f"post_add={list(action.add_effects)}; post_delete={list(action.delete_effects)}"
    )


def _lines(values) -> str:
    materialized = list(values)
    return "\n".join(f"- {value}" for value in materialized) or "- none"
