"""KIOS-style prompts adapted only at the common-domain input boundary."""

from __future__ import annotations

import json
from typing import Any

from ..domain import Scenario, scenario_to_dict

SYSTEM_PROMPT = (
    "You are a behavior-tree assembly planner. Use only the supplied predicates, "
    "objects, robots, actions, transition models, and KIOS JSON grammar. Return JSON only."
)


def build_decomposition_prompt(scenario: Scenario) -> str:
    return _join(
        "ASSEMBLY PLANNING: decompose the instruction into a sequential list of grounded subgoals.",
        _problem(scenario),
        {
            "output_schema": {
                "explanation": "short string",
                "subgoals": [
                    {
                        "id": "unique short id",
                        "robot": "exact robot id",
                        "target": "grounded predicate(...) made true by this subgoal",
                        "instruction": "short natural-language instruction",
                    }
                ],
            },
            "rules": [
                "subgoals are executed in listed order",
                "assign each subgoal to a robot owning every action needed for it",
                "cover every goal and necessary intermediate transition",
                "do not mention a failure, validator, or recovery",
            ],
        },
    )


def build_one_step_prompt(
    scenario: Scenario, subgoal: dict[str, str], world_state: list[str]
) -> str:
    return _join(
        "ONE-STEP BT GENERATION: generate the complete KIOS behavior tree in one response.",
        _subproblem(scenario, subgoal, world_state),
        _kios_contract(),
        {
            "output_schema": {
                "thought": "short string",
                "action_sequence": ["capability(robot_id, arg, ...)"],
                "behavior_tree": {"summary": "...", "name": "selector: ...", "children": []},
            }
        },
    )


def build_iterative_prompt(
    scenario: Scenario,
    subgoal: dict[str, str],
    world_state: list[str],
    previous_tree: dict[str, Any] | None,
    execution_result: dict[str, Any] | None,
) -> str:
    return _join(
        "ITERATIVE BT GENERATION: generate a whole KIOS tree, using native execution feedback "
        "from the preceding attempt when supplied.",
        _subproblem(scenario, subgoal, world_state),
        _kios_contract(),
        {"last_behavior_tree": previous_tree, "last_execution_result": execution_result},
        {
            "output_schema": {
                "thought": "short string",
                "action_sequence": ["capability(robot_id, arg, ...)"],
                "behavior_tree": {"summary": "...", "name": "selector: ...", "children": []},
            }
        },
    )


def build_sequential_plan_prompt(
    scenario: Scenario, subgoal: dict[str, str], world_state: list[str]
) -> str:
    return _join(
        "HUMAN-IN-THE-LOOP ACTION PLANNING: propose an ordered grounded action plan.",
        _subproblem(scenario, subgoal, world_state),
        {
            "output_schema": {
                "explanation": "short string",
                "task_plan": ["capability(robot_id, arg, ...)"],
            }
        },
    )


def build_human_tree_prompt(
    scenario: Scenario,
    subgoal: dict[str, str],
    world_state: list[str],
    task_plan: list[str],
    *,
    previous_tree: dict[str, Any] | None = None,
    human_feedback: str | None = None,
) -> str:
    return _join(
        "HUMAN-IN-THE-LOOP BT GENERATION: encode the action plan as one complete KIOS tree.",
        _subproblem(scenario, subgoal, world_state),
        _kios_contract(),
        {
            "task_plan": task_plan,
            "previous_behavior_tree": previous_tree,
            "human_feedback": human_feedback,
            "output_schema": {"summary": "...", "name": "selector: ...", "children": []},
        },
    )


def build_make_plan_prompt(scenario: Scenario, robot: str, goal: str, state: list[str]) -> str:
    return _join(
        "RECURSIVE MakePlan: find an action sequence whose final action establishes the target.",
        _robot_domain(scenario, robot),
        {"start_world_state": state, "target": goal},
        {
            "output_schema": {
                "explanation": "short string",
                "task_plan": ["capability(robot_id, arg, ...)"],
            },
            "rules": ["return an empty task_plan when the target is already true"],
        },
    )


def build_make_tree_prompt(scenario: Scenario, robot: str, action: str) -> str:
    return _join(
        "RECURSIVE MakeTree: generate exactly one KIOS unit subtree for the supplied action.",
        _robot_domain(scenario, robot),
        _kios_contract(),
        {"action": action, "output_schema": {"summary": "...", "name": "selector: ..."}},
    )


def build_predict_state_prompt(
    scenario: Scenario, robot: str, state: list[str], action_plan: list[str]
) -> str:
    return _join(
        "RECURSIVE PredictState: apply the declared transition models in order.",
        _robot_domain(scenario, robot),
        {"start_world_state": state, "action_plan": action_plan},
        {"output_schema": {"explanation": "short string", "estimated_world_state": ["fact()"]}},
    )


def _problem(scenario: Scenario) -> dict[str, Any]:
    document = scenario_to_dict(scenario)
    return {
        "instruction": scenario.instruction,
        "objects_and_properties": document["entities"],
        "constraints": document["resources"],
        "relations_initially_true": list(scenario.initial_state),
        "target_relations": list(scenario.goal_state),
        "robots_and_actions": document["robots"],
    }


def _subproblem(
    scenario: Scenario, subgoal: dict[str, str], world_state: list[str]
) -> dict[str, Any]:
    return {
        "subgoal": subgoal,
        "initial_world_state": world_state,
        "robot_domain": _robot_domain(scenario, subgoal["robot"]),
    }


def _robot_domain(scenario: Scenario, robot_id: str) -> dict[str, Any]:
    robot = scenario.robot(robot_id)
    actions: list[dict[str, Any]] = []
    if robot is not None:
        for capability in robot.capabilities:
            arguments = ", ".join([robot_id, *capability.parameters])
            actions.append(
                {
                    "signature": f"{capability.name}({arguments})",
                    "parameter_types": list(capability.parameter_types),
                    "preconditions": list(capability.preconditions),
                    "effects_add": list(capability.effects.add),
                    "effects_delete": list(capability.effects.delete),
                    "resources": list(capability.resources),
                }
            )
    return {
        "robot": robot_id,
        "actions": actions,
        "constants": sorted(scenario.constants),
    }


def _kios_contract() -> dict[str, Any]:
    return {
        "kios_json_grammar": {
            "all_nodes": ["summary", "name"],
            "composites": ["selector: text", "sequence: text", "parallel: text"],
            "leaves": [
                "target: grounded_predicate(...) ",
                "precondition: grounded_predicate(...)",
                "condition: grounded_predicate(...)",
                "action: capability(exact_robot_id, arg, ...)",
            ],
            "unit_subtree": [
                "selector root",
                "target leaf as first child",
                "sequence as second child",
                "sequence contains every action precondition then the action",
            ],
        },
        "rules": [
            "selector and sequence use memoryless KIOS semantics",
            "use only exact declared grounded predicates, actions, robot ids, and constants",
            "every composite has a non-empty children array; leaves have no children",
            "do not output canonical BT node types or resource lock nodes",
        ],
    }


def _join(*parts: object) -> str:
    return "\n\n".join(
        part if isinstance(part, str) else json.dumps(part, indent=2, sort_keys=True)
        for part in parts
    )
