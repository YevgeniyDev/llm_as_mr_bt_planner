"""Prompt construction and JSON extraction for the LLM planner."""

from __future__ import annotations

import json
import re
from typing import Any

from .domain import Scenario, candidate_producers
from .predicates import parse_predicate, unify_effect_args
from .simulation import SimulationReport

SYSTEM_PROMPT = "You produce strict JSON plans for multi-robot behavior tree planning."

_SCHEMA = """Required output schema:
{
  "task_graph": [
    {"id": "t1", "action": "action_name", "parameters": ["arg"], "depends_on": []}
  ],
  "assignments": [
    {"task_id": "t1", "robot": "robot_id"}
  ],
  "synchronization": [
    {"condition": "predicate(arg)", "producer": "robot_id", "consumer": "robot_id"}
  ],
  "behavior_trees": {
    "robot_id": {
      "type": "Sequence",
      "children": [
        {"type": "Condition", "name": "predicate", "parameters": ["arg"]},
        {"type": "Action", "name": "action_name", "parameters": ["arg"]}
      ]
    }
  }
}
"""

_RULES = """Planning rules:
1. Use only the robot, object, location, action, and predicate names supplied above.
2. Treat capability preconditions/effects as authoritative when choosing action parameters.
3. Infer the task graph from goal_state by matching goals and missing preconditions to capability add-effects.
4. Include every action needed to make the goals true from initial_state; there is no hidden task list.
5. Assign each task graph action to exactly one robot that has that capability.
6. For every precondition not already true in initial_state, include an earlier action whose add-effects create it (a producer), assigned to a capable robot.
7. task_graph.action and BT Action.name must be bare names, never robot.action.
8. A Condition node only waits; it never creates a predicate or replaces a missing producer action.
9. The task_graph must be acyclic.
10. Keep BTs simple: Sequence roots with Action and Condition children (Fallback/Parallel are allowed but rarely needed).
11. Every assigned task action must appear as an Action node in that robot's BT with identical parameters.
12. Same action names on different robots are distinct tasks; include separate task nodes and BT Actions for each robot.
13. Robot-specific predicates are not interchangeable: holding(robot_a, x) does not satisfy holding(robot_b, x).
14. Use synchronization only for inter-robot waits: if robot B needs a predicate produced by robot A, add it to synchronization and put the exact Condition before B consumes it.
15. Do not add synchronization entries for same-robot sequencing or predicates already true in initial_state.
16. A producer BT must execute an action whose add-effects create each synchronization condition.
17. Match predicate arguments exactly. If a later precondition needs object_at(x, location_a), the producer action must use location_a, not a nearby or generic location.
18. Before a robot waits for a downstream completion condition, it must first produce any resources that downstream robots need from it.
19. Do not add a Condition for a predicate that no robot can produce and that is not already in initial_state.
20. Follow the scenario instruction, but derive executable steps from capability preconditions/effects.
21. Never omit predicate arguments in Condition nodes or synchronization.condition. Use the full predicate from preconditions/effects/goals.
22. In BT nodes, 'name' is only the bare action or predicate name.
   Correct: {"type":"Condition","name":"drawer_closed","parameters":["parts_drawer"]}
   Wrong:   {"type":"Condition","name":"drawer_closed(parts_drawer)"}
23. Every Action and Condition node must include a parameters array, even if empty.
24. Completeness beats compactness: never omit producer actions needed for preconditions.
"""

_METHOD = """Planning method (reason through this yourself; output only JSON):
1. Start from goal_state. For each goal not already in initial_state, choose a capability whose add-effects
   create it; add that action and assign it to a robot that has the capability.
2. For every action you include, check each of its preconditions. A precondition must be one of:
   (a) already in initial_state, (b) created earlier by an add-effect in the SAME robot's behavior tree, or
   (c) produced by ANOTHER robot and waited on with a synchronization Condition placed before it.
3. If a precondition is none of these, add the action that produces it (plus its task node and assignment),
   then recurse on that new action's preconditions. Continue until every precondition bottoms out at
   initial_state. Do not stop early.
4. Robot-scoped predicates such as holding(robot, object) are created ONLY by that same robot's own actions.
   If a robot must hold a tool or part, that robot must itself execute the grasp/pick action that adds
   holding(robot, ...). Another robot placing the object nearby does NOT make this robot hold it - you must
   add this robot's own pick action.
5. Before answering, re-check every BT Action's preconditions once more. A missing producer action is the most
   common error.

Illustrative pattern (use the scenario's real names, not these placeholders):
- goal needs assembled(part); the capability assemble(part) requires holding(armB, part).
- holding(armB, part) is added only by armB's own pick(part).
- therefore armB's behavior tree must contain pick(part) before assemble(part), and both are armB tasks.
"""

_HINTS_INSTRUCTION = (
    "Capability dependency hints (precomputed producer candidates) are provided above. "
    "They are optional aids derived from preconditions/effects; use them to place producer actions, "
    "but the scenario data remains authoritative.\n"
)


def _scenario_context(scenario: Scenario, include_hints: bool) -> str:
    robot_summary = [
        {
            "id": robot.id,
            "type": robot.type,
            "capabilities": [
                {
                    "name": cap.name,
                    "parameters": list(cap.parameters),
                    "preconditions": list(cap.preconditions),
                    "effects": {"add": list(cap.effects.add), "delete": list(cap.effects.delete)},
                }
                for cap in robot.capabilities
            ],
        }
        for robot in scenario.robots
    ]
    task_summary = {
        "task_id": scenario.task_id,
        "instruction": scenario.instruction,
        "initial_state": list(scenario.initial_state),
        "goal_state": list(scenario.goal_state),
        "objects": list(scenario.objects),
        "locations": list(scenario.locations),
    }
    parts = [
        f"Scenario:\n{json.dumps(task_summary, indent=2)}\n",
        f"Robot capability library:\n{json.dumps(robot_summary, indent=2)}\n",
    ]
    if include_hints:
        parts.append(
            "Capability dependency hints derived from preconditions/effects:\n"
            f"{json.dumps(build_dependency_hints(scenario), indent=2)}\n"
        )
    return "\n".join(parts)


def build_prompt(scenario: Scenario, include_hints: bool = False) -> str:
    parts = [
        "Generate a compact symbolic multi-robot behavior-tree plan for the scenario below.\n"
        "Return ONLY valid JSON. Do not use markdown or explanatory text.\n",
        _scenario_context(scenario, include_hints),
        _SCHEMA,
        _RULES,
        _METHOD,
    ]
    if include_hints:
        parts.append(_HINTS_INSTRUCTION)
    return "\n".join(parts)


def build_correction_prompt(
    scenario: Scenario,
    validation_errors: list[dict[str, str]],
    simulation: SimulationReport,
    previous_plan: dict[str, Any] | None = None,
    include_hints: bool = False,
) -> str:
    previous_block = ""
    if previous_plan is not None:
        previous_block = f"Previous plan that failed (start from this and fix it):\n{json.dumps(previous_plan, indent=2)}\n\n"
    return (
        f"{build_prompt(scenario, include_hints=include_hints)}\n\n"
        f"{previous_block}"
        "The previous JSON plan failed validation or simulation. Return a COMPLETE corrected plan.\n"
        "Keep the parts of the previous plan that are already correct; change only what the errors below require, "
        "adding the minimum necessary nodes. Do not drop actions or synchronization that were already correct.\n\n"
        f"Validator errors:\n{json.dumps(validation_errors, indent=2)}\n\n"
        "Simulation result:\n"
        f"{json.dumps(_compact_simulation(simulation), indent=2)}\n\n"
        "How to fix each error type:\n"
        "- unsupported_precondition / unsupported_goal: the named predicate is never produced. Add the action "
        "whose add-effects create it, plus its task node and assignment, and place it before the consumer. If the "
        "predicate is robot-scoped (e.g. holding(R, x)), the producing action must be R's OWN action.\n"
        "- unsupported_condition: either add a producer action for it, or delete the Condition if it is not needed.\n"
        "- missing_bt_action: add the assigned action (with exact parameters) to that robot's behavior tree.\n"
        "- unassigned_bt_action: add a matching task node + assignment, or remove the stray BT action.\n"
        "- missing_sync_condition / missing_sync_producer: ensure the producer robot's BT runs an action that "
        "adds the condition, and the consumer robot's BT has the exact Condition before it consumes it.\n"
        "- condition_before_producer: move the Condition after the action that creates it, or remove it.\n"
        "- cyclic_dependency / deadlock: reorder so producers run before consumers; never have two robots wait on "
        "each other.\n"
        "- Every BT Action/Condition node must have a bare name and a parameters array.\n"
        "- Return only the complete corrected JSON object.\n"
    )


def _compact_simulation(simulation: SimulationReport) -> dict[str, Any]:
    return {
        "success": simulation.success,
        "goal_success": simulation.goal_success,
        "errors": simulation.errors,
        "final_state": simulation.final_state,
        "trace_tail": simulation.trace[-8:],
    }


_ACTION_PLAN_SCHEMA = """Required output schema (action plan only - NO behavior trees, conditions, or synchronization yet):
{
  "action_plan": {
    "robot_id": [
      {"action": "action_name", "parameters": ["arg"]},
      {"action": "action_name", "parameters": ["arg"]}
    ]
  }
}
"""


def build_action_plan_prompt(scenario: Scenario, include_hints: bool = False) -> str:
    """Stage 1: ask only for the ordered list of actions each robot performs."""
    parts = [
        "Plan a multi-robot task. In this FIRST stage, output only the ordered list of actions each robot\n"
        "performs - no behavior-tree structure, no Condition nodes, no synchronization yet.\n"
        "Return ONLY valid JSON. Do not use markdown or explanatory text.\n",
        _scenario_context(scenario, include_hints),
        _ACTION_PLAN_SCHEMA,
        "Rules:\n"
        "1. Use only the action and object/location names from the capability library.\n"
        "2. Include every action needed to make all goal_state predicates true from initial_state.\n"
        "3. Order each robot's actions so that, by the time an action runs, its preconditions are either in\n"
        "   initial_state or have been produced earlier by some robot's action.\n"
        "4. Each action's parameters must match the capability and the predicates exactly.\n",
        _METHOD,
    ]
    if include_hints:
        parts.append(_HINTS_INSTRUCTION)
    return "\n".join(parts)


def build_action_plan_correction_prompt(
    scenario: Scenario,
    validation_errors: list[dict[str, str]],
    simulation: SimulationReport,
    previous_action_plan: dict[str, Any],
    include_hints: bool = False,
) -> str:
    return (
        f"{build_action_plan_prompt(scenario, include_hints=include_hints)}\n\n"
        f"Previous action plan that failed (start from this and fix it):\n"
        f"{json.dumps({'action_plan': previous_action_plan}, indent=2)}\n\n"
        "The action plan was checked by running each robot's actions in order (an action waits until its\n"
        "preconditions hold). It failed. Add the missing producer actions and/or fix the ordering so every\n"
        "action's preconditions become satisfiable and all goals are reached. Remember: robot-scoped predicates\n"
        "like holding(R, x) are produced only by R's own action.\n\n"
        f"Validator errors:\n{json.dumps(validation_errors, indent=2)}\n\n"
        f"Simulation result:\n{json.dumps(_compact_simulation(simulation), indent=2)}\n\n"
        "Return only the corrected action-plan JSON object."
    )


def build_bt_encoding_prompt(
    scenario: Scenario,
    action_plan: dict[str, Any],
    include_hints: bool = False,
) -> str:
    """Stage 2: wrap a fixed, feasible action plan into behavior trees + synchronization."""
    return (
        "Encode the given multi-robot action plan as behavior trees with explicit synchronization.\n"
        "Return ONLY valid JSON. Do not use markdown or explanatory text.\n\n"
        f"{_scenario_context(scenario, include_hints)}\n"
        f"Fixed action plan (use these exact actions, per robot, in this order):\n"
        f"{json.dumps({'action_plan': action_plan}, indent=2)}\n\n"
        f"{_SCHEMA}\n"
        "Encoding rules:\n"
        "1. Each robot's behavior_tree is a Sequence containing exactly its actions from the action plan, in the\n"
        "   same order, with identical parameters. Do NOT add, remove, or reorder actions.\n"
        "2. Build task_graph and assignments so every action is one task assigned to its robot.\n"
        "3. For any action precondition that is produced by ANOTHER robot (not already in initial_state and not\n"
        "   produced earlier by this same robot), insert a Condition node for that exact predicate immediately\n"
        "   before the action, and add a synchronization entry {condition, producer, consumer}.\n"
        "4. Do not add Conditions for predicates that are already true in initial_state or produced earlier by\n"
        "   the same robot.\n"
        "5. BT node 'name' is the bare action/predicate name; parameters go in the parameters array.\n"
        "6. Return only the complete plan JSON object."
    )


def build_bt_encoding_correction_prompt(
    scenario: Scenario,
    validation_errors: list[dict[str, str]],
    simulation: SimulationReport,
    previous_plan: dict[str, Any],
    action_plan: dict[str, Any],
    include_hints: bool = False,
) -> str:
    return (
        f"{build_bt_encoding_prompt(scenario, action_plan, include_hints=include_hints)}\n\n"
        f"Previous plan that failed (fix it; keep the same actions per robot):\n{json.dumps(previous_plan, indent=2)}\n\n"
        "Keep each robot's actions exactly as in the action plan. Fix only the behavior-tree structure,\n"
        "Condition placement, synchronization, task_graph, and assignments per the errors below.\n\n"
        f"Validator errors:\n{json.dumps(validation_errors, indent=2)}\n\n"
        f"Simulation result:\n{json.dumps(_compact_simulation(simulation), indent=2)}\n\n"
        "Return only the complete corrected plan JSON object."
    )


def build_dependency_hints(scenario: Scenario) -> list[dict[str, Any]]:
    initial_state = set(scenario.initial_state)
    constants = scenario.constants
    hints: list[dict[str, Any]] = []
    for robot in scenario.robots:
        for cap in robot.capabilities:
            consumer = {"robot": robot.id, "action": cap.name, "parameters": list(cap.parameters)}
            for precondition in cap.preconditions:
                if _template_satisfied_by_initial(precondition, initial_state, cap.parameters, constants):
                    continue
                producers = candidate_producers(precondition, scenario)
                if producers:
                    hints.append(
                        {
                            "consumer": consumer,
                            "needs": precondition,
                            "candidate_producers": [
                                {"robot": p.robot, "action": p.action, "parameters": list(p.parameters)}
                                for p in producers
                            ],
                        }
                    )
    return hints


def _template_satisfied_by_initial(
    predicate: str,
    initial_state: set[str],
    action_parameters: tuple[str, ...],
    constants: set[str],
) -> bool:
    name, args = parse_predicate(predicate)
    for fact in initial_state:
        fact_name, fact_args = parse_predicate(fact)
        if fact_name == name and unify_effect_args(args, fact_args, action_parameters, constants) is not None:
            return True
    return False


# --------------------------------------------------------------------------- #
# JSON extraction
# --------------------------------------------------------------------------- #


def extract_json(text: str) -> dict[str, Any]:
    cleaned = _strip_fence(text.strip())
    candidate = _first_json_object(cleaned)
    if candidate is None:
        raise ValueError("Could not find a JSON object in the LLM response.")
    return json.loads(candidate)


def _strip_fence(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
