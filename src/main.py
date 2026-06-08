from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    scenario_path = resolve_project_path(args.scenario)
    output_path = resolve_project_path(args.output)
    scenario = load_json(scenario_path)

    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if args.trials == 1:
        result = run_trial(
            scenario=scenario,
            model=model,
            max_corrections=args.max_corrections,
            max_ticks=args.max_ticks,
            trial_index=1,
        )
    else:
        result = run_trials(
            scenario=scenario,
            model=model,
            max_corrections=args.max_corrections,
            max_ticks=args.max_ticks,
            trials=args.trials,
        )

    save_json(output_path, result)
    print_summary(result, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and simulate multi-robot behavior trees with an LLM.")
    parser.add_argument("--scenario", default="data/scenario.json", help="Scenario JSON file.")
    parser.add_argument("--output", default="outputs/run.json", help="Single JSON result file.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL for this run.")
    parser.add_argument("--max-ticks", type=int, default=80, help="Maximum symbolic simulator ticks.")
    parser.add_argument("--max-corrections", type=int, default=4, help="LLM self-correction rounds after validation or simulation errors.")
    parser.add_argument("--trials", type=int, default=1, help="Run repeated independent LLM attempts for this scenario.")
    args = parser.parse_args()
    if args.max_ticks < 1:
        parser.error("--max-ticks must be at least 1")
    if args.max_corrections < 0:
        parser.error("--max-corrections cannot be negative")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    return args


def run_trial(
    scenario: dict[str, Any],
    model: str,
    max_corrections: int,
    max_ticks: int,
    trial_index: int,
) -> dict[str, Any]:
    plan, validation, simulation, attempts = generate_evaluated_plan(
        scenario=scenario,
        model=model,
        max_corrections=max_corrections,
        max_ticks=max_ticks,
    )
    return {
        "task_id": scenario.get("task_id"),
        "model": model,
        "trial": trial_index,
        "valid": validation["valid"],
        "success": simulation["success"],
        "goal_success": simulation["goal_success"],
        "correction_rounds": len(attempts) - 1,
        "attempts": attempts,
        "plan": plan,
        "validation_errors": validation["errors"],
        "simulation": {
            "final_state": simulation["final_state"],
            "trace": simulation["trace"],
            "errors": simulation["errors"],
        },
    }


def run_trials(
    scenario: dict[str, Any],
    model: str,
    max_corrections: int,
    max_ticks: int,
    trials: int,
) -> dict[str, Any]:
    results = [
        run_trial(
            scenario=scenario,
            model=model,
            max_corrections=max_corrections,
            max_ticks=max_ticks,
            trial_index=index,
        )
        for index in range(1, trials + 1)
    ]
    return {
        "task_id": scenario.get("task_id"),
        "model": model,
        "trials": trials,
        "success_count": sum(1 for item in results if item["success"]),
        "valid_count": sum(1 for item in results if item["valid"]),
        "goal_success_count": sum(1 for item in results if item["goal_success"]),
        "results": results,
    }


def generate_evaluated_plan(
    scenario: dict[str, Any],
    model: str,
    max_corrections: int,
    max_ticks: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = query_plan(prompt=build_prompt(scenario), model=model)
    attempts = []

    for round_index in range(0, max_corrections + 1):
        plan.setdefault("task_id", scenario.get("task_id"))
        validation = validate_plan(plan, scenario)
        simulation = (
            simulate_plan(plan, scenario, max_ticks=max_ticks)
            if validation["valid"]
            else skipped_simulation()
        )
        attempts.append(attempt_summary(round_index=round_index, validation=validation, simulation=simulation))

        if validation["valid"] and simulation["success"]:
            break
        if round_index == max_corrections:
            break

        plan = query_plan(
            prompt=build_correction_prompt(scenario, plan, validation["errors"], simulation),
            model=model,
        )

    return plan, validation, simulation, attempts


def query_plan(prompt: str, model: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add a key.")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You produce strict JSON plans for multi-robot behavior tree planning.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    request = urllib.request.Request(
        openai_chat_completions_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    timeout = float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed: HTTP {error.code}: {safe_api_error(detail)}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"LLM request failed: {safe_api_error(str(error.reason))}") from error

    api_result = json.loads(response_body)
    content = api_result["choices"][0]["message"]["content"]
    plan = extract_json(content)
    plan.setdefault("_llm", {"model": model})
    return plan


def attempt_summary(
    round_index: int,
    validation: dict[str, Any],
    simulation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "round": round_index,
        "valid": validation["valid"],
        "success": simulation["success"],
        "goal_success": simulation["goal_success"],
        "num_validation_errors": len(validation["errors"]),
        "num_simulation_errors": len(simulation["errors"]),
    }


def build_prompt(scenario: dict[str, Any]) -> str:
    robot_summary = [
        {
            "id": robot["id"],
            "type": robot.get("type"),
            "capabilities": [
                {
                    "name": capability["name"],
                    "parameters": capability.get("parameters", []),
                    "preconditions": capability.get("preconditions", []),
                    "effects": capability.get("effects", []),
                }
                for capability in robot.get("capabilities", [])
            ],
        }
        for robot in scenario.get("robots", [])
    ]

    required_tasks = scenario.get("required_tasks", [])
    required_actions = scenario_required_actions(scenario)
    task_summary = {
        "task_id": scenario.get("task_id"),
        "instruction": scenario.get("instruction"),
        "initial_state": scenario.get("initial_state", []),
        "goal_state": scenario.get("goal_state", []),
        "objects": scenario.get("objects", []),
        "locations": scenario.get("locations", []),
        "handoffs": scenario.get("handoffs", []),
        "constraints": scenario.get("constraints", []),
        "required_tasks": required_tasks,
        "required_actions": required_actions,
    }
    handoff_condition_nodes = [
        {
            "condition": handoff.get("condition"),
            "consumer": handoff.get("consumer"),
            "consumer_bt_condition_node": condition_node_from_predicate(handoff.get("condition")),
            "must_appear_before": handoff.get("before"),
        }
        for handoff in scenario.get("handoffs", [])
    ]

    return (
        "Generate a compact symbolic multi-robot behavior-tree plan for the scenario below.\n"
        "Return ONLY valid JSON. Do not use markdown or explanatory text.\n\n"
        f"Scenario:\n{json.dumps(task_summary, indent=2)}\n\n"
        f"Robot capability library:\n{json.dumps(robot_summary, indent=2)}\n\n"
        "Required task/action coverage checklist:\n"
        f"{json.dumps(required_tasks or required_actions, indent=2)}\n\n"
        "Required handoff condition nodes:\n"
        f"{json.dumps(handoff_condition_nodes, indent=2)}\n\n"
        "Required output schema:\n"
        "{\n"
        '  "task_graph": [\n'
        '    {"id": "t1", "action": "action_name", "parameters": ["arg"], "depends_on": []}\n'
        "  ],\n"
        '  "assignments": [\n'
        '    {"task_id": "t1", "robot": "robot_id"}\n'
        "  ],\n"
        '  "synchronization": [\n'
        '    {"condition": "predicate(arg)", "producer": "robot_id", "consumer": "robot_id"}\n'
        "  ],\n"
        '  "behavior_trees": {\n'
        '    "robot_id": {\n'
        '      "type": "Sequence",\n'
        '      "children": [\n'
        '        {"type": "Condition", "name": "predicate", "parameters": ["arg"]},\n'
        '        {"type": "Action", "name": "action_name", "parameters": ["arg"]}\n'
        "      ]\n"
        "    }\n"
        "  }\n"
        "}\n\n"
        "Planning rules:\n"
        "1. Use only the robot, object, location, action, and predicate names supplied above.\n"
        "2. Assign each task graph action to exactly one robot with that capability.\n"
        "3. The task_graph must be acyclic: no task may depend on a later task that depends back on it.\n"
        "4. Add every inter-robot handoff from Scenario.handoffs to synchronization.\n"
        "5. Copy each handoff condition string exactly into synchronization; do not shorten or omit arguments.\n"
        "6. If a handoff has producer_action, include that exact action in the producer task graph, assignment, and BT.\n"
        "7. In the consumer robot BT, place the exact Required handoff condition node before the consuming Action.\n"
        "8. Follow every Scenario.constraints item exactly.\n"
        "9. Every Scenario.required_tasks item must appear exactly in task_graph, assignments, and that robot's BT.\n"
        "10. If there is no required_tasks list, every Scenario.required_actions item must appear in task_graph, assignments, and that robot's BT.\n"
        "11. If required_tasks or required_actions repeats the same action and parameters for different robots, create one distinct task per robot.\n"
        "12. Keep BTs simple: Sequence roots with Action and Condition children only.\n"
        "13. Every assigned task action must appear as an Action node in that robot's BT.\n"
        "14. Do not truncate BTs: include all steps needed to reach every goal_state predicate.\n"
        "15. A producer BT must execute an action whose effects create each handoff condition before any robot waits on it.\n"
        "16. Avoid cyclic waits. Example: if franka2 waits for tool_at(screwdriver, tool_zone), "
        "go2_z1 must place_tool(screwdriver, tool_zone) before waiting for screw_fastened(gearbase).\n"
        "17. In BT nodes, 'name' is only the bare action or predicate name.\n"
        "   Correct: {\"type\":\"Condition\",\"name\":\"drawer_closed\",\"parameters\":[\"parts_drawer\"]}\n"
        "   Wrong:   {\"type\":\"Condition\",\"name\":\"drawer_closed(parts_drawer)\"}\n"
        "18. Every Action and Condition node must include a parameters array, even if empty.\n"
        "19. Handoff Condition parameters must exactly match the handoff predicate arguments.\n"
        "20. Prefer the smallest plan that reaches all goal_state predicates.\n"
    )


def scenario_required_actions(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    if scenario.get("required_actions"):
        return scenario["required_actions"]
    return [
        {
            "robot": task.get("robot"),
            "action": task.get("action"),
            "parameters": task.get("parameters", []),
        }
        for task in scenario.get("required_tasks", [])
    ]


def condition_node_from_predicate(predicate: str | None) -> dict[str, Any]:
    name, parameters = parse_predicate(predicate)
    return {"type": "Condition", "name": name, "parameters": parameters}


def build_correction_prompt(
    scenario: dict[str, Any],
    invalid_plan: dict[str, Any],
    validation_errors: list[dict[str, str]],
    simulation_result: dict[str, Any],
) -> str:
    return (
        f"{build_prompt(scenario)}\n\n"
        "The previous JSON plan failed validation or simulation. Return a complete corrected plan, not a patch.\n"
        "Fix every issue below while preserving the same scenario and output schema.\n\n"
        f"Validator errors:\n{json.dumps(validation_errors, indent=2)}\n\n"
        "Simulation result:\n"
        f"{json.dumps(compact_simulation_feedback(simulation_result), indent=2)}\n\n"
        f"Invalid plan:\n{json.dumps(invalid_plan, indent=2)}\n\n"
        "Correction requirements:\n"
        "- Every BT Action/Condition node must have a bare name and a parameters array.\n"
        "- If Scenario.required_tasks is present, copy those task ids/actions/parameters/dependencies into task_graph and assignments.\n"
        "- For each assignment, the assigned action with exact parameters must appear in that robot's BT.\n"
        "- If the same action name appears for multiple robots, add separate task_graph nodes and assignments for each robot.\n"
        "- Consumer BTs must wait for handoff conditions before the consuming actions.\n"
        "- Handoff Condition nodes must use the exact parameter list from the handoff condition.\n"
        "- Producer BTs must execute actions that create handoff conditions before consumers wait for them.\n"
        "- Task graph dependencies must be acyclic and point from consumers to already-required producer tasks.\n"
        "- Avoid cyclic waits such as robot A waiting for robot B before A produces a condition that B needs.\n"
        "- Return only the complete corrected JSON object.\n"
    )


def compact_simulation_feedback(simulation_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": simulation_result.get("success"),
        "goal_success": simulation_result.get("goal_success"),
        "errors": simulation_result.get("errors", []),
        "final_state": simulation_result.get("final_state", []),
        "trace_tail": simulation_result.get("trace", [])[-8:],
    }


def validate_plan(plan: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    robot_capabilities = {
        robot["id"]: {capability["name"] for capability in robot.get("capabilities", [])}
        for robot in scenario.get("robots", [])
    }

    for field in ["task_graph", "assignments", "synchronization", "behavior_trees"]:
        if field not in plan:
            add_error(errors, "missing_field", f"Plan is missing '{field}'.")
    if errors:
        return {"valid": False, "errors": errors}

    tasks = validate_task_graph(plan.get("task_graph", []), errors)
    assignments = validate_assignments(plan.get("assignments", []), tasks, robot_capabilities, errors)
    validate_behavior_trees(plan.get("behavior_trees", {}), robot_capabilities, errors)
    validate_assigned_actions_are_in_trees(plan.get("behavior_trees", {}), tasks, assignments, errors)
    validate_bt_actions_are_assigned(plan.get("behavior_trees", {}), tasks, assignments, errors)
    validate_required_tasks(plan.get("behavior_trees", {}), tasks, assignments, scenario, errors)
    validate_required_actions(plan.get("behavior_trees", {}), tasks, assignments, scenario, errors)
    validate_handoffs(plan, scenario, errors)

    return {"valid": not errors, "errors": errors}


def validate_task_graph(task_graph: Any, errors: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    if not isinstance(task_graph, list):
        add_error(errors, "invalid_task_graph", "task_graph must be a list.")
        return tasks

    for node in task_graph:
        if not isinstance(node, dict):
            add_error(errors, "invalid_task_graph", "Each task graph node must be an object.")
            continue
        task_id = node.get("id")
        if not task_id:
            add_error(errors, "invalid_task_graph", "Task graph node is missing id.")
            continue
        if task_id in tasks:
            add_error(errors, "duplicate_task", f"Task graph contains duplicate task id '{task_id}'.")
            continue
        if not node.get("action"):
            add_error(errors, "invalid_task_graph", f"Task '{task_id}' is missing action.")
        if not isinstance(node.get("parameters", []), list):
            add_error(errors, "invalid_task_graph", f"Task '{task_id}' parameters must be a list.")
        if not isinstance(node.get("depends_on", []), list):
            add_error(errors, "invalid_task_graph", f"Task '{task_id}' depends_on must be a list.")
        tasks[task_id] = node

    for task_id, node in tasks.items():
        for dependency in node.get("depends_on", []):
            if dependency not in tasks:
                add_error(errors, "unknown_dependency", f"Task '{task_id}' depends on unknown task '{dependency}'.")
    validate_dependency_graph_is_acyclic(tasks, errors)
    return tasks


def validate_dependency_graph_is_acyclic(
    tasks: dict[str, dict[str, Any]],
    errors: list[dict[str, str]],
) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str, path: list[str]) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            cycle = path[path.index(task_id) :] + [task_id]
            add_error(errors, "cyclic_dependency", f"Task graph has dependency cycle: {' -> '.join(cycle)}.")
            return

        visiting.add(task_id)
        for dependency in tasks.get(task_id, {}).get("depends_on", []):
            if dependency in tasks:
                visit(dependency, [*path, dependency])
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id, [task_id])


def validate_assignments(
    assignments: Any,
    tasks: dict[str, dict[str, Any]],
    robot_capabilities: dict[str, set[str]],
    errors: list[dict[str, str]],
) -> dict[str, str]:
    assigned: dict[str, str] = {}
    if not isinstance(assignments, list):
        add_error(errors, "invalid_assignments", "assignments must be a list.")
        return assigned

    for assignment in assignments:
        if not isinstance(assignment, dict):
            add_error(errors, "invalid_assignment", "Each assignment must be an object.")
            continue
        task_id = assignment.get("task_id")
        robot = assignment.get("robot")
        if task_id not in tasks:
            add_error(errors, "unknown_task", f"Assignment references unknown task '{task_id}'.")
            continue
        if robot not in robot_capabilities:
            add_error(errors, "unknown_robot", f"Task '{task_id}' is assigned to unknown robot '{robot}'.")
            continue
        action = tasks[task_id].get("action")
        if action not in robot_capabilities[robot]:
            add_error(errors, "invalid_capability", f"Robot '{robot}' cannot execute action '{action}'.")
        if task_id in assigned:
            add_error(errors, "duplicate_assignment", f"Task '{task_id}' has more than one assignment.")
        assigned[task_id] = robot

    for task_id in tasks:
        if task_id not in assigned:
            add_error(errors, "unassigned_task", f"Task '{task_id}' has no robot assignment.")
    return assigned


def validate_behavior_trees(
    behavior_trees: Any,
    robot_capabilities: dict[str, set[str]],
    errors: list[dict[str, str]],
) -> None:
    if not isinstance(behavior_trees, dict):
        add_error(errors, "invalid_bt", "behavior_trees must be an object keyed by robot id.")
        return

    for robot, tree in behavior_trees.items():
        if robot not in robot_capabilities:
            add_error(errors, "unknown_robot", f"Behavior tree is defined for unknown robot '{robot}'.")
            continue
        validate_bt_node(tree, robot, robot_capabilities, errors, f"behavior_trees.{robot}")


def validate_bt_node(
    node: Any,
    robot: str,
    robot_capabilities: dict[str, set[str]],
    errors: list[dict[str, str]],
    path: str,
) -> None:
    if not isinstance(node, dict):
        add_error(errors, "invalid_bt", f"{path} must be an object.")
        return

    node_type = node.get("type")
    if node_type == "Sequence":
        children = node.get("children")
        if not isinstance(children, list):
            add_error(errors, "invalid_bt", f"{path}.children must be a list.")
            return
        for index, child in enumerate(children):
            validate_bt_node(child, robot, robot_capabilities, errors, f"{path}.children[{index}]")
        return

    if node_type == "Action":
        name = node.get("name")
        if name not in robot_capabilities.get(robot, set()):
            add_error(errors, "invalid_bt_action", f"Robot '{robot}' cannot execute BT action '{name}' at {path}.")
        if not isinstance(node.get("parameters"), list):
            add_error(errors, "invalid_bt", f"{path}.parameters must be a list.")
        return

    if node_type == "Condition":
        if not node.get("name"):
            add_error(errors, "invalid_bt", f"{path} condition is missing name.")
        if not isinstance(node.get("parameters"), list):
            add_error(errors, "invalid_bt", f"{path}.parameters must be a list.")
        return

    add_error(errors, "invalid_bt", f"{path} uses unsupported node type '{node_type}'.")


def validate_assigned_actions_are_in_trees(
    behavior_trees: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    errors: list[dict[str, str]],
) -> None:
    flattened_by_robot = {robot: flatten_tree(tree) for robot, tree in behavior_trees.items()}
    for task_id, robot in assignments.items():
        task = tasks.get(task_id, {})
        if not has_action(flattened_by_robot.get(robot, []), task.get("action"), task.get("parameters", [])):
            add_error(
                errors,
                "missing_bt_action",
                f"Robot '{robot}' is assigned task '{task_id}', but its BT lacks action "
                f"{format_predicate(task.get('action', ''), task.get('parameters', []))}.",
            )


def validate_bt_actions_are_assigned(
    behavior_trees: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    errors: list[dict[str, str]],
) -> None:
    assigned_actions = {
        (robot, tasks[task_id].get("action"), tuple(tasks[task_id].get("parameters", [])))
        for task_id, robot in assignments.items()
        if task_id in tasks
    }

    for robot, tree in behavior_trees.items():
        for node in flatten_tree(tree):
            if node.get("type") != "Action":
                continue
            action_key = (robot, node.get("name"), tuple(node.get("parameters", [])))
            if action_key not in assigned_actions:
                add_error(
                    errors,
                    "unassigned_bt_action",
                    f"Robot '{robot}' BT has action {action_label(node)}, but no matching task assignment exists.",
                )


def validate_required_actions(
    behavior_trees: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    scenario: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    flattened_by_robot = {robot: flatten_tree(tree) for robot, tree in behavior_trees.items()}
    assigned_actions = {
        (robot, tasks[task_id].get("action"), tuple(tasks[task_id].get("parameters", [])))
        for task_id, robot in assignments.items()
        if task_id in tasks
    }

    for required in scenario_required_actions(scenario):
        robot = required.get("robot")
        action = required.get("action")
        parameters = required.get("parameters", [])
        action_text = format_predicate(action or "", parameters)
        key = (robot, action, tuple(parameters))

        if key not in assigned_actions:
            add_error(
                errors,
                "missing_required_assignment",
                f"Required action {action_text} for robot '{robot}' is missing from task_graph or assignments.",
            )

        if not has_action(flattened_by_robot.get(robot, []), action, parameters):
            add_error(
                errors,
                "missing_required_bt_action",
                f"Required action {action_text} is missing from robot '{robot}' BT.",
            )


def validate_required_tasks(
    behavior_trees: dict[str, Any],
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    scenario: dict[str, Any],
    errors: list[dict[str, str]],
) -> None:
    flattened_by_robot = {robot: flatten_tree(tree) for robot, tree in behavior_trees.items()}

    for required in scenario.get("required_tasks", []):
        task_id = required.get("id")
        robot = required.get("robot")
        action = required.get("action")
        parameters = required.get("parameters", [])
        depends_on = required.get("depends_on", [])
        action_text = format_predicate(action or "", parameters)
        task = tasks.get(task_id)

        if not task:
            add_error(errors, "missing_required_task", f"Required task '{task_id}' for {action_text} is missing.")
            continue

        if task.get("action") != action or task.get("parameters", []) != parameters:
            add_error(
                errors,
                "required_task_mismatch",
                f"Required task '{task_id}' must be {action_text}, but plan has "
                f"{format_predicate(task.get('action', ''), task.get('parameters', []))}.",
            )

        missing_dependencies = [dependency for dependency in depends_on if dependency not in task.get("depends_on", [])]
        if missing_dependencies:
            add_error(
                errors,
                "missing_required_dependency",
                f"Required task '{task_id}' is missing dependencies {missing_dependencies}.",
            )

        if assignments.get(task_id) != robot:
            add_error(
                errors,
                "required_assignment_mismatch",
                f"Required task '{task_id}' must be assigned to robot '{robot}'.",
            )

        if not has_action(flattened_by_robot.get(robot, []), action, parameters):
            add_error(
                errors,
                "missing_required_bt_action",
                f"Required task '{task_id}' action {action_text} is missing from robot '{robot}' BT.",
            )


def validate_handoffs(plan: dict[str, Any], scenario: dict[str, Any], errors: list[dict[str, str]]) -> None:
    sync_conditions = {
        sync.get("condition")
        for sync in plan.get("synchronization", [])
        if isinstance(sync, dict) and sync.get("condition")
    }
    behavior_trees = plan.get("behavior_trees", {})
    robot_actions = {
        robot["id"]: {capability["name"]: capability for capability in robot.get("capabilities", [])}
        for robot in scenario.get("robots", [])
    }
    robot_ids = set(robot_actions)
    nodes_by_robot = {
        robot: flatten_tree(tree)
        for robot, tree in behavior_trees.items()
    }
    for sync in plan.get("synchronization", []):
        if not isinstance(sync, dict):
            add_error(errors, "invalid_synchronization", "Each synchronization entry must be an object.")
            continue
        condition = sync.get("condition")
        producer = sync.get("producer")
        consumer = sync.get("consumer")
        if not condition:
            add_error(errors, "invalid_synchronization", "Synchronization entry is missing condition.")
        if producer not in robot_ids:
            add_error(errors, "unknown_robot", f"Synchronization producer '{producer}' is not a scenario robot.")
        if consumer not in robot_ids:
            add_error(errors, "unknown_robot", f"Synchronization consumer '{consumer}' is not a scenario robot.")

    for handoff in scenario.get("handoffs", []):
        condition = handoff.get("condition")
        if condition not in sync_conditions:
            add_error(errors, "missing_synchronization", f"Missing handoff synchronization '{condition}'.")

        producer = handoff.get("producer")
        producer_nodes = nodes_by_robot.get(producer, [])
        producer_action = handoff.get("producer_action", {})
        if producer_action:
            expected_action_index = find_action(
                producer_nodes,
                producer_action.get("action"),
                producer_action.get("parameters", []),
            )
            if expected_action_index is None:
                add_error(
                    errors,
                    "missing_handoff_producer_action",
                    f"Producer '{producer}' BT must include action "
                    f"{format_predicate(producer_action.get('action', ''), producer_action.get('parameters', []))} "
                    f"to create '{condition}'.",
                )

        produced_index = find_effect_index(producer_nodes, producer, condition, robot_actions)
        if produced_index is None:
            add_error(
                errors,
                "missing_handoff_producer",
                f"Producer '{producer}' never creates handoff condition '{condition}' in its BT.",
            )

        before = handoff.get("before", {})
        robot = before.get("robot")
        nodes = nodes_by_robot.get(robot, [])
        condition_name, condition_args = parse_predicate(condition)
        condition_index = find_condition(nodes, condition_name, condition_args)
        action_index = find_action(nodes, before.get("action"), before.get("parameters", []))
        if action_index is not None and condition_index is None:
            mismatched_index = find_condition_name_before(nodes, condition_name, action_index)
            if mismatched_index is not None:
                mismatched = nodes[mismatched_index]
                add_error(
                    errors,
                    "handoff_condition_parameters",
                    f"Robot '{robot}' waits for {action_label(mismatched)}, but must wait for exact "
                    f"condition '{condition}' before action "
                    f"{format_predicate(before.get('action', ''), before.get('parameters', []))}.",
                )
        if action_index is not None and (condition_index is None or condition_index > action_index):
            add_error(
                errors,
                "handoff_order",
                f"Robot '{robot}' must wait for '{condition}' before action "
                f"{format_predicate(before.get('action', ''), before.get('parameters', []))}.",
            )


def simulate_plan(plan: dict[str, Any], scenario: dict[str, Any], max_ticks: int) -> dict[str, Any]:
    robots = {
        robot["id"]: {capability["name"]: capability for capability in robot.get("capabilities", [])}
        for robot in scenario.get("robots", [])
    }
    state = set(scenario.get("initial_state", []))
    trees = {
        robot: flatten_tree(tree)
        for robot, tree in plan.get("behavior_trees", {}).items()
    }
    cursors = {robot: 0 for robot in trees}
    trace: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for tick in range(1, max_ticks + 1):
        if all(cursors[robot] >= len(nodes) for robot, nodes in trees.items()):
            return simulation_result(state, trace, errors, scenario)

        progress = False
        waiting: list[dict[str, Any]] = []
        for robot, nodes in trees.items():
            if cursors[robot] >= len(nodes):
                continue

            node = nodes[cursors[robot]]
            if node.get("type") == "Condition":
                predicate = format_predicate(node.get("name", ""), node.get("parameters", []))
                if predicate_satisfied(predicate, state):
                    trace.append({"tick": tick, "robot": robot, "event": "condition", "condition": predicate})
                    cursors[robot] += 1
                    progress = True
                else:
                    waiting.append({"robot": robot, "condition": predicate})
                continue

            if node.get("type") == "Action":
                capability = robots.get(robot, {}).get(node.get("name"))
                missing = missing_preconditions(capability, node.get("parameters", []), state)
                if missing:
                    waiting.append({"robot": robot, "action": action_label(node), "missing_preconditions": missing})
                    continue

                effects = apply_action(capability, node.get("parameters", []), state)
                trace.append(
                    {
                        "tick": tick,
                        "robot": robot,
                        "event": "action",
                        "action": action_label(node),
                        "effects": effects,
                    }
                )
                cursors[robot] += 1
                progress = True

        if not progress:
            errors.append({"type": "deadlock", "waiting": waiting})
            return simulation_result(state, trace, errors, scenario)

    errors.append({"type": "timeout", "message": f"Simulation exceeded {max_ticks} ticks."})
    return simulation_result(state, trace, errors, scenario)


def missing_preconditions(capability: dict[str, Any] | None, actual_parameters: list[str], state: set[str]) -> list[str]:
    if not capability:
        return ["unknown_capability"]
    bindings = dict(zip(capability.get("parameters", []), actual_parameters))
    preconditions = [substitute(predicate, bindings) for predicate in capability.get("preconditions", [])]
    return [predicate for predicate in preconditions if not predicate_satisfied(predicate, state)]


def apply_action(capability: dict[str, Any], actual_parameters: list[str], state: set[str]) -> list[str]:
    bindings = dict(zip(capability.get("parameters", []), actual_parameters))
    effects = [substitute(predicate, bindings) for predicate in capability.get("effects", [])]
    for effect in effects:
        apply_effect(effect, state)
    return effects


def apply_effect(effect: str, state: set[str]) -> None:
    name, args = parse_predicate(effect)
    if name.startswith("not_"):
        remove_matching(state, name.removeprefix("not_"), args)
        return

    if name.endswith("_open"):
        state.discard(format_predicate(name.removesuffix("_open") + "_closed", args))
    if name.endswith("_closed"):
        state.discard(format_predicate(name.removesuffix("_closed") + "_open", args))
    if name == "robot_at" or name.endswith("_at"):
        remove_matching(state, name, args[:1])

    state.add(effect)


def predicate_satisfied(predicate: str, state: set[str]) -> bool:
    if predicate in state:
        return True

    name, args = parse_predicate(predicate)
    if name == "robot_near" and len(args) == 2:
        robot, obj = args
        robot_locations = {
            values[1]
            for item in state
            for item_name, values in [parse_predicate(item)]
            if item_name == "robot_at" and len(values) == 2 and values[0] == robot
        }
        object_locations = {
            values[1]
            for item in state
            for item_name, values in [parse_predicate(item)]
            if item_name.endswith("_at") and len(values) == 2 and values[0] == obj
        }
        return bool(robot_locations & object_locations)
    return False


def simulation_result(
    state: set[str],
    trace: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    goal_success = all(predicate in state for predicate in scenario.get("goal_state", []))
    return {
        "success": not errors and goal_success,
        "goal_success": goal_success,
        "final_state": sorted(state),
        "trace": trace,
        "errors": errors,
    }


def skipped_simulation() -> dict[str, Any]:
    return {
        "success": False,
        "goal_success": False,
        "final_state": [],
        "trace": [],
        "errors": [{"type": "validation_failed", "message": "Simulation skipped because validation failed."}],
    }


def flatten_tree(node: Any) -> list[dict[str, Any]]:
    if not isinstance(node, dict):
        return []
    if node.get("type") == "Sequence":
        flattened: list[dict[str, Any]] = []
        for child in node.get("children", []):
            flattened.extend(flatten_tree(child))
        return flattened
    return [node]


def has_action(nodes: list[dict[str, Any]], name: str, parameters: list[str]) -> bool:
    return find_action(nodes, name, parameters) is not None


def find_action(nodes: list[dict[str, Any]], name: str | None, parameters: list[str]) -> int | None:
    for index, node in enumerate(nodes):
        if node.get("type") == "Action" and node.get("name") == name and node.get("parameters") == parameters:
            return index
    return None


def find_condition(nodes: list[dict[str, Any]], name: str, parameters: list[str]) -> int | None:
    for index, node in enumerate(nodes):
        if node.get("type") == "Condition" and node.get("name") == name and node.get("parameters") == parameters:
            return index
    return None


def find_condition_name_before(nodes: list[dict[str, Any]], name: str, before_index: int) -> int | None:
    for index, node in enumerate(nodes[:before_index]):
        if node.get("type") == "Condition" and node.get("name") == name:
            return index
    return None


def find_effect_index(
    nodes: list[dict[str, Any]],
    robot: str | None,
    target_effect: str,
    robot_actions: dict[str, dict[str, dict[str, Any]]],
) -> int | None:
    if robot is None:
        return None

    for index, node in enumerate(nodes):
        if node.get("type") != "Action":
            continue
        capability = robot_actions.get(robot, {}).get(node.get("name"))
        if not capability:
            continue
        bindings = dict(zip(capability.get("parameters", []), node.get("parameters", [])))
        effects = [substitute(effect, bindings) for effect in capability.get("effects", [])]
        if target_effect in effects:
            return index
    return None


def substitute(predicate: str, bindings: dict[str, str]) -> str:
    result = predicate
    for variable, value in bindings.items():
        result = re.sub(rf"\b{re.escape(variable)}\b", value, result)
    return result


def remove_matching(state: set[str], name: str, args_prefix: list[str]) -> None:
    state.difference_update(
        {
            predicate
            for predicate in state
            for predicate_name, predicate_args in [parse_predicate(predicate)]
            if predicate_name == name and predicate_args[: len(args_prefix)] == args_prefix
        }
    )


def parse_predicate(predicate: str | None) -> tuple[str, list[str]]:
    if not predicate or "(" not in predicate or not predicate.endswith(")"):
        return str(predicate or ""), []
    name, raw_args = predicate.split("(", 1)
    args = [argument.strip() for argument in raw_args[:-1].split(",") if argument.strip()]
    return name.strip(), args


def format_predicate(name: str, parameters: list[str]) -> str:
    return f"{name}({', '.join(parameters)})"


def action_label(node: dict[str, Any]) -> str:
    return format_predicate(node.get("name", ""), node.get("parameters", []))


def add_error(errors: list[dict[str, str]], error_type: str, message: str) -> None:
    errors.append({"type": error_type, "message": message})


def extract_json(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fence(text.strip())
    candidate = first_json_object(cleaned)
    if candidate is None:
        raise ValueError("Could not find a JSON object in the LLM response.")
    return json.loads(candidate)


def strip_markdown_fence(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def first_json_object(text: str) -> str | None:
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


def openai_chat_completions_url() -> str:
    explicit_url = os.environ.get("OPENAI_API_URL")
    if explicit_url:
        return explicit_url
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


def safe_api_error(message: str) -> str:
    message = message.replace("\r", " ").replace("\n", " ")
    message = re.sub(r"sk-proj-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    return message[:600]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def print_summary(result: dict[str, Any], output_path: Path) -> None:
    print("LLM multi-robot BT planner")
    print("=" * 28)
    print(f"Task: {result['task_id']}")
    if "results" in result:
        print(f"Trials: {result['trials']}")
        print(f"Valid: {result['valid_count']}/{result['trials']}")
        print(f"Goal reached: {result['goal_success_count']}/{result['trials']}")
        print(f"Success: {result['success_count']}/{result['trials']}")
    else:
        print(f"Valid: {yes_no(result['valid'])}")
        print(f"Goal reached: {yes_no(result['goal_success'])}")
        print(f"Correction rounds: {result['correction_rounds']}")
    print(f"Result file: {output_path}")

    if "results" in result:
        failed = [item for item in result["results"] if not item["success"]]
        if failed:
            print("\nFailed trials:")
            for item in failed:
                print(
                    f"- trial {item['trial']}: valid={yes_no(item['valid'])}, "
                    f"goal={yes_no(item['goal_success'])}, corrections={item['correction_rounds']}"
                )
        return

    if result["validation_errors"]:
        print("\nValidation errors:")
        for error in result["validation_errors"]:
            print(f"- [{error['type']}] {error['message']}")
    elif result["simulation"]["errors"]:
        print("\nSimulation errors:")
        for error in result["simulation"]["errors"]:
            print(f"- [{error.get('type')}] {error}")


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
