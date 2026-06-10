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

    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    result = run_planner(
        scenario=scenario,
        model=model,
        max_corrections=args.max_corrections,
        max_ticks=args.max_ticks,
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
    args = parser.parse_args()
    if args.max_ticks < 1:
        parser.error("--max-ticks must be at least 1")
    if args.max_corrections < 0:
        parser.error("--max-corrections cannot be negative")
    return args


def run_planner(
    scenario: dict[str, Any],
    model: str,
    max_corrections: int,
    max_ticks: int,
) -> dict[str, Any]:
    plan, validation, simulation, correction_rounds = generate_evaluated_plan(
        scenario=scenario,
        model=model,
        max_corrections=max_corrections,
        max_ticks=max_ticks,
    )
    return {
        "task_id": scenario.get("task_id"),
        "model": model,
        "valid": validation["valid"],
        "success": simulation["success"],
        "goal_success": simulation["goal_success"],
        "correction_rounds": correction_rounds,
        "plan": plan,
        "validation_errors": validation["errors"],
        "simulation": {
            "final_state": simulation["final_state"],
            "trace": simulation["trace"],
            "errors": simulation["errors"],
        },
    }


def generate_evaluated_plan(
    scenario: dict[str, Any],
    model: str,
    max_corrections: int,
    max_ticks: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    plan = query_plan(prompt=build_prompt(scenario), model=model)
    correction_rounds = 0

    for round_index in range(0, max_corrections + 1):
        plan.setdefault("task_id", scenario.get("task_id"))
        validation = validate_plan(plan, scenario)
        simulation = (
            simulate_plan(plan, scenario, max_ticks=max_ticks)
            if validation["valid"]
            else skipped_simulation()
        )

        if validation["valid"] and simulation["success"]:
            break
        if round_index == max_corrections:
            break

        correction_rounds += 1
        plan = query_plan(
            prompt=build_correction_prompt(scenario, validation["errors"], simulation),
            model=model,
        )

    return plan, validation, simulation, correction_rounds


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

    task_summary = {
        "task_id": scenario.get("task_id"),
        "instruction": scenario.get("instruction"),
        "initial_state": scenario.get("initial_state", []),
        "goal_state": scenario.get("goal_state", []),
        "objects": scenario.get("objects", []),
        "locations": scenario.get("locations", []),
    }

    return (
        "Generate a compact symbolic multi-robot behavior-tree plan for the scenario below.\n"
        "Return ONLY valid JSON. Do not use markdown or explanatory text.\n\n"
        f"Scenario:\n{json.dumps(task_summary, indent=2)}\n\n"
        f"Robot capability library:\n{json.dumps(robot_summary, indent=2)}\n\n"
        "Capability dependency hints derived from preconditions/effects:\n"
        f"{json.dumps(build_dependency_hints(scenario), indent=2)}\n\n"
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
        "2. Treat capability preconditions/effects as authoritative when choosing action parameters.\n"
        "3. Infer the task graph from goal_state by matching goals and missing preconditions to capability effects.\n"
        "4. Include every action needed to make the goals true from initial_state; there is no hidden task list.\n"
        "5. Assign each task graph action to exactly one robot with that capability.\n"
        "6. Use the dependency hints to add producer actions for every non-initial precondition.\n"
        "7. If you select a consumer from a dependency hint, instantiate one candidate_producer with "
        "matching concrete parameters and include it before the consumer.\n"
        "8. Dependency hints separate robot and action fields; task_graph.action and BT Action.name must be bare names, never robot.action.\n"
        "9. A Condition node only waits; it never creates a predicate or replaces a missing producer action.\n"
        "10. The task_graph must be acyclic.\n"
        "11. Keep BTs simple: Sequence roots with Action and Condition children only.\n"
        "12. Every assigned task action must appear as an Action node in that robot's BT with identical parameters.\n"
        "13. Same action names on different robots are distinct tasks; include separate task nodes and BT Actions for each robot.\n"
        "14. Robot-specific predicates are not interchangeable: holding(robot_a, x) does not satisfy holding(robot_b, x).\n"
        "15. Use synchronization only for inter-robot waits: if robot B needs a predicate produced by robot A, "
        "add it to synchronization and put the exact Condition before B consumes it.\n"
        "16. Do not add synchronization entries for same-robot sequencing or predicates already true in initial_state.\n"
        "17. A producer BT must execute an action whose effects create each synchronization condition.\n"
        "18. Match predicate arguments exactly. If a later precondition needs object_at(x, location_a), "
        "the producer action must use location_a, not a nearby or generic location.\n"
        "19. Before a robot waits for a downstream completion condition, it must first produce any resources "
        "that downstream robots need from it.\n"
        "20. Do not add a Condition for a predicate that no robot can produce and that is not already in initial_state.\n"
        "21. Follow the scenario instruction, but derive executable steps from capability preconditions/effects.\n"
        "22. Never omit predicate arguments in Condition nodes or synchronization.condition. "
        "Use the full predicate from preconditions/effects/goals.\n"
        "23. In BT nodes, 'name' is only the bare action or predicate name.\n"
        "   Correct: {\"type\":\"Condition\",\"name\":\"drawer_closed\",\"parameters\":[\"parts_drawer\"]}\n"
        "   Wrong:   {\"type\":\"Condition\",\"name\":\"drawer_closed(parts_drawer)\"}\n"
        "24. Every Action and Condition node must include a parameters array, even if empty.\n"
        "25. Keep the plan compact, but completeness beats compactness: never omit producer actions needed for preconditions.\n"
    )


def build_dependency_hints(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    initial_state = set(scenario.get("initial_state", []))
    constants = scenario_constants(scenario)
    hints: list[dict[str, Any]] = []
    for robot in scenario.get("robots", []):
        robot_id = robot.get("id", "")
        for capability in robot.get("capabilities", []):
            action_parameters = capability.get("parameters", [])
            action = {
                "robot": robot_id,
                "action": capability.get("name", ""),
                "parameters": action_parameters,
            }
            for precondition in capability.get("preconditions", []):
                if predicate_template_satisfied_by_initial(precondition, initial_state, action_parameters, constants):
                    continue
                producers = candidate_producer_specs(precondition, scenario)
                if producers:
                    hints.append(
                        {
                            "consumer": action,
                            "needs": precondition,
                            "candidate_producers": producers,
                        }
                    )
    return hints


def predicate_template_satisfied_by_initial(
    predicate: str,
    initial_state: set[str],
    action_parameters: list[str],
    constants: set[str],
) -> bool:
    name, args = parse_predicate(predicate)
    for fact in initial_state:
        fact_name, fact_args = parse_predicate(fact)
        if fact_name == name and unify_effect_args(args, fact_args, action_parameters, constants) is not None:
            return True
    return False


def build_correction_prompt(
    scenario: dict[str, Any],
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
        "Correction requirements:\n"
        "- Rebuild the complete plan from scratch; do not copy malformed nodes from the failed plan.\n"
        "- Every BT Action/Condition node must have a bare name and a parameters array.\n"
        "- task_graph.action and BT Action.name must never include a robot prefix such as robot.action.\n"
        "- Do not preserve malformed conditions with missing parameters; copy the exact predicate arguments from "
        "capability preconditions/effects.\n"
        "- Infer missing actions from goal_state and capability preconditions/effects; there is no hidden task list.\n"
        "- For each assignment, the assigned action with exact parameters must appear in that robot's BT.\n"
        "- If the same action name appears for multiple robots, add separate task_graph nodes and assignments for each robot.\n"
        "- Robot-specific predicates must match exactly; one robot holding an object does not make another robot hold it.\n"
        "- For unsupported_condition, add the listed candidate producer Action instead of waiting on the condition when the producer is the same robot.\n"
        "- Consumer BTs should wait for inter-robot synchronization conditions before the consuming actions.\n"
        "- Do not create synchronization entries for same-robot ordering or initial-state predicates.\n"
        "- Synchronization Condition nodes must use the exact predicate arguments produced by the producer action.\n"
        "- Producer BTs must execute actions that create synchronization conditions before consumers wait for them.\n"
        "- A Condition node cannot fix unsupported_precondition; add the candidate producer action instead.\n"
        "- Treat unsupported_precondition and simulation missing_preconditions literally: add or re-parameterize "
        "earlier actions so an effect string exactly equals each missing predicate.\n"
        "- For every unsupported_precondition, one listed candidate producer action must appear in task_graph, assignments, and the producer robot BT.\n"
        "- If a missing predicate contains a location, use that exact location in the producer action parameters.\n"
        "- Before a robot waits for a downstream condition, make sure it has already produced every resource "
        "that downstream robots need from it.\n"
        "- Task graph dependencies must be acyclic and point from consumers to producer tasks that happen earlier.\n"
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
    validate_predicate_support(plan, scenario, errors)
    validate_synchronization(plan, scenario, errors)

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


def validate_predicate_support(plan: dict[str, Any], scenario: dict[str, Any], errors: list[dict[str, str]]) -> None:
    initial_state = set(scenario.get("initial_state", []))
    behavior_trees = plan.get("behavior_trees", {})
    robot_actions = {
        robot["id"]: {capability["name"]: capability for capability in robot.get("capabilities", [])}
        for robot in scenario.get("robots", [])
    }
    produced = produced_predicates_by_generated_actions(behavior_trees, robot_actions)

    for goal in scenario.get("goal_state", []):
        if goal not in initial_state and goal not in produced:
            candidates = candidate_producers(goal, scenario)
            add_error(
                errors,
                "unsupported_goal",
                f"Goal '{goal}' is not initially true and no generated BT action creates it."
                f"{candidate_text(candidates)}",
            )

    for robot, tree in behavior_trees.items():
        for node in flatten_tree(tree):
            if node.get("type") == "Condition":
                predicate = format_predicate(node.get("name", ""), node.get("parameters", []))
                initially_satisfied = predicate_satisfied(predicate, initial_state)
                if initially_satisfied or predicate in produced:
                    if not initially_satisfied:
                        validate_condition_not_before_same_robot_producer(
                            robot,
                            tree,
                            node,
                            predicate,
                            robot_actions,
                            errors,
                        )
                    continue
                add_error(
                    errors,
                    "unsupported_condition",
                    f"Condition '{predicate}' in robot '{robot}' BT is not initially true and no generated action creates it."
                    f"{same_name_predicate_text(predicate, initial_state | produced)}"
                    f"{candidate_text(candidate_producers(predicate, scenario))}",
                )
                continue

            if node.get("type") != "Action":
                continue
            capability = robot_actions.get(robot, {}).get(node.get("name"))
            if not capability:
                continue
            bindings = dict(zip(capability.get("parameters", []), node.get("parameters", [])))
            for precondition in capability.get("preconditions", []):
                predicate = substitute(precondition, bindings)
                if predicate_satisfied(predicate, initial_state) or predicate in produced:
                    continue
                add_error(
                    errors,
                    "unsupported_precondition",
                    f"Action {format_predicate(node.get('name', ''), node.get('parameters', []))} "
                    f"on robot '{robot}' needs '{predicate}', but no initial predicate or generated action creates it."
                    f"{candidate_text(candidate_producers(predicate, scenario))}",
                )


def produced_predicates_by_generated_actions(
    behavior_trees: dict[str, Any],
    robot_actions: dict[str, dict[str, dict[str, Any]]],
) -> set[str]:
    produced: set[str] = set()
    for robot, tree in behavior_trees.items():
        for node in flatten_tree(tree):
            if node.get("type") != "Action":
                continue
            capability = robot_actions.get(robot, {}).get(node.get("name"))
            if not capability:
                continue
            bindings = dict(zip(capability.get("parameters", []), node.get("parameters", [])))
            for effect in capability.get("effects", []):
                substituted = substitute(effect, bindings)
                if not parse_predicate(substituted)[0].startswith("not_"):
                    produced.add(substituted)
    return produced


def validate_condition_not_before_same_robot_producer(
    robot: str,
    tree: Any,
    condition_node: dict[str, Any],
    predicate: str,
    robot_actions: dict[str, dict[str, dict[str, Any]]],
    errors: list[dict[str, str]],
) -> None:
    nodes = flatten_tree(tree)
    condition_index = next((index for index, node in enumerate(nodes) if node is condition_node), None)
    if condition_index is None:
        return
    producer_index = find_effect_index(nodes, robot, predicate, robot_actions)
    if producer_index is not None and condition_index < producer_index:
        add_error(
            errors,
            "condition_before_producer",
            f"Robot '{robot}' waits for '{predicate}' before its own BT action creates it.",
        )


def candidate_producers(predicate: str, scenario: dict[str, Any]) -> list[str]:
    return [
        f"robot {candidate['robot']} action {format_predicate(candidate['action'], candidate['parameters'])}"
        for candidate in candidate_producer_specs(predicate, scenario)
    ]


def candidate_producer_specs(predicate: str, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    target_name, target_args = parse_predicate(predicate)
    constants = scenario_constants(scenario)
    for robot in scenario.get("robots", []):
        robot_id = robot.get("id", "")
        for capability in robot.get("capabilities", []):
            for effect in capability.get("effects", []):
                effect_name, effect_args = parse_predicate(effect)
                if effect_name.startswith("not_") or effect_name != target_name:
                    continue
                bindings = unify_effect_args(effect_args, target_args, capability.get("parameters", []), constants)
                if bindings is None:
                    continue
                if predicate not in set(scenario.get("goal_state", [])) and requires_goal_predicate(capability, bindings, scenario):
                    continue
                parameters = [bindings.get(parameter, parameter) for parameter in capability.get("parameters", [])]
                candidates.append(
                    {
                        "robot": robot_id,
                        "action": capability.get("name", ""),
                        "parameters": parameters,
                    }
                )
    return candidates[:5]


def requires_goal_predicate(
    capability: dict[str, Any],
    bindings: dict[str, str],
    scenario: dict[str, Any],
) -> bool:
    goals = set(scenario.get("goal_state", []))
    return any(substitute(precondition, bindings) in goals for precondition in capability.get("preconditions", []))


def scenario_constants(scenario: dict[str, Any]) -> set[str]:
    robot_ids = {robot.get("id", "") for robot in scenario.get("robots", [])}
    return set(scenario.get("objects", [])) | set(scenario.get("locations", [])) | robot_ids


def unify_effect_args(
    effect_args: list[str],
    target_args: list[str],
    action_parameters: list[str],
    constants: set[str],
) -> dict[str, str] | None:
    if len(effect_args) != len(target_args):
        return None

    action_parameter_set = set(action_parameters)
    bindings: dict[str, str] = {}
    for effect_arg, target_arg in zip(effect_args, target_args):
        if effect_arg in constants:
            if effect_arg != target_arg:
                return None
        elif effect_arg in action_parameter_set:
            if target_arg not in constants and target_arg != effect_arg:
                return None
            existing = bindings.get(effect_arg)
            if existing is not None and existing != target_arg:
                return None
            bindings[effect_arg] = target_arg
        elif effect_arg != target_arg:
            return None
    return bindings


def candidate_text(candidates: list[str]) -> str:
    if not candidates:
        return ""
    return f" Candidate producer actions: {', '.join(candidates)}."


def same_name_predicate_text(predicate: str, candidates: set[str]) -> str:
    name, _ = parse_predicate(predicate)
    matches = sorted(
        candidate
        for candidate in candidates
        if parse_predicate(candidate)[0] == name and candidate != predicate
    )
    if not matches:
        return ""
    return f" Did you mean: {', '.join(matches[:5])}?"


def validate_synchronization(plan: dict[str, Any], scenario: dict[str, Any], errors: list[dict[str, str]]) -> None:
    synchronization = plan.get("synchronization", [])
    if not isinstance(synchronization, list):
        add_error(errors, "invalid_synchronization", "synchronization must be a list.")
        return

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

    for sync in synchronization:
        if not isinstance(sync, dict):
            add_error(errors, "invalid_synchronization", "Each synchronization entry must be an object.")
            continue
        condition = sync.get("condition")
        producer = sync.get("producer")
        consumer = sync.get("consumer")
        if not condition:
            add_error(errors, "invalid_synchronization", "Synchronization entry is missing condition.")
            continue
        if producer not in robot_ids:
            add_error(errors, "unknown_robot", f"Synchronization producer '{producer}' is not a scenario robot.")
            continue
        if consumer not in robot_ids:
            add_error(errors, "unknown_robot", f"Synchronization consumer '{consumer}' is not a scenario robot.")
            continue
        producer_nodes = nodes_by_robot.get(producer, [])
        produced_index = find_effect_index(producer_nodes, producer, condition, robot_actions)
        if produced_index is None:
            produced = produced_predicates_by_generated_actions(behavior_trees, robot_actions)
            add_error(
                errors,
                "missing_sync_producer",
                f"Producer '{producer}' never creates synchronization condition '{condition}' in its BT."
                f"{same_name_predicate_text(condition, produced)}",
            )

        nodes = nodes_by_robot.get(consumer, [])
        condition_name, condition_args = parse_predicate(condition)
        condition_index = find_condition(nodes, condition_name, condition_args)
        if condition_index is None:
            add_error(
                errors,
                "missing_sync_condition",
                f"Consumer '{consumer}' BT must include Condition '{condition}' for synchronization.",
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
    print(f"Valid: {yes_no(result['valid'])}")
    print(f"Goal reached: {yes_no(result['goal_success'])}")
    print(f"Correction rounds: {result['correction_rounds']}")
    print(f"Result file: {output_path}")

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
