"""Static validation of a generated plan against a scenario.

The validator turns "is this plan correct?" into concrete, machine-checkable
errors with candidate-producer suggestions - the structured feedback that makes
LLM self-correction tractable. Every check from the original prototype is
preserved; they now operate on the typed :class:`llm_mr_bt_planner.plan.Plan` and the
declarative effect model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bt import BTNode, iter_leaves
from .domain import Scenario, candidate_producers, positive_effects
from .plan import Plan
from .predicates import format_predicate, parse_predicate, substitute


@dataclass
class ValidationError:
    type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "message": self.message}


@dataclass
class ValidationReport:
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, error_type: str, message: str) -> None:
        self.errors.append(ValidationError(error_type, message))

    def to_dicts(self) -> list[dict[str, str]]:
        return [error.to_dict() for error in self.errors]


def validate_plan(plan: Plan, scenario: Scenario, suggest_producers: bool = False) -> ValidationReport:
    """Validate ``plan`` against ``scenario``.

    The checks are task-agnostic (structure, capability match, predicate support,
    synchronization consistency). With ``suggest_producers=False`` (the default)
    the report says *what* is wrong but never names a specific producer action -
    keeping the loop a general checker rather than a task-specific planner. Set
    ``suggest_producers=True`` only for ablation/assisted-mode experiments.
    """
    report = ValidationReport()

    for field_name in plan.missing_fields():
        report.add("missing_field", f"Plan is missing '{field_name}'.")
    for robot, reason in plan.unparsable_trees.items():
        report.add("invalid_bt", f"Behavior tree for '{robot}' could not be parsed: {reason}.")
    if report.errors:
        return report

    tasks = _validate_task_graph(plan, report)
    assignments = _validate_assignments(plan, tasks, scenario, report)
    _validate_behavior_trees(plan, scenario, report)
    _validate_assigned_actions_present(plan, tasks, assignments, report)
    _validate_bt_actions_assigned(plan, tasks, assignments, report)
    _validate_predicate_support(plan, scenario, report, suggest_producers)
    _validate_synchronization(plan, scenario, report, suggest_producers)
    return report


# --------------------------------------------------------------------------- #
# Task graph
# --------------------------------------------------------------------------- #


def _validate_task_graph(plan: Plan, report: ValidationReport) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for node in plan.task_graph:
        if not isinstance(node, dict):
            report.add("invalid_task_graph", "Each task graph node must be an object.")
            continue
        task_id = node.get("id")
        if not task_id:
            report.add("invalid_task_graph", "Task graph node is missing id.")
            continue
        if task_id in tasks:
            report.add("duplicate_task", f"Task graph contains duplicate task id '{task_id}'.")
            continue
        if not node.get("action"):
            report.add("invalid_task_graph", f"Task '{task_id}' is missing action.")
        if not isinstance(node.get("parameters", []), list):
            report.add("invalid_task_graph", f"Task '{task_id}' parameters must be a list.")
        if not isinstance(node.get("depends_on", []), list):
            report.add("invalid_task_graph", f"Task '{task_id}' depends_on must be a list.")
        tasks[task_id] = node

    for task_id, node in tasks.items():
        for dependency in node.get("depends_on", []):
            if dependency not in tasks:
                report.add("unknown_dependency", f"Task '{task_id}' depends on unknown task '{dependency}'.")
    _check_acyclic(tasks, report)
    return tasks


def _check_acyclic(tasks: dict[str, dict[str, Any]], report: ValidationReport) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str, path: list[str]) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            cycle = path[path.index(task_id):] + [task_id]
            report.add("cyclic_dependency", f"Task graph has dependency cycle: {' -> '.join(cycle)}.")
            return
        visiting.add(task_id)
        for dependency in tasks.get(task_id, {}).get("depends_on", []):
            if dependency in tasks:
                visit(dependency, [*path, dependency])
        visiting.discard(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id, [task_id])


# --------------------------------------------------------------------------- #
# Assignments
# --------------------------------------------------------------------------- #


def _validate_assignments(
    plan: Plan,
    tasks: dict[str, dict[str, Any]],
    scenario: Scenario,
    report: ValidationReport,
) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for assignment in plan.assignments:
        if not isinstance(assignment, dict):
            report.add("invalid_assignment", "Each assignment must be an object.")
            continue
        task_id = assignment.get("task_id")
        robot_id = assignment.get("robot")
        if task_id not in tasks:
            report.add("unknown_task", f"Assignment references unknown task '{task_id}'.")
            continue
        robot = scenario.robot(robot_id)
        if robot is None:
            report.add("unknown_robot", f"Task '{task_id}' is assigned to unknown robot '{robot_id}'.")
            continue
        action = tasks[task_id].get("action")
        if action not in robot.capability_names:
            report.add("invalid_capability", f"Robot '{robot_id}' cannot execute action '{action}'.")
        if task_id in assigned:
            report.add("duplicate_assignment", f"Task '{task_id}' has more than one assignment.")
            continue  # keep the first assignment so downstream BT cross-checks stay consistent
        assigned[task_id] = robot_id

    for task_id in tasks:
        if task_id not in assigned:
            report.add("unassigned_task", f"Task '{task_id}' has no robot assignment.")
    return assigned


# --------------------------------------------------------------------------- #
# Behavior trees
# --------------------------------------------------------------------------- #


def _validate_behavior_trees(plan: Plan, scenario: Scenario, report: ValidationReport) -> None:
    for robot_id, tree in plan.behavior_trees.items():
        robot = scenario.robot(robot_id)
        if robot is None:
            report.add("unknown_robot", f"Behavior tree is defined for unknown robot '{robot_id}'.")
            continue
        _validate_bt_node(tree, robot.capability_names, robot_id, report, f"behavior_trees.{robot_id}")


def _validate_bt_node(
    node: BTNode,
    capability_names: set[str],
    robot_id: str,
    report: ValidationReport,
    path: str,
) -> None:
    if node.type in {"Sequence", "Fallback", "Parallel"}:
        for index, child in enumerate(node.children):
            _validate_bt_node(child, capability_names, robot_id, report, f"{path}.children[{index}]")
        return
    if node.type == "Action":
        if node.name not in capability_names:
            report.add("invalid_bt_action", f"Robot '{robot_id}' cannot execute BT action '{node.name}' at {path}.")
        return
    if node.type == "Condition":
        if not node.name:
            report.add("invalid_bt", f"{path} condition is missing name.")
        return
    report.add("invalid_bt", f"{path} uses unsupported node type '{node.type}'.")


# --------------------------------------------------------------------------- #
# Action <-> assignment cross-checks
# --------------------------------------------------------------------------- #


def _validate_assigned_actions_present(
    plan: Plan,
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    report: ValidationReport,
) -> None:
    leaves_by_robot = {robot: list(iter_leaves(tree)) for robot, tree in plan.behavior_trees.items()}
    for task_id, robot_id in assignments.items():
        task = tasks.get(task_id, {})
        action = task.get("action")
        parameters = tuple(task.get("parameters", []))
        leaves = leaves_by_robot.get(robot_id, [])
        if not _find_action(leaves, action, parameters):
            report.add(
                "missing_bt_action",
                f"Robot '{robot_id}' is assigned task '{task_id}', but its BT lacks action "
                f"{format_predicate(action or '', parameters)}.",
            )


def _validate_bt_actions_assigned(
    plan: Plan,
    tasks: dict[str, dict[str, Any]],
    assignments: dict[str, str],
    report: ValidationReport,
) -> None:
    assigned_actions = {
        (robot, tasks[task_id].get("action"), tuple(tasks[task_id].get("parameters", [])))
        for task_id, robot in assignments.items()
        if task_id in tasks
    }
    for robot_id, tree in plan.behavior_trees.items():
        for leaf in iter_leaves(tree):
            if leaf.type != "Action":
                continue
            key = (robot_id, leaf.name, tuple(leaf.parameters))
            if key not in assigned_actions:
                report.add(
                    "unassigned_bt_action",
                    f"Robot '{robot_id}' BT has action {leaf.label()}, but no matching task assignment exists.",
                )


# --------------------------------------------------------------------------- #
# Predicate support
# --------------------------------------------------------------------------- #


def _validate_predicate_support(
    plan: Plan, scenario: Scenario, report: ValidationReport, suggest_producers: bool
) -> None:
    initial_state = set(scenario.initial_state)
    produced = produced_predicates(plan, scenario)

    for goal in scenario.goal_state:
        if goal not in initial_state and goal not in produced:
            report.add(
                "unsupported_goal",
                f"Goal '{goal}' is not initially true and no generated BT action creates it."
                f"{_candidate_text(candidate_producers(goal, scenario), suggest_producers)}",
            )

    for robot_id, tree in plan.behavior_trees.items():
        robot = scenario.robot(robot_id)
        if robot is None:
            continue
        leaves = list(iter_leaves(tree))
        for index, leaf in enumerate(leaves):
            if leaf.type == "Condition":
                _check_condition(robot_id, leaves, index, leaf, scenario, initial_state, produced, report, suggest_producers)
            elif leaf.type == "Action":
                _check_action_preconditions(robot_id, leaf, scenario, initial_state, produced, report, suggest_producers)


def _check_condition(
    robot_id: str,
    leaves: list[BTNode],
    index: int,
    leaf: BTNode,
    scenario: Scenario,
    initial_state: set[str],
    produced: set[str],
    report: ValidationReport,
    suggest_producers: bool,
) -> None:
    predicate = leaf.label()
    initially = predicate in initial_state
    if initially or predicate in produced:
        if not initially:
            _check_condition_after_producer(robot_id, leaves, index, predicate, scenario, report)
        return
    report.add(
        "unsupported_condition",
        f"Condition '{predicate}' in robot '{robot_id}' BT is not initially true and no generated action creates it."
        f"{_same_name_text(predicate, initial_state | produced, suggest_producers)}"
        f"{_candidate_text(candidate_producers(predicate, scenario), suggest_producers)}",
    )


def _check_condition_after_producer(
    robot_id: str,
    leaves: list[BTNode],
    condition_index: int,
    predicate: str,
    scenario: Scenario,
    report: ValidationReport,
) -> None:
    producer_index = _producer_index(leaves, robot_id, predicate, scenario)
    if producer_index is not None and condition_index < producer_index:
        report.add(
            "condition_before_producer",
            f"Robot '{robot_id}' waits for '{predicate}' before its own BT action creates it.",
        )


def _check_action_preconditions(
    robot_id: str,
    leaf: BTNode,
    scenario: Scenario,
    initial_state: set[str],
    produced: set[str],
    report: ValidationReport,
    suggest_producers: bool,
) -> None:
    capability = scenario.capability(robot_id, leaf.name or "")
    if capability is None:
        return
    bindings = dict(zip(capability.parameters, leaf.parameters))
    for precondition in capability.preconditions:
        predicate = substitute(precondition, bindings)
        if predicate in initial_state or predicate in produced:
            continue
        report.add(
            "unsupported_precondition",
            f"Action {leaf.label()} on robot '{robot_id}' needs '{predicate}', "
            f"but no initial predicate or generated action creates it."
            f"{_candidate_text(candidate_producers(predicate, scenario), suggest_producers)}",
        )


def produced_predicates(plan: Plan, scenario: Scenario) -> set[str]:
    """Every positive predicate any generated BT action can create."""
    produced: set[str] = set()
    for robot_id, tree in plan.behavior_trees.items():
        for leaf in iter_leaves(tree):
            if leaf.type != "Action":
                continue
            capability = scenario.capability(robot_id, leaf.name or "")
            if capability is None:
                continue
            bindings = dict(zip(capability.parameters, leaf.parameters))
            produced.update(positive_effects(capability.effects, bindings))
    return produced


# --------------------------------------------------------------------------- #
# Synchronization
# --------------------------------------------------------------------------- #


def _validate_synchronization(
    plan: Plan, scenario: Scenario, report: ValidationReport, suggest_producers: bool
) -> None:
    leaves_by_robot = {robot: list(iter_leaves(tree)) for robot, tree in plan.behavior_trees.items()}
    for sync in plan.synchronization:
        if not isinstance(sync, dict):
            report.add("invalid_synchronization", "Each synchronization entry must be an object.")
            continue
        condition = sync.get("condition")
        producer = sync.get("producer")
        consumer = sync.get("consumer")
        if not condition:
            report.add("invalid_synchronization", "Synchronization entry is missing condition.")
            continue
        if producer not in scenario.robot_ids:
            report.add("unknown_robot", f"Synchronization producer '{producer}' is not a scenario robot.")
            continue
        if consumer not in scenario.robot_ids:
            report.add("unknown_robot", f"Synchronization consumer '{consumer}' is not a scenario robot.")
            continue

        producer_leaves = leaves_by_robot.get(producer, [])
        if _producer_index(producer_leaves, producer, condition, scenario) is None:
            report.add(
                "missing_sync_producer",
                f"Producer '{producer}' never creates synchronization condition '{condition}' in its BT."
                f"{_same_name_text(condition, produced_predicates(plan, scenario), suggest_producers)}",
            )

        consumer_leaves = leaves_by_robot.get(consumer, [])
        if _find_condition(consumer_leaves, condition) is None:
            report.add(
                "missing_sync_condition",
                f"Consumer '{consumer}' BT must include Condition '{condition}' for synchronization.",
            )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _find_action(leaves: list[BTNode], name: str | None, parameters: tuple[str, ...]) -> bool:
    return any(
        leaf.type == "Action" and leaf.name == name and tuple(leaf.parameters) == parameters
        for leaf in leaves
    )


def _find_condition(leaves: list[BTNode], condition: str) -> int | None:
    name, args = parse_predicate(condition)
    for index, leaf in enumerate(leaves):
        if leaf.type == "Condition" and leaf.name == name and list(leaf.parameters) == args:
            return index
    return None


def _producer_index(leaves: list[BTNode], robot_id: str, target: str, scenario: Scenario) -> int | None:
    for index, leaf in enumerate(leaves):
        if leaf.type != "Action":
            continue
        capability = scenario.capability(robot_id, leaf.name or "")
        if capability is None:
            continue
        bindings = dict(zip(capability.parameters, leaf.parameters))
        if target in positive_effects(capability.effects, bindings):
            return index
    return None


def _candidate_text(candidates: list[Any], suggest_producers: bool) -> str:
    if not suggest_producers or not candidates:
        return ""
    return f" Candidate producer actions: {', '.join(c.describe() for c in candidates)}."


def _same_name_text(predicate: str, candidates: set[str], suggest_producers: bool) -> str:
    if not suggest_producers:
        return ""
    name, _ = parse_predicate(predicate)
    matches = sorted(c for c in candidates if parse_predicate(c)[0] == name and c != predicate)
    if not matches:
        return ""
    return f" Did you mean: {', '.join(matches[:5])}?"
