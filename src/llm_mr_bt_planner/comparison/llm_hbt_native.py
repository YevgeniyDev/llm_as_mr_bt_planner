"""Native condition queue and online subtree construction for LLM-HBT.

LLM-HBT begins with LLM-selected condition nodes, ticks them, and places false
conditions in a failure queue.  Alex assigns a robot, a second LLM decision
selects one action node, and the action's preconditions become further failure
nodes until the condition can be extended.  This module implements those
published state transitions while leaving every semantic choice to the caller.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..bt import BTNode
from ..domain import Scenario, apply_grounded
from ..predicates import canonical_predicate, parse_predicate
from .llm_bt_native import GroundAction, ground_action_templates

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class LLMHBTNativeError(ValueError):
    """Raised when a native LLM-HBT decision violates its supplied library."""


@dataclass(frozen=True)
class Assignment:
    robot: str
    mode: str
    task: str

    def to_dict(self) -> dict[str, str]:
        return {"robot": self.robot, "mode": self.mode, "task": self.task}


@dataclass(frozen=True)
class PlannedAction:
    target_condition: str
    requester: str | None
    assignment: Assignment
    action: GroundAction
    external_preconditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_condition": self.target_condition,
            "requester": self.requester,
            "assignment": self.assignment.to_dict(),
            "action": self.action.to_dict(),
            "external_preconditions": list(self.external_preconditions),
        }


@dataclass
class HBTConstruction:
    trees: dict[str, BTNode]
    initial_conditions: tuple[str, ...]
    actions: list[PlannedAction]
    trace: list[dict[str, Any]]
    failure_queue: list[dict[str, Any]]
    final_planning_state: tuple[str, ...]
    unresolved: list[dict[str, Any]] = field(default_factory=list)


class DecisionInterface(Protocol):
    def assign(
        self,
        failed_condition: str,
        requester: str | None,
        observed_state: set[str],
    ) -> Assignment:
        ...

    def select_action(
        self,
        failed_condition: str,
        assignment: Assignment,
        observed_state: set[str],
    ) -> GroundAction:
        ...


def condition_library(scenario: Scenario) -> list[str]:
    values = set(scenario.initial_state)
    for action in ground_action_templates(scenario):
        values.update(action.preconditions)
        values.update(action.add_effects)
    return sorted(values)


def parse_initialization_response(text: str, scenario: Scenario) -> tuple[str, ...]:
    document = _json_object(text, "task initialization")
    if set(document) != {"conditions"} or not isinstance(document["conditions"], list):
        raise LLMHBTNativeError(
            "LLM-HBT initialization must contain only a 'conditions' string array."
        )
    raw = document["conditions"]
    if not raw or not all(isinstance(item, str) and item.strip() for item in raw):
        raise LLMHBTNativeError("LLM-HBT initialization conditions must be non-empty strings.")
    conditions = tuple(canonical_predicate(item) for item in raw)
    if len(set(conditions)) != len(conditions):
        raise LLMHBTNativeError("LLM-HBT initialization contains duplicate condition nodes.")
    allowed = set(condition_library(scenario))
    unknown = [condition for condition in conditions if condition not in allowed]
    if unknown:
        raise LLMHBTNativeError(
            "LLM-HBT initialization selected condition(s) outside the supplied library: "
            + ", ".join(unknown)
        )
    return conditions


def parse_assignment_response(
    text: str,
    scenario: Scenario,
    *,
    requester: str | None,
) -> Assignment:
    document = _json_object(text, "Alex assignment")
    if set(document) != {"robot", "mode", "task"}:
        raise LLMHBTNativeError("Alex assignment requires exactly robot, mode, and task.")
    robot = document.get("robot")
    mode = document.get("mode")
    task = document.get("task")
    if not isinstance(robot, str) or robot not in scenario.robot_ids:
        raise LLMHBTNativeError(f"Alex selected unknown robot '{robot}'.")
    if mode not in {"local", "delegated"}:
        raise LLMHBTNativeError("Alex mode must be 'local' or 'delegated'.")
    if not isinstance(task, str) or not task.strip():
        raise LLMHBTNativeError("Alex assignment task must be a non-empty string.")
    expected_mode = "local" if requester in {None, robot} else "delegated"
    if mode != expected_mode:
        raise LLMHBTNativeError(
            f"Alex marked assignment as '{mode}', but requester/producer ownership requires "
            f"'{expected_mode}'."
        )
    return Assignment(robot=robot, mode=mode, task=task.strip())


def parse_action_response(
    text: str,
    actions: list[GroundAction],
    *,
    robot: str,
    failed_condition: str,
) -> GroundAction:
    document = _json_object(text, "robot action selection")
    if set(document) != {"action"} or not isinstance(document["action"], str):
        raise LLMHBTNativeError("Robot action selection requires only one action string.")
    name, parameters = parse_predicate(document["action"])
    candidates = [
        action
        for action in actions
        if action.robot == robot and action.name == name and action.parameters == tuple(parameters)
    ]
    if len(candidates) != 1:
        raise LLMHBTNativeError(
            f"Robot {robot} selected unknown or ambiguously grounded action "
            f"'{document['action']}'."
        )
    selected = candidates[0]
    if failed_condition not in selected.add_effects:
        raise LLMHBTNativeError(
            f"Selected action '{document['action']}' does not establish failed condition "
            f"'{failed_condition}'."
        )
    return selected


def construct_forest(
    scenario: Scenario,
    conditions: tuple[str, ...],
    decisions: DecisionInterface,
    *,
    namespace: str = "llmhbt",
    max_extensions: int = 100,
) -> HBTConstruction:
    """Execute the paper's failure-queue/BT-extension loop to a planning fixpoint."""
    if max_extensions < 1:
        raise LLMHBTNativeError("max_extensions must be positive.")
    state = set(scenario.initial_state)
    planned: list[PlannedAction] = []
    trace: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    producers: dict[str, str] = {}
    active: list[str] = []

    def ensure(condition: str, requester: str | None) -> None:
        if condition in state:
            trace.append(
                {
                    "event": "tick_condition_success",
                    "condition": condition,
                    "requester": requester,
                    "support": "observed_or_previously_established",
                }
            )
            return
        if condition in active:
            raise LLMHBTNativeError(
                "LLM-HBT extension encountered a cyclic condition dependency: "
                + " -> ".join([*active, condition])
            )
        if len(planned) >= max_extensions:
            raise LLMHBTNativeError(
                f"LLM-HBT exceeded the {max_extensions}-extension safety bound."
            )
        queue_event = {
            "index": len(queue) + 1,
            "event": "failure_node_detected",
            "condition": condition,
            "requester": requester,
            "status": "queued",
        }
        queue.append(queue_event)
        trace.append(dict(queue_event))
        assignment = decisions.assign(condition, requester, set(state))
        action = decisions.select_action(condition, assignment, set(state))
        active.append(condition)
        external: list[str] = []
        for precondition in action.preconditions:
            if precondition not in state:
                ensure(precondition, action.robot)
            producer = producers.get(precondition)
            if producer is not None and producer != action.robot:
                external.append(precondition)
        active.pop()
        missing = [precondition for precondition in action.preconditions if precondition not in state]
        if missing:
            raise LLMHBTNativeError(
                f"Action {action.name}{action.parameters} remains unsupported after extension: "
                + ", ".join(missing)
            )
        selected = PlannedAction(
            target_condition=condition,
            requester=requester,
            assignment=assignment,
            action=action,
            external_preconditions=tuple(external),
        )
        planned.append(selected)
        apply_grounded(state, list(action.add_effects), list(action.delete_effects))
        for effect in action.add_effects:
            producers[effect] = action.robot
        queue_event["status"] = "resolved"
        queue_event["assigned_robot"] = action.robot
        queue_event["selected_action"] = f"{action.name}({','.join(action.parameters)})"
        trace.append(
            {
                "event": "bt_extension",
                "failed_condition": condition,
                "requester": requester,
                "assignment": assignment.to_dict(),
                "selected_action": action.to_dict(),
                "operation": (
                    "independent_extension"
                    if assignment.mode == "local"
                    else "delegated_root_insertion_and_requester_monitor"
                ),
                "construction": "Selector(condition, Sequence(preconditions, action))",
            }
        )

    for condition in conditions:
        ensure(condition, _condition_owner_hint(condition, scenario))

    trees = _build_canonical_trees(scenario, conditions, planned, producers, namespace)
    return HBTConstruction(
        trees=trees,
        initial_conditions=conditions,
        actions=planned,
        trace=trace,
        failure_queue=queue,
        final_planning_state=tuple(sorted(state)),
    )


def native_forest_document(construction: HBTConstruction) -> dict[str, Any]:
    """Serialize the clean-room native notation separately from common BT JSON."""

    def native(node: BTNode) -> dict[str, Any]:
        if node.type in {"Sequence", "Fallback"}:
            return {
                "node": "Sequence" if node.type == "Sequence" else "Selector",
                "id": node.node_id,
                "children": [native(child) for child in node.children],
            }
        if node.type == "WaitFor":
            return {
                "node": "Monitor",
                "id": node.node_id,
                "condition": node.label(),
                "timeout_ticks": node.timeout_ticks,
            }
        if node.type in {"AcquireResource", "ReleaseResource"}:
            return {"node": node.type, "id": node.node_id, "resource": node.name}
        return {
            "node": node.type,
            "id": node.node_id,
            "label": node.label(),
        }

    return {robot: native(tree) for robot, tree in construction.trees.items()}


def _build_canonical_trees(
    scenario: Scenario,
    conditions: tuple[str, ...],
    planned: list[PlannedAction],
    producers: dict[str, str],
    namespace: str,
) -> dict[str, BTNode]:
    counter = _NodeCounter(namespace)
    children: dict[str, list[BTNode]] = {robot.id: [] for robot in scenario.robots}
    for selected in planned:
        action = selected.action
        steps: list[BTNode] = []
        for precondition in action.preconditions:
            producer = producers.get(precondition)
            if producer is not None and producer != action.robot:
                steps.append(_wait(precondition, counter.next(action.robot, "monitor")))
            else:
                steps.append(_condition(precondition, counter.next(action.robot, "precondition")))
        for resource in sorted(action.resources):
            steps.append(
                BTNode(
                    type="AcquireResource",
                    node_id=counter.next(action.robot, "acquire"),
                    name=resource,
                    timeout_ticks=max(1, action.timeout_ticks),
                    source="llm",
                )
            )
        steps.append(
            BTNode(
                type="Action",
                node_id=counter.next(action.robot, action.name),
                task_id=counter.next(action.robot, "task"),
                name=action.name,
                parameters=action.parameters,
                source="llm",
            )
        )
        for resource in reversed(sorted(action.resources)):
            steps.append(
                BTNode(
                    type="ReleaseResource",
                    node_id=counter.next(action.robot, "release"),
                    name=resource,
                    source="llm",
                )
            )
        branch = BTNode(
            type="Fallback",
            node_id=counter.next(action.robot, "extension"),
            source="llm",
            children=[
                _condition(
                    selected.target_condition,
                    counter.next(action.robot, "target_condition"),
                ),
                BTNode(
                    type="Sequence",
                    node_id=counter.next(action.robot, "extension_sequence"),
                    children=steps,
                    source="llm",
                ),
            ],
        )
        children[action.robot].append(branch)

    for condition in conditions:
        owner = producers.get(condition) or _condition_owner_hint(condition, scenario)
        if owner is None:
            owner = scenario.robots[0].id
        children[owner].append(_condition(condition, counter.next(owner, "final_condition")))

    trees: dict[str, BTNode] = {}
    for robot in scenario.robots:
        if not children[robot.id]:
            ready = f"robot_ready({robot.id})"
            children[robot.id].append(_condition(ready, counter.next(robot.id, "idle")))
        trees[robot.id] = BTNode(
            type="Sequence",
            node_id=f"{namespace}.{robot.id}.root",
            children=children[robot.id],
            source="llm",
        )
    return trees


def _condition_owner_hint(condition: str, scenario: Scenario) -> str | None:
    _name, arguments = parse_predicate(condition)
    return next((argument for argument in arguments if argument in scenario.robot_ids), None)


def _condition(predicate: str, node_id: str) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="Condition",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        source="llm",
    )


def _wait(predicate: str, node_id: str) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="WaitFor",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        timeout_ticks=80,
        source="llm",
    )


def _json_object(text: str, label: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise LLMHBTNativeError(f"LLM-HBT {label} response is empty.")
    match = _FENCE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    try:
        document = json.loads(candidate)
    except json.JSONDecodeError as error:
        raise LLMHBTNativeError(f"LLM-HBT {label} response is not valid JSON: {error}.") from error
    if not isinstance(document, dict):
        raise LLMHBTNativeError(f"LLM-HBT {label} response must be one JSON object.")
    return document


class _NodeCounter:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.value = 0

    def next(self, robot: str, label: str) -> str:
        self.value += 1
        return f"{self.namespace}.{robot}.{self.value:04d}.{label}"
