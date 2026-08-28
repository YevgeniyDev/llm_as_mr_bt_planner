"""Common-domain Action Template Library and dynamic expansion for LLM-BT."""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from ..bt import BTNode
from ..domain import Scenario, apply_grounded, ground_effects
from ..predicates import parse_predicate, substitute
from .llm_bt_parser import ParsedMove


class LLMBTNativeError(ValueError):
    """Raised when native reasoning/parser output cannot address the adapted ATL."""


@dataclass(frozen=True)
class GroundAction:
    robot: str
    name: str
    parameters: tuple[str, ...]
    preconditions: tuple[str, ...]
    add_effects: tuple[str, ...]
    delete_effects: tuple[str, ...]
    resources: tuple[str, ...]
    timeout_ticks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "name": self.name,
            "parameters": list(self.parameters),
            "preconditions": list(self.preconditions),
            "effects": {
                "add": list(self.add_effects),
                "delete": list(self.delete_effects),
            },
            "resources": list(self.resources),
            "timeout_ticks": self.timeout_ticks,
        }


@dataclass(frozen=True)
class AliasEntry:
    target: str
    destination: str
    phrase: str
    robot: str
    predicate: str
    action: str
    parameters: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str]:
        return self.target, self.destination

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "destination": self.destination,
            "phrase": self.phrase,
            "robot": self.robot,
            "predicate": self.predicate,
            "backing_action": self.action,
            "backing_parameters": list(self.parameters),
        }


@dataclass(frozen=True)
class ParsedGoal:
    order: int
    robot: str
    predicate: str
    target: str
    destination: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "robot": self.robot,
            "predicate": self.predicate,
            "alias": {"target": self.target, "destination": self.destination},
        }


@dataclass(frozen=True)
class ExpansionResult:
    trees: dict[str, BTNode]
    initial_trees: dict[str, BTNode]
    assigned_goals: tuple[ParsedGoal, ...]
    trace: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]
    relaxed_reachable: tuple[str, ...]


def ground_action_templates(scenario: Scenario) -> list[GroundAction]:
    """Ground the manually supplied common capability contracts into an ATL."""
    templates: list[GroundAction] = []
    for robot in scenario.robots:
        for capability in robot.capabilities:
            domains = [
                _values_for_type(scenario, expected_type)
                for expected_type in capability.parameter_types
            ]
            if any(not values for values in domains):
                continue
            assignments = itertools.product(*domains) if domains else [()]
            for parameters in assignments:
                bindings = dict(zip(capability.parameters, parameters))
                adds, deletes = ground_effects(capability.effects, bindings)
                templates.append(
                    GroundAction(
                        robot=robot.id,
                        name=capability.name,
                        parameters=tuple(parameters),
                        preconditions=tuple(
                            substitute(predicate, bindings)
                            for predicate in capability.preconditions
                        ),
                        add_effects=tuple(adds),
                        delete_effects=tuple(deletes),
                        resources=capability.resources,
                        timeout_ticks=capability.timeout_ticks,
                    )
                )
    return templates


def build_alias_catalog(scenario: Scenario) -> list[AliasEntry]:
    """Map released-parser-compatible move phrases to grounded ATL postconditions."""
    catalog: list[AliasEntry] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    index = 1
    for action in ground_action_templates(scenario):
        for effect in action.add_effects:
            signature = (action.robot, action.name, action.parameters, effect)
            if signature in seen:
                continue
            seen.add(signature)
            target = f"object_{index}"
            destination = f"position_{index + 10}"
            catalog.append(
                AliasEntry(
                    target=target,
                    destination=destination,
                    phrase=f"Move object {index} to position {index + 10}.",
                    robot=action.robot,
                    predicate=effect,
                    action=action.name,
                    parameters=action.parameters,
                )
            )
            index += 1
    return catalog


def map_moves_to_goals(
    moves: list[ParsedMove],
    catalog: list[AliasEntry],
) -> list[ParsedGoal]:
    lookup = {entry.key: entry for entry in catalog}
    goals: list[ParsedGoal] = []
    for order, move in enumerate(moves, start=1):
        entry = lookup.get((move.target, move.destination))
        if entry is None:
            raise LLMBTNativeError(
                "BERT parser produced an alias outside the supplied Action Template Library: "
                f"{move.target}, {move.destination}."
            )
        goals.append(
            ParsedGoal(
                order=order,
                robot=entry.robot,
                predicate=entry.predicate,
                target=entry.target,
                destination=entry.destination,
            )
        )
    return goals


def semantic_map_xml(scenario: Scenario) -> str:
    """Represent the common symbolic observation in the paper's XML map medium."""
    root = ET.Element("semantic_map", {"task_id": scenario.task_id})
    entities = ET.SubElement(root, "entities")
    for entity in scenario.entities:
        ET.SubElement(entities, "entity", {"id": entity.id, "type": entity.type})
    robots = ET.SubElement(root, "robots")
    for robot in scenario.robots:
        ET.SubElement(robots, "robot", {"id": robot.id, "type": robot.type})
    facts = ET.SubElement(root, "observed_facts")
    for predicate in scenario.initial_state:
        ET.SubElement(facts, "fact", {"predicate": predicate})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def build_initial_trees(
    scenario: Scenario,
    goals: list[ParsedGoal],
    *,
    node_namespace: str = "llmbt",
) -> dict[str, BTNode]:
    """Construct the released architecture's sequence of parsed goal conditions."""
    by_robot: dict[str, list[str]] = {robot.id: [] for robot in scenario.robots}
    for goal in goals:
        by_robot.setdefault(goal.robot, []).append(goal.predicate)
    trees: dict[str, BTNode] = {}
    for robot in scenario.robots:
        conditions = [
            _condition(
                predicate,
                f"{node_namespace}.initial.{robot.id}.{index:04d}",
                source="llm",
            )
            for index, predicate in enumerate(by_robot[robot.id], start=1)
        ]
        trees[robot.id] = BTNode(
            type="Sequence",
            node_id=f"{node_namespace}.initial.{robot.id}.root",
            children=conditions,
            source="llm",
        )
    return trees


def expand_initial_trees(
    scenario: Scenario,
    goals: list[ParsedGoal],
    *,
    node_namespace: str = "llmbt",
    grounded_actions: list[GroundAction] | None = None,
) -> ExpansionResult:
    """Compute the tree fixpoint produced by repeated failed-condition expansion."""
    initial_trees = build_initial_trees(
        scenario,
        goals,
        node_namespace=node_namespace,
    )
    actions = grounded_actions if grounded_actions is not None else ground_action_templates(scenario)
    reachable, reachable_actions, distances = _relaxed_reachability(scenario, actions)
    assigned_goals, assignment_trace = _derive_external_goal_assignments(
        scenario,
        goals,
        actions,
        reachable_actions,
        distances,
    )
    trace: list[dict[str, Any]] = assignment_trace
    unresolved: list[dict[str, Any]] = []
    counter = _NodeCounter(node_namespace)
    by_robot: dict[str, list[str]] = {robot.id: [] for robot in scenario.robots}
    for goal in assigned_goals:
        by_robot.setdefault(goal.robot, []).append(goal.predicate)

    trees: dict[str, BTNode] = {}
    for robot in scenario.robots:
        known_state = set(scenario.initial_state)
        expanded_children = []
        for predicate in by_robot[robot.id]:
            expanded_children.append(
                _expand_condition(
                    scenario,
                    robot.id,
                    predicate,
                    actions,
                    reachable_actions,
                    distances,
                    counter,
                    trace,
                    unresolved,
                    stack=(),
                    known_state=known_state,
                )
            )
        if not expanded_children:
            idle_predicate = f"robot_ready({robot.id})"
            expanded_children.append(
                _condition(
                    idle_predicate,
                    counter.next(robot.id, "idle_condition"),
                    source="llm",
                )
            )
            trace.append(
                {
                    "event": "common_idle_tree",
                    "robot": robot.id,
                    "condition": idle_predicate,
                    "reason": "the native single-robot method assigned no remaining goal to this team member",
                }
            )
        expanded_children = _apply_root_insert_rule(expanded_children, trace, robot.id)
        trees[robot.id] = BTNode(
            type="Sequence",
            node_id=f"{node_namespace}.expanded.{robot.id}.root",
            children=expanded_children,
            source="llm",
        )
    return ExpansionResult(
        trees=trees,
        initial_trees=initial_trees,
        assigned_goals=tuple(assigned_goals),
        trace=trace,
        unresolved=unresolved,
        relaxed_reachable=tuple(sorted(reachable)),
    )


def _derive_external_goal_assignments(
    scenario: Scenario,
    seed_goals: list[ParsedGoal],
    actions: list[GroundAction],
    reachable_actions: set[GroundAction],
    distances: dict[str, int],
) -> tuple[list[ParsedGoal], list[dict[str, Any]]]:
    """Partition a single-robot ATL dependency chain over the common robot team."""
    assigned = list(seed_goals)
    trace: list[dict[str, Any]] = []
    seen = {(goal.robot, goal.predicate) for goal in assigned}

    def visit(robot_id: str, predicate: str, stack: tuple[tuple[str, str], ...]) -> None:
        key = (robot_id, predicate)
        if key in stack or predicate in scenario.initial_state:
            return
        local = [
            action
            for action in actions
            if action.robot == robot_id
            and predicate in action.add_effects
            and action in reachable_actions
        ]
        if not local:
            return
        selected = min(
            local,
            key=lambda action: (
                max((distances.get(pre, 10**6) for pre in action.preconditions), default=0),
                action.name,
                action.parameters,
            ),
        )
        for precondition in selected.preconditions:
            if precondition in scenario.initial_state:
                continue
            local_producer = any(
                action.robot == robot_id
                and precondition in action.add_effects
                and action in reachable_actions
                for action in actions
            )
            if local_producer:
                visit(robot_id, precondition, (*stack, key))
                continue
            external = [
                action
                for action in actions
                if action.robot != robot_id
                and precondition in action.add_effects
                and action in reachable_actions
            ]
            if not external:
                continue
            producer = min(
                external,
                key=lambda action: (
                    max((distances.get(pre, 10**6) for pre in action.preconditions), default=0),
                    action.robot,
                    action.name,
                    action.parameters,
                ),
            )
            producer_key = (producer.robot, precondition)
            if producer_key not in seen:
                derived = ParsedGoal(
                    order=len(assigned) + 1,
                    robot=producer.robot,
                    predicate=precondition,
                    target="derived_external_precondition",
                    destination=f"consumer_{robot_id}",
                )
                assigned.append(derived)
                seen.add(producer_key)
                trace.append(
                    {
                        "event": "partition_external_goal",
                        "consumer_robot": robot_id,
                        "predicate": precondition,
                        "producer_robot": producer.robot,
                        "backing_action": producer.name,
                        "adaptation": "single-robot ATL dependency assigned to its common-domain owner",
                    }
                )
            visit(producer.robot, precondition, (*stack, key))

    index = 0
    while index < len(assigned):
        goal = assigned[index]
        visit(goal.robot, goal.predicate, ())
        index += 1
    return assigned, trace


def _expand_condition(
    scenario: Scenario,
    robot_id: str,
    predicate: str,
    all_actions: list[GroundAction],
    reachable_actions: set[GroundAction],
    distances: dict[str, int],
    counter: _NodeCounter,
    trace: list[dict[str, Any]],
    unresolved: list[dict[str, Any]],
    *,
    stack: tuple[str, ...],
    known_state: set[str],
) -> BTNode:
    condition = _condition(predicate, counter.next(robot_id, "condition"), source="llm")
    if predicate in known_state:
        return condition
    if predicate in stack:
        unresolved.append(
            {"robot": robot_id, "predicate": predicate, "reason": "cyclic_action_template_dependency"}
        )
        return condition

    local = [
        action
        for action in all_actions
        if action.robot == robot_id
        and predicate in action.add_effects
        and action in reachable_actions
    ]
    if local:
        minimum = min(
            max((distances.get(precondition, 10**6) for precondition in action.preconditions), default=0)
            for action in local
        )
        candidates = [
            action
            for action in local
            if max(
                (distances.get(precondition, 10**6) for precondition in action.preconditions),
                default=0,
            )
            == minimum
        ]
        candidates = candidates[:1]
        branches = [condition]
        for action in candidates:
            children: list[BTNode] = []
            for precondition in action.preconditions:
                if precondition in known_state:
                    children.append(
                        _condition(
                            precondition,
                            counter.next(robot_id, "precondition"),
                            source="llm",
                        )
                    )
                    continue
                local_producer = any(
                    candidate.robot == robot_id
                    and precondition in candidate.add_effects
                    and candidate in reachable_actions
                    for candidate in all_actions
                )
                external_producer = any(
                    candidate.robot != robot_id
                    and precondition in candidate.add_effects
                    and candidate in reachable_actions
                    for candidate in all_actions
                )
                if not local_producer and external_producer:
                    children.append(
                        _wait(precondition, counter.next(robot_id, "wait"), timeout_ticks=80)
                    )
                    known_state.add(precondition)
                else:
                    children.append(
                        _expand_condition(
                            scenario,
                            robot_id,
                            precondition,
                            all_actions,
                            reachable_actions,
                            distances,
                            counter,
                            trace,
                            unresolved,
                            stack=(*stack, predicate),
                            known_state=known_state,
                        )
                    )
            for resource in sorted(action.resources):
                children.append(
                    BTNode(
                        type="AcquireResource",
                        node_id=counter.next(robot_id, "acquire"),
                        name=resource,
                        timeout_ticks=max(1, action.timeout_ticks),
                        source="llm",
                    )
                )
            children.append(
                BTNode(
                    type="Action",
                    node_id=counter.next(robot_id, action.name),
                    task_id=counter.next(robot_id, "task"),
                    name=action.name,
                    parameters=action.parameters,
                    source="llm",
                )
            )
            for resource in reversed(sorted(action.resources)):
                children.append(
                    BTNode(
                        type="ReleaseResource",
                        node_id=counter.next(robot_id, "release"),
                        name=resource,
                        source="llm",
                    )
                )
            apply_grounded(
                known_state,
                list(action.add_effects),
                list(action.delete_effects),
            )
            branches.append(
                BTNode(
                    type="Sequence",
                    node_id=counter.next(robot_id, "action_template"),
                    children=children,
                    source="llm",
                )
            )
        trace.append(
            {
                "event": "expand",
                "robot": robot_id,
                "failed_condition": predicate,
                "producer_templates": [action.to_dict() for action in candidates],
                "construction": "Fallback(condition, Sequence(preconditions, action))",
            }
        )
        return BTNode(
            type="Fallback",
            node_id=counter.next(robot_id, "fallback"),
            children=branches,
            source="llm",
        )

    external = [
        action
        for action in all_actions
        if action.robot != robot_id
        and predicate in action.add_effects
        and action in reachable_actions
    ]
    if external:
        known_state.add(predicate)
        trace.append(
            {
                "event": "expand_external_condition",
                "robot": robot_id,
                "failed_condition": predicate,
                "producer_robots": sorted({action.robot for action in external}),
                "adaptation": "bounded WaitFor supplied as a common-domain ATL primitive",
            }
        )
        return BTNode(
            type="Fallback",
            node_id=counter.next(robot_id, "external_fallback"),
            children=[condition, _wait(predicate, counter.next(robot_id, "wait"), 80)],
            source="llm",
        )

    unresolved.append(
        {"robot": robot_id, "predicate": predicate, "reason": "no_reachable_action_template"}
    )
    trace.append(
        {
            "event": "expand_failed",
            "robot": robot_id,
            "failed_condition": predicate,
            "reason": "no reachable ATL action has the condition as a postcondition",
        }
    )
    return condition


def _relaxed_reachability(
    scenario: Scenario,
    actions: list[GroundAction],
) -> tuple[set[str], set[GroundAction], dict[str, int]]:
    reachable = set(scenario.initial_state)
    distances = {predicate: 0 for predicate in reachable}
    enabled: set[GroundAction] = set()
    changed = True
    while changed:
        changed = False
        for action in actions:
            if not set(action.preconditions).issubset(reachable):
                continue
            enabled.add(action)
            distance = 1 + max(
                (distances[precondition] for precondition in action.preconditions),
                default=0,
            )
            for effect in action.add_effects:
                if effect not in reachable:
                    reachable.add(effect)
                    distances[effect] = distance
                    changed = True
                else:
                    distances[effect] = min(distances[effect], distance)
    return reachable, enabled, distances


def _apply_root_insert_rule(
    children: list[BTNode],
    trace: list[dict[str, Any]],
    robot_id: str,
) -> list[BTNode]:
    """Record Insert conflicts while preserving the common partition's goal order."""
    result: list[BTNode] = []
    for child in children:
        child_goal = _guard_predicate(child)
        precondition_types = _action_precondition_types(child)
        conflict_index = next(
            (
                index
                for index, existing in enumerate(result)
                if _predicate_name(_guard_predicate(existing)) in precondition_types
            ),
            None,
        )
        if conflict_index is None:
            result.append(child)
            trace.append(
                {
                    "event": "insert",
                    "robot": robot_id,
                    "condition": child_goal,
                    "mode": "replace_failed_condition",
                }
            )
        else:
            result.append(child)
            trace.append(
                {
                    "event": "insert",
                    "robot": robot_id,
                    "condition": child_goal,
                    "mode": "conflict_detected_preserve_partitioned_goal_order",
                    "target_index": conflict_index,
                    "adaptation": (
                        "the paper's central-tree priority move is recorded but not applied across "
                        "partitioned per-robot common trees"
                    ),
                }
            )
    return result


def _guard_predicate(node: BTNode) -> str:
    selected = node.children[0] if node.type == "Fallback" and node.children else node
    return selected.label() if selected.type == "Condition" else ""


def _action_precondition_types(node: BTNode) -> set[str]:
    types: set[str] = set()
    for child in _walk(node):
        if child.type == "Condition" and child is not (node.children[0] if node.children else None):
            types.add(_predicate_name(child.label()))
    return types


def _walk(node: BTNode):
    yield node
    for child in node.children:
        yield from _walk(child)


def _predicate_name(predicate: str) -> str:
    if not predicate:
        return ""
    return parse_predicate(predicate)[0]


def _condition(predicate: str, node_id: str, *, source: str) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="Condition",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        source=source,
    )


def _wait(predicate: str, node_id: str, timeout_ticks: int) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="WaitFor",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        timeout_ticks=timeout_ticks,
        source="llm",
    )


def _values_for_type(scenario: Scenario, expected_type: str) -> tuple[str, ...]:
    if expected_type == "robot":
        return tuple(robot.id for robot in scenario.robots)
    values = tuple(entity.id for entity in scenario.entities if entity.type == expected_type)
    if values:
        return values
    return tuple(sorted(scenario.constants)) if not expected_type else ()


class _NodeCounter:
    def __init__(self, namespace: str = "llmbt") -> None:
        self.value = 0
        self.namespace = namespace

    def next(self, robot: str, kind: str) -> str:
        self.value += 1
        return f"{self.namespace}.{robot}.{self.value:05d}.{kind}"
