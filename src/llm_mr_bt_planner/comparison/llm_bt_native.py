"""Common-domain Action Template Library and dynamic expansion for LLM-BT."""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
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


def reachable_action_templates(
    scenario: Scenario,
    actions: list[GroundAction] | None = None,
) -> list[GroundAction]:
    """Return grounded templates whose preconditions are relaxed-reachable."""
    candidates = actions if actions is not None else ground_action_templates(scenario)
    _reachable, enabled, _distances = _relaxed_reachability(scenario, candidates)
    return [action for action in candidates if action in enabled]


def build_alias_catalog(scenario: Scenario) -> list[AliasEntry]:
    """Map released-parser-compatible move phrases to grounded ATL postconditions."""
    candidates: list[tuple[GroundAction, str]] = []
    seen: set[tuple[str, str, tuple[str, ...], str]] = set()
    for action in ground_action_templates(scenario):
        for effect in action.add_effects:
            signature = (action.robot, action.name, action.parameters, effect)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append((action, effect))

    # The released NER checkpoint was trained on short synthetic identifiers.  Give every
    # protocol goal one low-numbered alias before retaining the complete ATL catalog.  This
    # changes only the natural-language aliases, not action selection or BT expansion.
    ordered: list[tuple[GroundAction, str]] = []
    selected_effects: set[str] = set()
    for goal in scenario.goal_state:
        goal_candidates = [item for item in candidates if item[1] == goal]
        if not goal_candidates:
            continue
        action, effect = min(
            goal_candidates,
            key=lambda item: (
                item[0].timeout_ticks,
                item[0].robot,
                item[0].name,
                item[0].parameters,
            ),
        )
        ordered.append((action, effect))
        selected_effects.add(effect)
    for action, effect in candidates:
        if effect in selected_effects:
            continue
        ordered.append((action, effect))
        selected_effects.add(effect)

    catalog: list[AliasEntry] = []
    for index, (action, effect) in enumerate(ordered, start=1):
        # Use a grid of short numeric aliases instead of ever larger numerals.  The released
        # model recognizes these training-like tokens much more reliably, while the pair is
        # still a unique key for every catalog entry.
        target_number = ((index - 1) % 9) + 1
        destination_number = ((index - 1) // 9) + 11
        target = f"object_{target_number}"
        destination = f"position_{destination_number}"
        catalog.append(
            AliasEntry(
                target=target,
                destination=destination,
                phrase=f"Move object {target_number} to position {destination_number}.",
                robot=action.robot,
                predicate=effect,
                action=action.name,
                parameters=action.parameters,
            )
        )
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
    chronological = chronological_action_plan(
        scenario,
        [goal.predicate for goal in goals],
        actions,
    )
    if chronological is not None:
        chronological_trees, construction_trace = materialize_chronological_expansion(
            scenario,
            chronological,
            node_namespace=node_namespace,
        )
        return ExpansionResult(
            trees=chronological_trees,
            initial_trees=initial_trees,
            assigned_goals=tuple(assigned_goals),
            trace=[*assignment_trace, *construction_trace],
            unresolved=[],
            relaxed_reachable=tuple(sorted(reachable)),
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


def chronological_action_plan(
    scenario: Scenario,
    predicates: list[str],
    actions: list[GroundAction],
    *,
    max_expansions: int = 100_000,
) -> tuple[GroundAction, ...] | None:
    """Find a feasible ATL trace while respecting non-monotonic deploy/stow effects."""
    goals = frozenset(predicates)
    initial = frozenset(scenario.initial_state)
    if goals.issubset(initial):
        return ()

    relevant = set(goals)
    changed = True
    while changed:
        changed = False
        for action in actions:
            if not relevant.intersection(action.add_effects):
                continue
            for predicate in (*action.preconditions, *action.add_effects, *action.delete_effects):
                if predicate not in relevant:
                    relevant.add(predicate)
                    changed = True
    candidates = sorted(
        (action for action in actions if relevant.intersection(action.add_effects)),
        key=lambda action: _action_grounding_preference(scenario, action),
    )

    serial = count()
    frontier: list[tuple[int, int, int, frozenset[str], tuple[GroundAction, ...]]] = []
    heappush(frontier, (len(goals - initial), 0, next(serial), initial, ()))
    best_cost: dict[frozenset[str], int] = {initial: 0}
    expansions = 0
    while frontier and expansions < max_expansions:
        _estimate, cost, _serial, state, path = heappop(frontier)
        if best_cost.get(state) != cost:
            continue
        if goals.issubset(state):
            return path
        expansions += 1
        for action in candidates:
            if not set(action.preconditions).issubset(state):
                continue
            following = set(state)
            apply_grounded(following, list(action.add_effects), list(action.delete_effects))
            following_state = frozenset(following)
            if following_state == state:
                continue
            following_cost = cost + 1
            if following_cost >= best_cost.get(following_state, 10**9):
                continue
            best_cost[following_state] = following_cost
            heuristic = len(goals - following_state)
            heappush(
                frontier,
                (
                    following_cost + heuristic,
                    following_cost,
                    next(serial),
                    following_state,
                    (*path, action),
                ),
            )
    return None


def _action_grounding_preference(
    scenario: Scenario,
    action: GroundAction,
) -> tuple[int, int, str, str, tuple[str, ...]]:
    capability = scenario.capability(action.robot, action.name)
    parameter_names = capability.parameters if capability is not None else ()
    mismatch = sum(
        0 if name.lower() in value.lower().split("_") else 1
        for name, value in zip(parameter_names, action.parameters)
    )
    return mismatch, action.timeout_ticks, action.robot, action.name, action.parameters


def materialize_chronological_expansion(
    scenario: Scenario,
    plan: tuple[GroundAction, ...],
    *,
    node_namespace: str,
    source: str = "llm",
) -> tuple[dict[str, BTNode], list[dict[str, Any]]]:
    """Compile one feasible ATL trace into partitioned reactive method-native units."""
    counter = _NodeCounter(node_namespace)
    per_robot: dict[str, list[BTNode]] = {robot.id: [] for robot in scenario.robots}
    trace: list[dict[str, Any]] = []
    state = set(scenario.initial_state)
    last_add: dict[str, tuple[int, str]] = {}
    synchronized: dict[str, dict[str, int]] = {
        robot.id: {} for robot in scenario.robots
    }
    deleted_by: dict[str, set[str]] = {}
    for action in plan:
        for predicate in action.delete_effects:
            deleted_by.setdefault(predicate, set()).add(action.robot)

    for plan_index, action in enumerate(plan):
        before = set(state)
        precondition_nodes: list[BTNode] = []
        wait_predicates: list[str] = []
        action_robot = action.robot

        def synchronization_key(
            predicate: str,
            robot_id: str = action_robot,
        ) -> tuple[int, int]:
            writer = last_add.get(predicate)
            if writer is None or writer[1] == robot_id:
                return 0, -1
            return 1, writer[0]

        # Preconditions supplied by different robots are awaited in causal production order.
        # This matters when an initially true mutable fact (for example ``stowed``) is later
        # invalidated and restored: its wait must follow the scan that triggers restoration.
        ordered_preconditions = sorted(action.preconditions, key=synchronization_key)
        for predicate in ordered_preconditions:
            writer = last_add.get(predicate)
            mutable_initial = bool(
                predicate in scenario.initial_state
                and (deleted_by.get(predicate, set()) - {action.robot})
            )
            needs_wait = bool(
                writer is not None
                and writer[1] != action.robot
                and writer[0] > synchronized[action.robot].get(predicate, -1)
                and (predicate not in scenario.initial_state or mutable_initial)
            )
            if needs_wait:
                assert writer is not None
                precondition_nodes.append(
                    _wait(
                        predicate,
                        counter.next(action.robot, "wait"),
                        timeout_ticks=160,
                        source=source,
                    )
                )
                synchronized[action.robot][predicate] = writer[0]
                wait_predicates.append(predicate)
            else:
                precondition_nodes.append(
                    _condition(
                        predicate,
                        counter.next(action.robot, "precondition"),
                        source=source,
                    )
                )

        newly_true = [effect for effect in action.add_effects if effect not in before]
        target = next(
            (effect for effect in newly_true if effect in scenario.goal_state),
            newly_true[0] if newly_true else action.add_effects[0],
        )
        sequence_children = list(precondition_nodes)
        for resource in sorted(action.resources):
            sequence_children.append(
                BTNode(
                    type="AcquireResource",
                    node_id=counter.next(action.robot, "acquire"),
                    name=resource,
                    timeout_ticks=max(1, action.timeout_ticks),
                    source=source,
                )
            )
        sequence_children.append(
            BTNode(
                type="Action",
                node_id=counter.next(action.robot, action.name),
                task_id=counter.next(action.robot, "task"),
                name=action.name,
                parameters=action.parameters,
                source=source,
            )
        )
        for resource in reversed(sorted(action.resources)):
            sequence_children.append(
                BTNode(
                    type="ReleaseResource",
                    node_id=counter.next(action.robot, "release"),
                    name=resource,
                    source=source,
                )
            )
        unit = BTNode(
            type="Fallback",
            node_id=counter.next(action.robot, "fallback"),
            children=[
                _condition(
                    target,
                    counter.next(action.robot, "condition"),
                    source=source,
                ),
                BTNode(
                    type="Sequence",
                    node_id=counter.next(action.robot, "action_template"),
                    children=sequence_children,
                    source=source,
                ),
            ],
            source=source,
        )
        per_robot[action.robot].append(unit)
        trace.append(
            {
                "event": "expand",
                "global_order": plan_index + 1,
                "robot": action.robot,
                "failed_condition": target,
                "producer_templates": [action.to_dict()],
                "cross_robot_waits": wait_predicates,
                "construction": "Fallback(condition, Sequence(preconditions, action))",
                "ordering": "state-aware ATL fixpoint materialization",
            }
        )
        apply_grounded(state, list(action.add_effects), list(action.delete_effects))
        for effect in action.add_effects:
            last_add[effect] = (plan_index, action.robot)

    trees: dict[str, BTNode] = {}
    for robot in scenario.robots:
        children = per_robot[robot.id]
        if not children:
            children = [
                _condition(
                    f"robot_ready({robot.id})",
                    counter.next(robot.id, "idle_condition"),
                    source=source,
                )
            ]
        trees[robot.id] = BTNode(
            type="Sequence",
            node_id=f"{node_namespace}.expanded.{robot.id}.root",
            children=children,
            source=source,
        )
    return trees, trace


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


def _wait(
    predicate: str,
    node_id: str,
    timeout_ticks: int,
    *,
    source: str = "llm",
) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="WaitFor",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        timeout_ticks=timeout_ticks,
        source=source,
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
