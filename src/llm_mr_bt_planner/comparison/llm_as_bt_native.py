"""Strict KIOS JSON parsing, native dummy execution, and canonical observation."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..bt import BTNode
from ..domain import Scenario, apply_grounded, ground_effects
from ..predicates import format_predicate, parse_predicate, substitute

_NODE_NAME = re.compile(
    r"^(selector|sequence|parallel|target|precondition|condition|action):\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_EXPRESSION = re.compile(r"^[A-Za-z][A-Za-z0-9_]*\([^()]*\)$")
_COMPOSITES = {"selector", "sequence", "parallel"}
_CONDITIONS = {"target", "precondition", "condition"}


class KiosTreeError(ValueError):
    """Raised when model output is not a valid native KIOS tree."""


@dataclass
class KiosNode:
    kind: str
    body: str
    summary: str
    children: list["KiosNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "summary": self.summary,
            "name": f"{self.kind}: {self.body}",
        }
        if self.kind in _COMPOSITES:
            document["children"] = [child.to_dict() for child in self.children]
        return document


@dataclass(frozen=True)
class NativeExecution:
    result: str
    summary: str
    world_state: tuple[str, ...]
    final_node: str | None
    trace: tuple[dict[str, Any], ...]

    @property
    def success(self) -> bool:
        return self.result == "success"

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "summary": self.summary,
            "world_state": list(self.world_state),
            "final_node": self.final_node,
            "trace": list(self.trace),
        }


def parse_kios_tree(document: Any, *, path: str = "root") -> KiosNode:
    """Parse the exact summary/name/children grammar used by KIOS."""
    if not isinstance(document, dict):
        raise KiosTreeError(f"KIOS node at {path} must be an object.")
    unknown = set(document) - {"summary", "name", "children"}
    if unknown:
        raise KiosTreeError(f"KIOS node at {path} has unknown field(s): {', '.join(sorted(unknown))}.")
    summary = document.get("summary")
    name = document.get("name")
    if not isinstance(summary, str) or not summary.strip():
        raise KiosTreeError(f"KIOS node at {path} requires a non-empty summary.")
    if not isinstance(name, str):
        raise KiosTreeError(f"KIOS node at {path} requires a name string.")
    match = _NODE_NAME.fullmatch(name.strip())
    if match is None:
        raise KiosTreeError(f"KIOS node name at {path} has no recognized type prefix: {name!r}.")
    kind, body = match.group(1).lower(), match.group(2).strip()
    if not body:
        raise KiosTreeError(f"KIOS node at {path} has an empty body.")
    raw_children = document.get("children")
    if kind in _COMPOSITES:
        if not isinstance(raw_children, list) or not raw_children:
            raise KiosTreeError(f"KIOS composite at {path} requires non-empty children.")
        children = [parse_kios_tree(child, path=f"{path}.{index}") for index, child in enumerate(raw_children)]
    else:
        if "children" in document:
            raise KiosTreeError(f"KIOS leaf at {path} must not contain children.")
        children = []
        if _EXPRESSION.fullmatch(body) is None:
            raise KiosTreeError(
                f"KIOS {kind} at {path} must use grounded predicate(...) syntax."
            )
        try:
            parse_predicate(body)
        except (TypeError, ValueError) as error:
            raise KiosTreeError(f"KIOS {kind} at {path} is not a predicate expression: {error}.") from error
    return KiosNode(kind=kind, body=body, summary=summary.strip(), children=children)


def iter_kios_nodes(node: KiosNode) -> Iterator[KiosNode]:
    yield node
    for child in node.children:
        yield from iter_kios_nodes(child)


def action_sequence(node: KiosNode) -> list[str]:
    return [item.body for item in iter_kios_nodes(node) if item.kind == "action"]


def validate_unit_subtrees(node: KiosNode) -> None:
    """Enforce the selector(target, sequence(preconditions..., action)) KIOS unit shape."""
    for item in iter_kios_nodes(node):
        if item.kind != "selector":
            continue
        if len(item.children) != 2:
            raise KiosTreeError("Every KIOS selector unit subtree must have exactly two children.")
        target, sequence = item.children
        if target.kind != "target" or sequence.kind != "sequence":
            raise KiosTreeError("A KIOS selector must contain target first and sequence second.")
        if not sequence.children or sequence.children[-1].kind != "action":
            raise KiosTreeError("A KIOS unit sequence must terminate in an action.")
        if any(
            child.kind not in _CONDITIONS and child.kind != "selector"
            for child in sequence.children[:-1]
        ):
            raise KiosTreeError(
                "Only condition leaves or recursively expanded selector units may precede "
                "a KIOS unit action."
            )


def simulate_kios_tree(
    tree: KiosNode,
    scenario: Scenario,
    robot_id: str,
    world_state: list[str] | tuple[str, ...],
) -> NativeExecution:
    """Execute the authors' dependency-free dummy-simulation semantics."""
    state = set(world_state)
    trace: list[dict[str, Any]] = []
    final: str | None = None

    def tick(node: KiosNode) -> bool:
        nonlocal final
        final = f"{node.kind}: {node.body}"
        if node.kind == "selector":
            return any(tick(child) for child in node.children)
        if node.kind == "sequence":
            return all(tick(child) for child in node.children)
        if node.kind == "parallel":
            return all(tick(child) for child in node.children)
        if node.kind in _CONDITIONS:
            predicate = _canonical_expression(node.body)
            success = predicate in state
            trace.append(
                {"node": final, "event": "condition", "predicate": predicate, "success": success}
            )
            return success
        action_name, arguments = parse_predicate(node.body)
        if not arguments or arguments[0] != robot_id:
            trace.append(
                {"node": final, "event": "action_rejected", "reason": "wrong assigned robot"}
            )
            return False
        capability = scenario.capability(robot_id, action_name)
        action_arguments = arguments[1:]
        if capability is None or len(action_arguments) != len(capability.parameters):
            trace.append(
                {"node": final, "event": "action_rejected", "reason": "unknown action signature"}
            )
            return False
        bindings = dict(zip(capability.parameters, action_arguments))
        missing = [
            substitute(item, bindings)
            for item in capability.preconditions
            if substitute(item, bindings) not in state
        ]
        if missing:
            trace.append(
                {
                    "node": final,
                    "event": "action_rejected",
                    "action": node.body,
                    "missing_preconditions": missing,
                }
            )
            return False
        adds, deletes = ground_effects(capability.effects, bindings)
        apply_grounded(state, adds, deletes)
        trace.append(
            {
                "node": final,
                "event": "action",
                "action": node.body,
                "effects": {"add": adds, "delete": deletes},
            }
        )
        return True

    try:
        success = tick(tree)
        return NativeExecution(
            result="success" if success else "failure",
            summary="behavior tree returned SUCCESS" if success else "behavior tree returned FAILURE",
            world_state=tuple(sorted(state)),
            final_node=final,
            trace=tuple(trace),
        )
    except Exception as error:
        return NativeExecution(
            result="error",
            summary=str(error),
            world_state=tuple(sorted(world_state)),
            final_node=final,
            trace=tuple(trace),
        )


def native_forest_to_plan(
    trees: list[tuple[str, str, KiosNode]],
    scenario: Scenario,
    *,
    wait_timeout_ticks: int,
) -> dict[str, Any]:
    """Observe native trees as per-robot canonical BTs without action repair."""
    builder = _CanonicalBuilder(scenario, wait_timeout_ticks)
    per_robot: dict[str, list[BTNode]] = {robot.id: [] for robot in scenario.robots}
    producers: dict[str, tuple[str, int]] = {}
    producer_version = 0
    for _subgoal_id, robot_id, tree in trees:
        per_robot[robot_id].append(builder.convert(tree, robot_id, producers))
        for node in iter_kios_nodes(tree):
            if node.kind != "action":
                continue
            action, arguments = parse_predicate(node.body)
            capability = scenario.capability(robot_id, action)
            if capability is None or not arguments or arguments[0] != robot_id:
                continue
            producer_version += 1
            bindings = dict(zip(capability.parameters, arguments[1:]))
            for effect in capability.effects.add:
                producers[substitute(effect, bindings)] = (robot_id, producer_version)

    document: dict[str, Any] = {
        "schema_version": "2.0",
        "mission_id": scenario.task_id,
        "behavior_trees": {},
    }
    for robot_id, nodes in per_robot.items():
        if not nodes:
            continue
        root = nodes[0] if len(nodes) == 1 else BTNode(
            type="Sequence",
            node_id=builder.next_id("subgoals"),
            children=nodes,
            source="llm",
        )
        document["behavior_trees"][robot_id] = root.to_dict()
    return document


def clone_tree(tree: KiosNode) -> KiosNode:
    return copy.deepcopy(tree)


class _CanonicalBuilder:
    def __init__(self, scenario: Scenario, wait_timeout_ticks: int) -> None:
        self.scenario = scenario
        self.wait_timeout_ticks = wait_timeout_ticks
        self.counter = 0
        self.synchronized: dict[str, dict[str, int]] = {
            robot.id: {} for robot in scenario.robots
        }
        self.deleted_by: dict[str, set[str]] = {}
        for robot in scenario.robots:
            for capability in robot.capabilities:
                for predicate in capability.effects.delete:
                    self.deleted_by.setdefault(predicate, set()).add(robot.id)

    def next_id(self, label: str) -> str:
        self.counter += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "node"
        return f"kios.{self.counter:04d}.{safe}"

    def convert(
        self,
        node: KiosNode,
        robot_id: str,
        producers: dict[str, tuple[str, int]],
    ) -> BTNode:
        if node.kind in _COMPOSITES:
            node_type = {"selector": "Fallback", "sequence": "ReactiveSequence", "parallel": "ParallelAll"}[node.kind]
            children = [self.convert(child, robot_id, producers) for child in node.children]
            if node.kind == "sequence" and len(children) > 1:
                prefix, final = children[:-1], children[-1]

                def synchronization_key(child: BTNode) -> tuple[int, int]:
                    if child.type != "WaitFor":
                        return 0, -1
                    producer = producers.get(child.label())
                    return 1, producer[1] if producer is not None else -1

                children = [*sorted(prefix, key=synchronization_key), final]
            return BTNode(
                type=node_type,
                node_id=self.next_id(node.kind),
                children=children,
                success_threshold=len(children) if node_type == "ParallelAll" else None,
                source="llm",
            )
        name, parameters = parse_predicate(node.body)
        if node.kind in _CONDITIONS:
            predicate = format_predicate(name, parameters)
            producer = producers.get(predicate)
            mutable_initial = bool(
                predicate in self.scenario.initial_state
                and (self.deleted_by.get(predicate, set()) - {robot_id})
            )
            cross_robot = bool(
                node.kind == "precondition"
                and producer is not None
                and producer[0] != robot_id
                and producer[1] > self.synchronized[robot_id].get(predicate, -1)
                and (predicate not in self.scenario.initial_state or mutable_initial)
            )
            if cross_robot and producer is not None:
                self.synchronized[robot_id][predicate] = producer[1]
            return BTNode(
                type="WaitFor" if cross_robot else "Condition",
                node_id=self.next_id(node.kind),
                name=name,
                parameters=tuple(parameters),
                timeout_ticks=self.wait_timeout_ticks if cross_robot else None,
                source="llm",
            )
        if not parameters or parameters[0] != robot_id:
            raise KiosTreeError(
                f"Action '{node.body}' must use assigned robot '{robot_id}' as its first argument."
            )
        capability = self.scenario.capability(robot_id, name)
        arguments = tuple(parameters[1:])
        if capability is None:
            raise KiosTreeError(f"Robot '{robot_id}' has no capability '{name}'.")
        if len(arguments) != len(capability.parameters):
            raise KiosTreeError(
                f"Action '{node.body}' expects {len(capability.parameters)} domain arguments."
            )
        action_id = self.next_id(name)
        action = BTNode(
            type="Action",
            node_id=action_id,
            task_id=f"{action_id}.task",
            name=name,
            parameters=arguments,
            source="llm",
        )
        if not capability.resources:
            return action
        wrapped: list[BTNode] = []
        for resource in capability.resources:
            wrapped.append(
                BTNode(
                    type="AcquireResource",
                    node_id=self.next_id(f"acquire-{resource}"),
                    name=resource,
                    timeout_ticks=self.wait_timeout_ticks,
                    source="llm",
                )
            )
        wrapped.append(action)
        for resource in reversed(capability.resources):
            wrapped.append(
                BTNode(
                    type="ReleaseResource",
                    node_id=self.next_id(f"release-{resource}"),
                    name=resource,
                    source="llm",
                )
            )
        return BTNode(
            type="Sequence",
            node_id=self.next_id(f"skill-{name}"),
            children=wrapped,
            source="llm",
        )


def _canonical_expression(value: str) -> str:
    name, arguments = parse_predicate(value)
    return format_predicate(name, arguments)
