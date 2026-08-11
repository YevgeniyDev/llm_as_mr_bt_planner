"""Behavior Tree node model.

The model is a real tree (not the flat node list of the original prototype) so
that it can express the composites a robot executor needs - ``Sequence``,
``Fallback`` (a.k.a. Selector), and ``Parallel`` - alongside ``Action`` and
``Condition`` leaves. Tick semantics live in :mod:`llm_mr_bt_planner.simulation`; this module
only defines the data structure, parsing, iteration, and export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from .predicates import format_predicate

COMPOSITES = {"Sequence", "ReactiveSequence", "Fallback", "Parallel", "ParallelAll"}
LEAVES = {"Action", "Condition", "WaitFor", "AcquireResource", "ReleaseResource"}


class Status(Enum):
    """Standard Behavior Tree tick result."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


@dataclass
class BTNode:
    type: str
    node_id: str | None = None
    name: str | None = None
    parameters: tuple[str, ...] = ()
    children: list["BTNode"] = field(default_factory=list)
    # Parallel-only: how many children must succeed for the node to succeed.
    success_threshold: int | None = None
    # Action-only traceability and WaitFor-only bounded waiting.
    task_id: str | None = None
    timeout_ticks: int | None = None
    id_generated: bool = False
    required_resources: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    expected_postconditions: tuple[str, ...] = ()
    duration_ticks: int = 1
    recovery_policy: str = "none"
    source: str | None = None
    contract_explicit: bool = False

    @property
    def is_leaf(self) -> bool:
        return self.type in LEAVES

    def label(self) -> str:
        return format_predicate(self.name or "", self.parameters)

    def to_dict(self) -> dict[str, Any]:
        if self.type in COMPOSITES:
            node: dict[str, Any] = {"type": self.type, "children": [c.to_dict() for c in self.children]}
            if not self.id_generated and self.node_id is not None:
                node["id"] = self.node_id
            if self.type in {"Parallel", "ParallelAll"} and self.success_threshold is not None:
                node["success_threshold"] = self.success_threshold
            if self.source is not None:
                node["source"] = self.source
            return node
        node = {"type": self.type, "name": self.name, "parameters": list(self.parameters)}
        if not self.id_generated and self.node_id is not None:
            node["id"] = self.node_id
        if self.type == "Action" and self.task_id is not None:
            node["task_id"] = self.task_id
        if self.type in {"Action", "WaitFor", "AcquireResource"} and self.timeout_ticks is not None:
            node["timeout_ticks"] = self.timeout_ticks
        if self.type == "Action" and self.contract_explicit:
            node["required_resources"] = list(self.required_resources)
            node["preconditions"] = list(self.preconditions)
            node["expected_postconditions"] = list(self.expected_postconditions)
            node["duration_ticks"] = self.duration_ticks
            node["recovery_policy"] = self.recovery_policy
        if self.source is not None:
            node["source"] = self.source
        return node


class BTParseError(ValueError):
    """Raised when a behavior-tree dict is malformed enough that it cannot be built."""


def parse_node(data: Any, path: str = "root") -> BTNode:
    """Build a :class:`BTNode` from plan JSON. Tolerant by design: structural
    problems are recorded by the validator, not raised here, so that a slightly
    malformed LLM plan can still be reported in full. Only completely
    un-parseable input raises.
    """
    if not isinstance(data, dict):
        raise BTParseError(f"Behavior-tree node must be an object, got {type(data).__name__}.")
    node_type = data.get("type")
    if node_type in COMPOSITES:
        children = data.get("children", [])
        if not isinstance(children, list):
            raise BTParseError(f"{node_type}.children must be a list.")
        return BTNode(
            type=node_type,
            node_id=str(data.get("id") or path),
            id_generated=not bool(data.get("id")),
            children=[parse_node(child, f"{path}.{index}") for index, child in enumerate(children)],
            success_threshold=data.get("success_threshold"),
            source=str(data["source"]) if data.get("source") is not None else None,
        )
    # Leaf (or unknown type - kept verbatim so the validator can flag it).
    parameters = data.get("parameters", [])
    return BTNode(
        type=str(node_type),
        node_id=str(data.get("id") or path),
        id_generated=not bool(data.get("id")),
        name=data.get("name"),
        parameters=tuple(str(p) for p in parameters) if isinstance(parameters, list) else (),
        task_id=str(data["task_id"]) if data.get("task_id") is not None else None,
        timeout_ticks=data.get("timeout_ticks"),
        required_resources=tuple(str(item) for item in data.get("required_resources", []))
        if isinstance(data.get("required_resources", []), list)
        else (),
        preconditions=tuple(str(item) for item in data.get("preconditions", []))
        if isinstance(data.get("preconditions", []), list)
        else (),
        expected_postconditions=tuple(str(item) for item in data.get("expected_postconditions", []))
        if isinstance(data.get("expected_postconditions", []), list)
        else (),
        duration_ticks=data.get("duration_ticks", 1),
        recovery_policy=str(data.get("recovery_policy", "none")),
        source=str(data["source"]) if data.get("source") is not None else None,
        contract_explicit=any(
            key in data
            for key in (
                "required_resources",
                "preconditions",
                "expected_postconditions",
                "duration_ticks",
                "timeout_ticks",
                "recovery_policy",
            )
        ),
    )


def iter_leaves(node: BTNode) -> Iterator[BTNode]:
    """Yield every Action/Condition/WaitFor leaf in left-to-right order."""
    if node.is_leaf:
        yield node
        return
    for child in node.children:
        yield from iter_leaves(child)


def iter_nodes(node: BTNode) -> Iterator[BTNode]:
    """Yield every node (composites and leaves), pre-order."""
    yield node
    for child in node.children:
        yield from iter_nodes(child)
