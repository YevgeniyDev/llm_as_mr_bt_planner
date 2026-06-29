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

COMPOSITES = {"Sequence", "Fallback", "Parallel"}
LEAVES = {"Action", "Condition"}


class Status(Enum):
    """Standard Behavior Tree tick result."""

    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


@dataclass
class BTNode:
    type: str
    name: str | None = None
    parameters: tuple[str, ...] = ()
    children: list["BTNode"] = field(default_factory=list)
    # Parallel-only: how many children must succeed for the node to succeed.
    success_threshold: int | None = None

    @property
    def is_leaf(self) -> bool:
        return self.type in LEAVES

    def label(self) -> str:
        return format_predicate(self.name or "", self.parameters)

    def to_dict(self) -> dict[str, Any]:
        if self.type in COMPOSITES:
            node: dict[str, Any] = {"type": self.type, "children": [c.to_dict() for c in self.children]}
            if self.type == "Parallel" and self.success_threshold is not None:
                node["success_threshold"] = self.success_threshold
            return node
        return {"type": self.type, "name": self.name, "parameters": list(self.parameters)}


class BTParseError(ValueError):
    """Raised when a behavior-tree dict is malformed enough that it cannot be built."""


def parse_node(data: Any) -> BTNode:
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
            children=[parse_node(child) for child in children],
            success_threshold=data.get("success_threshold"),
        )
    # Leaf (or unknown type - kept verbatim so the validator can flag it).
    parameters = data.get("parameters", [])
    return BTNode(
        type=str(node_type),
        name=data.get("name"),
        parameters=tuple(str(p) for p in parameters) if isinstance(parameters, list) else (),
    )


def iter_leaves(node: BTNode) -> Iterator[BTNode]:
    """Yield every Action/Condition leaf under ``node`` in left-to-right order."""
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
