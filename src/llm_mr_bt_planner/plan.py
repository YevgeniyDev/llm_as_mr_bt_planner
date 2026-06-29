"""Typed view over the JSON plan returned by the LLM.

The plan is kept close to its wire form (task graph / assignments /
synchronization are lists of plain dicts) because the validator needs to report
on malformed entries. Behavior trees are parsed into :class:`llm_mr_bt_planner.bt.BTNode`
trees up front, since both the validator and the simulator consume them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bt import BTNode, BTParseError, parse_node


@dataclass
class Plan:
    task_graph: list[dict[str, Any]] = field(default_factory=list)
    assignments: list[dict[str, Any]] = field(default_factory=list)
    synchronization: list[dict[str, Any]] = field(default_factory=list)
    behavior_trees: dict[str, BTNode] = field(default_factory=dict)
    # Robots whose BT could not be parsed at all (validator reports these).
    unparsable_trees: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_required_fields(self) -> bool:
        return all(
            key in self.raw for key in ("task_graph", "assignments", "synchronization", "behavior_trees")
        )

    def missing_fields(self) -> list[str]:
        return [
            key
            for key in ("task_graph", "assignments", "synchronization", "behavior_trees")
            if key not in self.raw
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_graph": self.task_graph,
            "assignments": self.assignments,
            "synchronization": self.synchronization,
            "behavior_trees": {robot: tree.to_dict() for robot, tree in self.behavior_trees.items()},
        }


def parse_plan(raw: dict[str, Any]) -> Plan:
    """Convert a raw plan dict into a :class:`Plan`. Never raises on a malformed
    plan - structural issues are surfaced by the validator instead.
    """
    plan = Plan(raw=dict(raw))
    plan.task_graph = _as_list(raw.get("task_graph"))
    plan.assignments = _as_list(raw.get("assignments"))
    plan.synchronization = _as_list(raw.get("synchronization"))

    trees = raw.get("behavior_trees")
    if isinstance(trees, dict):
        for robot, tree in trees.items():
            try:
                plan.behavior_trees[robot] = parse_node(tree)
            except BTParseError as error:
                plan.unparsable_trees[robot] = str(error)
    return plan


def _as_list(value: Any) -> list[dict[str, Any]]:
    return value if isinstance(value, list) else []
