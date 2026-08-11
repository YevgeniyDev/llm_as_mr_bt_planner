"""Canonical multi-robot Behavior Tree plan model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bt import BTNode, BTParseError, parse_node


@dataclass
class Plan:
    schema_version: str = ""
    mission_id: str = ""
    behavior_trees: dict[str, BTNode] = field(default_factory=dict)
    unparsable_trees: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_required_fields(self) -> bool:
        return not self.missing_fields()

    def missing_fields(self) -> list[str]:
        return [key for key in ("schema_version", "mission_id", "behavior_trees") if key not in self.raw]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "behavior_trees": {robot: tree.to_dict() for robot, tree in self.behavior_trees.items()},
        }


def parse_plan(raw: dict[str, Any]) -> Plan:
    """Parse canonical plan JSON while retaining structural errors for validation."""
    if not isinstance(raw, dict):
        raw = {}
    plan = Plan(
        schema_version=str(raw.get("schema_version", "")),
        mission_id=str(raw.get("mission_id", "")),
        raw=dict(raw),
    )
    trees = raw.get("behavior_trees")
    if isinstance(trees, dict):
        for robot, tree in trees.items():
            robot_id = str(robot)
            try:
                plan.behavior_trees[robot_id] = parse_node(tree, f"{robot_id}.root")
            except BTParseError as error:
                plan.unparsable_trees[robot_id] = str(error)
    return plan
