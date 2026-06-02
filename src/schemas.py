from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict:
    """Load a UTF-8 JSON file as a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, data: dict) -> None:
    """Save a dictionary as pretty-printed UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


@dataclass
class TaskGraphNode:
    id: str
    action: str
    parameters: list[str]
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Assignment:
    task_id: str
    robot: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynchronizationPoint:
    id: str
    producer: str
    consumer: str
    condition: str
    producer_task: str
    consumer_task: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BehaviorTreeNode:
    type: str
    name: str | None = None
    parameters: list[str] = field(default_factory=list)
    children: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        node = asdict(self)
        return {key: value for key, value in node.items() if value not in (None, [], {})}


@dataclass
class Plan:
    task_graph: list[dict]
    assignments: list[dict]
    synchronization: list[dict]
    behavior_trees: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
