from __future__ import annotations

import copy
from typing import Any

from llm_mr_bt_planner.domain import parse_scenario
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.validation import validate_plan


def _scenario():
    return parse_scenario(
        {
            "task_id": "toy",
            "instruction": "Make done true",
            "initial_state": [],
            "goal_state": ["done()"],
            "objects": [],
            "locations": [],
            "robots": [
                {
                    "id": "A",
                    "capabilities": [
                        {
                            "name": "make",
                            "parameters": [],
                            "preconditions": [],
                            "effects": {"add": ["p()"], "delete": []},
                        }
                    ],
                },
                {
                    "id": "B",
                    "capabilities": [
                        {
                            "name": "use",
                            "parameters": [],
                            "preconditions": ["p()"],
                            "effects": {"add": ["done()"], "delete": []},
                        }
                    ],
                },
            ],
        }
    )


def _base_plan() -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "mission_id": "toy",
        "behavior_trees": {
            "A": {
                "id": "A.root",
                "type": "Sequence",
                "source": "llm",
                "children": [
                    {"id": "A.make", "type": "Action", "source": "llm", "task_id": "make-task", "name": "make", "parameters": []}
                ],
            },
            "B": {
                "id": "B.root",
                "type": "Sequence",
                "source": "llm",
                "children": [
                    {"id": "B.wait.p", "type": "WaitFor", "source": "llm", "name": "p", "parameters": [], "timeout_ticks": 20},
                    {"id": "B.use", "type": "Action", "source": "llm", "task_id": "use-task", "name": "use", "parameters": []},
                ],
            },
        },
    }


def _types(document: dict[str, Any]) -> set[str]:
    return {error.type for error in validate_plan(parse_plan(document), _scenario()).errors}


def _nodes(document: dict[str, Any]):
    stack = list(document.get("behavior_trees", {}).values())
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.get("children", []))


def test_direct_llm_toy_plan_is_valid():
    report = validate_plan(parse_plan(_base_plan()), _scenario())
    assert report.valid, report.to_dicts()


def test_missing_canonical_field_is_rejected():
    assert "missing_field" in _types({"schema_version": "1.0", "mission_id": "toy"})


def test_wrong_schema_and_mission_are_rejected():
    document = _base_plan()
    document["schema_version"] = "3.0"
    document["mission_id"] = "another-mission"
    assert {"unsupported_schema_version", "mission_mismatch"} <= _types(document)


def test_missing_robot_tree_is_rejected():
    document = _base_plan()
    del document["behavior_trees"]["B"]
    assert "missing_robot_tree" in _types(document)


def test_invalid_capability_is_rejected():
    document = _base_plan()
    action = next(node for node in _nodes(document) if node.get("type") == "Action")
    action["name"] = "invented_action"
    assert "invalid_bt_action" in _types(document)


def test_duplicate_task_id_is_rejected():
    document = _base_plan()
    actions = [node for node in _nodes(document) if node.get("type") == "Action"]
    actions[1]["task_id"] = actions[0]["task_id"]
    assert "duplicate_task_id" in _types(document)


def test_hidden_action_contract_rewrite_field_is_rejected():
    document = _base_plan()
    action = next(node for node in _nodes(document) if node.get("name") == "use")
    action["preconditions"] = []
    assert "unknown_node_field" in _types(document)


def test_cross_robot_dependency_requires_explicit_wait():
    document = copy.deepcopy(_base_plan())

    def remove_waits(node: dict[str, Any]) -> None:
        if "children" not in node:
            return
        children = node.get("children", [])
        node["children"] = [child for child in children if child.get("type") != "WaitFor"]
        for child in node["children"]:
            remove_waits(child)

    remove_waits(document["behavior_trees"]["B"])
    assert "missing_wait_for" in _types(document)
