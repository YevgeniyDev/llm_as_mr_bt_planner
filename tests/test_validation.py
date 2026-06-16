from __future__ import annotations

import copy

from mrbtp.domain import parse_scenario
from mrbtp.plan import parse_plan
from mrbtp.validation import validate_plan


def _scenario():
    return parse_scenario(
        {
            "task_id": "toy",
            "instruction": "toy",
            "initial_state": [],
            "goal_state": ["done()"],
            "objects": [],
            "locations": [],
            "robots": [
                {"id": "A", "capabilities": [
                    {"name": "make", "parameters": [], "preconditions": [],
                     "effects": {"add": ["p()"], "delete": []}}]},
                {"id": "B", "capabilities": [
                    {"name": "use", "parameters": [], "preconditions": ["p()"],
                     "effects": {"add": ["done()"], "delete": []}}]},
            ],
        }
    )


def _base_plan():
    return {
        "task_graph": [
            {"id": "t1", "action": "make", "parameters": [], "depends_on": []},
            {"id": "t2", "action": "use", "parameters": [], "depends_on": ["t1"]},
        ],
        "assignments": [{"task_id": "t1", "robot": "A"}, {"task_id": "t2", "robot": "B"}],
        "synchronization": [{"condition": "p()", "producer": "A", "consumer": "B"}],
        "behavior_trees": {
            "A": {"type": "Sequence", "children": [{"type": "Action", "name": "make", "parameters": []}]},
            "B": {"type": "Sequence", "children": [
                {"type": "Condition", "name": "p", "parameters": []},
                {"type": "Action", "name": "use", "parameters": []},
            ]},
        },
    }


def _types(plan_dict):
    return {e.type for e in validate_plan(parse_plan(plan_dict), _scenario()).errors}


def test_base_toy_plan_is_valid():
    assert validate_plan(parse_plan(_base_plan()), _scenario()).valid


def test_missing_fields():
    assert "missing_field" in _types({"task_graph": []})


def test_cyclic_dependency():
    plan = _base_plan()
    plan["task_graph"][0]["depends_on"] = ["t2"]  # t1 <-> t2 cycle
    assert "cyclic_dependency" in _types(plan)


def test_invalid_capability():
    plan = _base_plan()
    plan["assignments"][0]["robot"] = "B"  # B cannot 'make'
    assert "invalid_capability" in _types(plan)


def test_missing_bt_action():
    plan = _base_plan()
    plan["behavior_trees"]["A"]["children"] = []  # assignment t1 has no BT action
    assert "missing_bt_action" in _types(plan)


def test_unassigned_bt_action():
    plan = _base_plan()
    plan["behavior_trees"]["A"]["children"].append({"type": "Action", "name": "make", "parameters": ["x"]})
    assert "unassigned_bt_action" in _types(plan)


def test_unsupported_goal_and_precondition():
    plan = _base_plan()
    # Remove the producer of p(): drop make from task graph, assignment and BT.
    plan["task_graph"] = [t for t in plan["task_graph"] if t["id"] != "t1"]
    plan["assignments"] = [a for a in plan["assignments"] if a["task_id"] != "t1"]
    plan["behavior_trees"]["A"]["children"] = []
    plan["synchronization"] = []
    types = _types(plan)
    assert "unsupported_precondition" in types  # use() needs p()


def test_missing_sync_condition():
    plan = _base_plan()
    plan["behavior_trees"]["B"]["children"] = [
        {"type": "Action", "name": "use", "parameters": []}
    ]  # dropped the Condition p()
    assert "missing_sync_condition" in _types(plan)


def test_condition_before_producer_same_robot():
    # A waits for its own future output before producing it.
    plan = {
        "task_graph": [{"id": "t1", "action": "make", "parameters": [], "depends_on": []}],
        "assignments": [{"task_id": "t1", "robot": "A"}],
        "synchronization": [],
        "behavior_trees": {
            "A": {"type": "Sequence", "children": [
                {"type": "Condition", "name": "p", "parameters": []},
                {"type": "Action", "name": "make", "parameters": []},
            ]},
        },
    }
    assert "condition_before_producer" in _types(plan)
