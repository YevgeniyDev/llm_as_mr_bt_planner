from __future__ import annotations

from mrbtp.domain import parse_scenario
from mrbtp.plan import parse_plan
from mrbtp.simulation import simulate


def _toy_scenario():
    return parse_scenario(
        {
            "task_id": "toy",
            "instruction": "toy",
            "initial_state": [],
            "goal_state": ["done()"],
            "objects": [],
            "locations": [],
            "robots": [
                {
                    "id": "A",
                    "capabilities": [
                        {"name": "make", "parameters": [], "preconditions": [],
                         "effects": {"add": ["p()"], "delete": []}}
                    ],
                },
                {
                    "id": "B",
                    "capabilities": [
                        {"name": "use", "parameters": [], "preconditions": ["p()"],
                         "effects": {"add": ["done()"], "delete": []}}
                    ],
                },
            ],
        }
    )


def _plan(b_tree):
    return parse_plan(
        {
            "task_graph": [],
            "assignments": [],
            "synchronization": [],
            "behavior_trees": {
                # B first so it must wait a tick for A to produce p().
                "B": b_tree,
                "A": {"type": "Sequence", "children": [{"type": "Action", "name": "make", "parameters": []}]},
            },
        }
    )


def test_toy_plan_simulates_to_success(toy_scenario, toy_plan):
    report = simulate(toy_plan, toy_scenario)
    assert report.success
    assert report.goal_success
    assert report.errors == []


def test_cross_robot_sync_succeeds():
    plan = _plan(
        {"type": "Sequence", "children": [
            {"type": "Condition", "name": "p", "parameters": []},
            {"type": "Action", "name": "use", "parameters": []},
        ]}
    )
    report = simulate(plan, _toy_scenario())
    assert report.success


def test_deadlock_detected():
    plan = _plan(
        {"type": "Sequence", "children": [
            {"type": "Condition", "name": "q", "parameters": []},  # never produced
            {"type": "Action", "name": "use", "parameters": []},
        ]}
    )
    report = simulate(plan, _toy_scenario())
    assert not report.success
    assert report.errors and report.errors[0]["type"] == "deadlock"


def test_timeout_detected():
    plan = _plan(
        {"type": "Sequence", "children": [
            {"type": "Condition", "name": "p", "parameters": []},
            {"type": "Action", "name": "use", "parameters": []},
        ]}
    )
    report = simulate(plan, _toy_scenario(), max_ticks=1)
    assert not report.success
    assert report.errors and report.errors[0]["type"] == "timeout"
