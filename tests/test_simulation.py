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


def _parallel_scenario():
    # One robot, two independent producers. With actions_per_tick=1 only one of
    # the two actions can fire per tick, so the Parallel must span two ticks.
    return parse_scenario(
        {
            "task_id": "par",
            "instruction": "parallel",
            "initial_state": [],
            "goal_state": ["p1()", "p2()"],
            "objects": [],
            "locations": [],
            "robots": [
                {
                    "id": "A",
                    "capabilities": [
                        {"name": "m1", "parameters": [], "preconditions": [],
                         "effects": {"add": ["p1()"], "delete": []}},
                        {"name": "m2", "parameters": [], "preconditions": [],
                         "effects": {"add": ["p2()"], "delete": []}},
                    ],
                },
            ],
        }
    )


def test_parallel_latches_completed_children():
    # Regression: a Parallel whose children take more than one tick (the action
    # budget only lets one child act per tick) must latch the child that already
    # succeeded instead of re-ticking it forever. Without latching the first
    # action re-fires every tick, the budget is never free for the second child,
    # and the run dead-ends instead of succeeding.
    plan = parse_plan(
        {
            "task_graph": [],
            "assignments": [],
            "synchronization": [],
            "behavior_trees": {
                "A": {"type": "Parallel", "children": [
                    {"type": "Action", "name": "m1", "parameters": []},
                    {"type": "Action", "name": "m2", "parameters": []},
                ]},
            },
        }
    )
    report = simulate(plan, _parallel_scenario())
    assert report.success
    assert report.goal_success
    # Each action fires exactly once (no re-execution of a latched child).
    actions = [e for e in report.trace if e["event"] == "action"]
    assert sorted(e["action"] for e in actions) == ["m1()", "m2()"]
