"""Execution-time recovery ladder: failure detection -> retry -> reassign.

Engine-only (no LLM, no physics): failures come from a deterministic injected
oracle so the whole ladder is exercised reproducibly.
"""

from __future__ import annotations

from llm_mr_bt_planner.bt import Status
from llm_mr_bt_planner.domain import parse_scenario
from llm_mr_bt_planner.execution import SymbolicExecutionBackend
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.recovery import (
    InjectedFailureOracle,
    RecoveryController,
    parse_injection_spec,
)
from llm_mr_bt_planner.simulation import simulate

# --- inline domains ----------------------------------------------------------


def _solo_scenario():
    """A single robot that produces the only goal - lets a failure surface with
    no other robot around to deadlock, so the failure is unambiguously an
    action_failure and not a deadlock."""
    return parse_scenario({
        "task_id": "solo", "instruction": "x",
        "initial_state": [], "goal_state": ["p()"],
        "objects": [], "locations": [],
        "robots": [
            {"id": "A", "capabilities": [
                {"name": "make", "parameters": [], "preconditions": [],
                 "effects": {"add": ["p()"], "delete": []}}]},
        ],
    })


def _solo_plan():
    return parse_plan({
        "task_graph": [{"id": "t1", "action": "make", "parameters": [], "depends_on": []}],
        "assignments": [{"task_id": "t1", "robot": "A"}],
        "synchronization": [],
        "behavior_trees": {
            "A": {"type": "Sequence", "children": [{"type": "Action", "name": "make", "parameters": []}]},
        },
    })


def _two_producer_scenario():
    """p() can be produced by A.make OR B.make_b; C.use consumes it for done()."""
    return parse_scenario({
        "task_id": "reassign", "instruction": "x",
        "initial_state": [], "goal_state": ["done()"],
        "objects": [], "locations": [],
        "robots": [
            {"id": "A", "capabilities": [
                {"name": "make", "parameters": [], "preconditions": [],
                 "effects": {"add": ["p()"], "delete": []}}]},
            {"id": "B", "capabilities": [
                {"name": "make_b", "parameters": [], "preconditions": [],
                 "effects": {"add": ["p()"], "delete": []}}]},
            {"id": "C", "capabilities": [
                {"name": "use", "parameters": [], "preconditions": ["p()"],
                 "effects": {"add": ["done()"], "delete": []}}]},
        ],
    })


def _two_producer_plan():
    """A makes p(); C uses it. B is idle until reassignment moves the producer to it."""
    return parse_plan({
        "task_graph": [
            {"id": "t1", "action": "make", "parameters": [], "depends_on": []},
            {"id": "t2", "action": "use", "parameters": [], "depends_on": ["t1"]},
        ],
        "assignments": [{"task_id": "t1", "robot": "A"}, {"task_id": "t2", "robot": "C"}],
        "synchronization": [{"condition": "p()", "producer": "A", "consumer": "C"}],
        "behavior_trees": {
            "A": {"type": "Sequence", "children": [{"type": "Action", "name": "make", "parameters": []}]},
            "C": {"type": "Sequence", "children": [
                {"type": "Condition", "name": "p", "parameters": []},
                {"type": "Action", "name": "use", "parameters": []},
            ]},
        },
    })


def _single_producer_scenario():
    """Only A can produce p(): no reassignment target exists (robot-scoped case)."""
    return parse_scenario({
        "task_id": "single", "instruction": "x",
        "initial_state": [], "goal_state": ["done()"],
        "objects": [], "locations": [],
        "robots": [
            {"id": "A", "capabilities": [
                {"name": "make", "parameters": [], "preconditions": [],
                 "effects": {"add": ["p()"], "delete": []}}]},
            {"id": "C", "capabilities": [
                {"name": "use", "parameters": [], "preconditions": ["p()"],
                 "effects": {"add": ["done()"], "delete": []}}]},
        ],
    })


# --- oracle seam -------------------------------------------------------------


def test_no_oracle_is_unchanged(toy_scenario, toy_plan):
    report = simulate(toy_plan, toy_scenario)
    assert report.success and report.goal_success
    assert report.failures == []


def test_action_failure_is_distinct_from_deadlock():
    report = simulate(_solo_plan(), _solo_scenario(), action_oracle=InjectedFailureOracle({"make": 1}))
    assert not report.success
    assert report.failures and report.failures[0]["name"] == "make"
    assert report.errors[0]["type"] == "action_failure"
    assert not any(error["type"] == "deadlock" for error in report.errors)
    assert any(event.get("event") == "action_failed" for event in report.trace)


def test_injected_oracle_tally_is_monotonic():
    oracle = InjectedFailureOracle({"make": 2})
    event = {"name": "make", "robot": "A", "action": "make()"}
    outcomes = [oracle(event) for _ in range(4)]
    assert outcomes == [Status.FAILURE, Status.FAILURE, Status.SUCCESS, Status.SUCCESS]


def test_parse_injection_spec():
    assert parse_injection_spec("pick_tool:2,open_drawer:1") == {"pick_tool": 2, "open_drawer": 1}
    assert parse_injection_spec("make") == {"make": 1}
    assert parse_injection_spec("") == {}


# --- the recovery ladder -----------------------------------------------------


def test_tier1_retry_same_robot_succeeds(toy_scenario, toy_plan):
    controller = RecoveryController(InjectedFailureOracle({"make": 1}), max_retries=2)
    result = controller.run(toy_plan, toy_scenario)
    assert result.success and result.goal_success
    assert [event.tier for event in result.recovery_log] == ["retry"]
    assert result.episodes == 2
    # The producer stayed on its original robot.
    assert "A" in result.plan["behavior_trees"]


def test_retry_exhausted_then_reassign_succeeds():
    controller = RecoveryController(InjectedFailureOracle({"make": 99}), max_retries=1, allow_reassign=True)
    result = controller.run(_two_producer_plan(), _two_producer_scenario())
    assert result.success and result.goal_success
    tiers = [(event.tier, event.outcome) for event in result.recovery_log]
    assert ("retry", "retried") in tiers
    assert ("reassign", "reassigned") in tiers
    # The producing action now lives on robot B.
    b_children = result.plan["behavior_trees"].get("B", {}).get("children", [])
    assert any(child.get("name") == "make_b" for child in b_children)


def test_reassign_disabled_surfaces_retries_exhausted():
    controller = RecoveryController(InjectedFailureOracle({"make": 99}), max_retries=1, allow_reassign=False)
    result = controller.run(_two_producer_plan(), _two_producer_scenario())
    assert not result.success
    assert result.error == "retries_exhausted"
    assert all(event.tier != "reassign" for event in result.recovery_log)


def test_no_capable_robot_surfaces_unrecovered_failure():
    controller = RecoveryController(InjectedFailureOracle({"make": 99}), max_retries=1, allow_reassign=True)
    result = controller.run(_single_producer_plan(), _single_producer_scenario())
    assert not result.success
    assert result.error == "unrecovered_failure"
    assert result.recovery_log[-1].tier == "reassign"
    assert result.recovery_log[-1].outcome == "no_candidate"


def _single_producer_plan():
    return parse_plan({
        "task_graph": [
            {"id": "t1", "action": "make", "parameters": [], "depends_on": []},
            {"id": "t2", "action": "use", "parameters": [], "depends_on": ["t1"]},
        ],
        "assignments": [{"task_id": "t1", "robot": "A"}, {"task_id": "t2", "robot": "C"}],
        "synchronization": [{"condition": "p()", "producer": "A", "consumer": "C"}],
        "behavior_trees": {
            "A": {"type": "Sequence", "children": [{"type": "Action", "name": "make", "parameters": []}]},
            "C": {"type": "Sequence", "children": [
                {"type": "Condition", "name": "p", "parameters": []},
                {"type": "Action", "name": "use", "parameters": []},
            ]},
        },
    })


def test_recovery_via_symbolic_backend(toy_scenario, toy_plan):
    controller = RecoveryController(InjectedFailureOracle({"make": 1}), max_retries=2)
    result = SymbolicExecutionBackend(recovery=controller).execute(toy_plan, toy_scenario)
    assert result.success and result.goal_success
    recovery = result.details["recovery"]
    assert recovery["episodes"] >= 2
    assert recovery["log"] and recovery["log"][0]["tier"] == "retry"
