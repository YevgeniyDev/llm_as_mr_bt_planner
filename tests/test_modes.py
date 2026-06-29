"""Tests for the 'pure' (default) vs 'assisted' (ablation) planning modes."""

from __future__ import annotations

from llm_mr_bt_planner.domain import parse_scenario
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.prompts import build_prompt
from llm_mr_bt_planner.validation import validate_plan


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


def _plan_missing_producer():
    # B.use needs p(), but nothing produces p() -> unsupported_precondition.
    return parse_plan(
        {
            "task_graph": [{"id": "t2", "action": "use", "parameters": [], "depends_on": []}],
            "assignments": [{"task_id": "t2", "robot": "B"}],
            "synchronization": [],
            "behavior_trees": {
                "B": {"type": "Sequence", "children": [{"type": "Action", "name": "use", "parameters": []}]},
            },
        }
    )


def test_prompt_pure_has_no_dependency_hints(gear_scenario):
    assert "dependency hints" not in build_prompt(gear_scenario).lower()


def test_prompt_assisted_includes_dependency_hints(gear_scenario):
    assert "dependency hints" in build_prompt(gear_scenario, include_hints=True).lower()


def test_validator_pure_reports_problem_without_suggesting_producer():
    report = validate_plan(_plan_missing_producer(), _scenario())  # suggest_producers=False (default)
    msgs = [e.message for e in report.errors if e.type == "unsupported_precondition"]
    assert msgs, "expected an unsupported_precondition error"
    assert all("Candidate producer" not in m for m in msgs)


def test_validator_assisted_suggests_producer():
    report = validate_plan(_plan_missing_producer(), _scenario(), suggest_producers=True)
    msgs = [e.message for e in report.errors if e.type == "unsupported_precondition"]
    assert any("Candidate producer actions: robot A action make()" in m for m in msgs)


def test_both_modes_still_validate_a_correct_plan(toy_scenario, toy_plan):
    assert validate_plan(toy_plan, toy_scenario).valid
    assert validate_plan(toy_plan, toy_scenario, suggest_producers=True).valid
