"""Shared deterministic fixtures for the standalone planner test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_mr_bt_planner.domain import Scenario, load_scenario, parse_scenario
from llm_mr_bt_planner.plan import Plan, parse_plan

ROOT = Path(__file__).resolve().parents[1]
COURIER_SCENARIO = ROOT / "examples" / "three_robot_courier.json"


@pytest.fixture
def courier_scenario() -> Scenario:
    return load_scenario(COURIER_SCENARIO, strict=True)


def make_toy_scenario() -> Scenario:
    """A minimal two-robot domain: A makes p(), B consumes p() to reach done()."""
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


def make_toy_plan() -> Plan:
    """A valid plan for `make_toy_scenario` that simulates to success."""
    return parse_plan(
        {
            "schema_version": "2.0",
            "mission_id": "toy",
            "behavior_trees": {
                "A": {
                    "id": "A.root",
                    "type": "Sequence",
                    "source": "llm",
                    "children": [
                        {"id": "A.make", "type": "Action", "task_id": "t1", "name": "make", "parameters": [], "source": "llm"}
                    ],
                },
                "B": {
                    "id": "B.root",
                    "type": "Sequence",
                    "source": "llm",
                    "children": [
                        {"id": "B.wait.p", "type": "WaitFor", "name": "p", "parameters": [], "timeout_ticks": 20, "source": "llm"},
                        {"id": "B.use", "type": "Action", "task_id": "t2", "name": "use", "parameters": [], "source": "llm"},
                    ],
                },
            },
        }
    )


@pytest.fixture
def toy_scenario() -> Scenario:
    return make_toy_scenario()


@pytest.fixture
def toy_plan() -> Plan:
    return make_toy_plan()
