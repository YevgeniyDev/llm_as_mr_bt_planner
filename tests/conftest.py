"""Shared test fixtures.

The suite is engine-only: it exercises the deterministic parser/validator/
simulator/visualizer with no LLM and no saved "answer" plans. Positive tests use
a tiny inline two-robot domain (`toy_scenario` + `toy_plan`) that is known to be
valid and to simulate to success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mrbtp.domain import Scenario, load_scenario, parse_scenario
from mrbtp.plan import Plan, parse_plan

DATA = Path(__file__).resolve().parents[1] / "data"

SCENARIOS = {
    "gear_assembly": DATA / "scenario.json",
    "sensor_calibration_cell": DATA / "scenario2.json",
}


@pytest.fixture
def gear_scenario() -> Scenario:
    return load_scenario(SCENARIOS["gear_assembly"])


@pytest.fixture
def sensor_scenario() -> Scenario:
    return load_scenario(SCENARIOS["sensor_calibration_cell"])


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
    )


@pytest.fixture
def toy_scenario() -> Scenario:
    return make_toy_scenario()


@pytest.fixture
def toy_plan() -> Plan:
    return make_toy_plan()
