"""Baseline generators: registry/dispatch and the flat & hier runners.

LLM-free: a stub client returns canned JSON so we exercise the generation flow and
confirm every baseline is scored by the same validator+simulator as the proposed
method. The toy domain (A produces p(); B consumes p() to reach done()) makes the
cross-robot dependency explicit, which is exactly what the no-synchronization
baselines cannot express.
"""

from __future__ import annotations

import json

import pytest

from llm_mr_bt_planner.baselines import BASELINES, get_runner
from llm_mr_bt_planner.baselines.flat_llm import run_flat
from llm_mr_bt_planner.baselines.hier_llm import run_hier
from llm_mr_bt_planner.planner import run_planner

# --- canned LLM responses -------------------------------------------------- #

_TOY_DECOMPOSITION = {"action_plan": {"A": [{"action": "make", "parameters": []}],
                                      "B": [{"action": "use", "parameters": []}]}}

_TOY_PLAN_WITH_SYNC = {
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

# Same plan but WITHOUT the cross-robot guard - what a no-sync baseline emits. The
# tick simulator's blocking-guard semantics still let B wait on A's Action precondition,
# so this is valid+successful: the validator checks global producedness, not the guard.
_TOY_PLAN_NO_SYNC = {
    "task_graph": _TOY_PLAN_WITH_SYNC["task_graph"],
    "assignments": _TOY_PLAN_WITH_SYNC["assignments"],
    "synchronization": [],
    "behavior_trees": {
        "A": {"type": "Sequence", "children": [{"type": "Action", "name": "make", "parameters": []}]},
        "B": {"type": "Sequence", "children": [{"type": "Action", "name": "use", "parameters": []}]},
    },
}

# The dominant real failure mode: a precondition whose producer action is omitted
# entirely (B consumes p() but no robot ever makes it) -> unsupported_precondition.
_TOY_PLAN_MISSING_PRODUCER = {
    "task_graph": [{"id": "t2", "action": "use", "parameters": [], "depends_on": []}],
    "assignments": [{"task_id": "t2", "robot": "B"}],
    "synchronization": [],
    "behavior_trees": {
        "B": {"type": "Sequence", "children": [{"type": "Action", "name": "use", "parameters": []}]},
    },
}


class StubClient:
    """Returns ``decomposition`` for stage-1 hier prompts and ``plan`` otherwise."""

    name = "stub"
    model = "stub-1"

    def __init__(self, plan: dict, decomposition: dict | None = None):
        self._plan = plan
        self._decomposition = decomposition or _TOY_DECOMPOSITION
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if user.startswith("Decompose the task"):
            return json.dumps(self._decomposition)
        return json.dumps(self._plan)


# --- registry / dispatch --------------------------------------------------- #

def test_registry_has_all_methods():
    assert set(BASELINES) == {"proposed", "flat", "hier", "mrbtp"}
    assert BASELINES["proposed"] is run_planner
    assert get_runner("flat") is run_flat
    assert get_runner("hier") is run_hier


def test_get_runner_unknown_raises():
    with pytest.raises(KeyError):
        get_runner("nope")


# --- flat (LLM-MARS-style) ------------------------------------------------- #

def test_flat_single_shot_no_correction(toy_scenario):
    """flat does exactly one generation and never loops, even with a budget."""
    scenario = toy_scenario
    client = StubClient(_TOY_PLAN_NO_SYNC)
    result = run_flat(scenario, client, max_corrections=4, samples=1)
    assert client.calls == 1
    assert result.correction_rounds == 0


def test_flat_scored_by_shared_verifier_exposes_omitted_producer(toy_scenario):
    """An omitted producer action is flagged by the same validator the proposed
    method uses - confirming baselines share the measurement instrument."""
    scenario = toy_scenario
    result = run_flat(scenario, StubClient(_TOY_PLAN_MISSING_PRODUCER), samples=1)
    assert not result.valid
    assert any(e["type"] == "unsupported_precondition" for e in result.validation_errors)


def test_flat_emits_no_synchronization(toy_scenario):
    """flat never declares synchronization edges; when ordering resolves via the
    simulator's blocking guards it can still succeed (no sync edge required)."""
    scenario = toy_scenario
    result = run_flat(scenario, StubClient(_TOY_PLAN_NO_SYNC), samples=1)
    assert result.plan["synchronization"] == []
    assert result.valid and result.success


def test_flat_accepts_a_correct_plan(toy_scenario):
    scenario = toy_scenario
    result = run_flat(scenario, StubClient(_TOY_PLAN_WITH_SYNC), samples=1)
    assert result.valid and result.success and result.goal_success
    assert result.correction_rounds == 0


# --- hier (LLM-as-BT-Planner-style) ---------------------------------------- #

def test_hier_runs_decompose_then_encode_then_correct(toy_scenario):
    """hier = 1 decomposition + (1 + max_corrections) encoding calls (samples=1)."""
    scenario = toy_scenario
    client = StubClient(_TOY_PLAN_MISSING_PRODUCER)
    result = run_hier(scenario, client, max_corrections=2, samples=1)
    assert client.calls == 1 + (1 + 2)
    # The stub keeps emitting the same producer-omitting plan, so it loops to budget.
    assert result.correction_rounds == 2
    assert not result.valid
    assert result.plan["synchronization"] == []


def test_hier_stops_early_on_success(toy_scenario):
    """A valid+successful encoding ends the loop immediately (rounds=0)."""
    scenario = toy_scenario
    result = run_hier(scenario, StubClient(_TOY_PLAN_WITH_SYNC), max_corrections=4, samples=1)
    assert result.valid and result.success
    assert result.correction_rounds == 0
