"""MRBTP adapter: tree conversion, scenario porting, and results ingest.

LLM-free and MRBTP-free: the MRBTP package is not imported here. We pin the
AnyTreeNode -> Plan converter on a fixture, check the scenario porting grounds
capabilities and expands delete patterns correctly, and verify the ingest path
re-scores a recorded plan with our own validator+simulator.
"""

from __future__ import annotations

import json

import pytest

from llm_mr_bt_planner.baselines.mrbtp_adapter import mrbtp_bt_to_plan, run_mrbtp
from llm_mr_bt_planner.baselines.mrbtp_port import port_scenario
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.simulation import simulate
from llm_mr_bt_planner.validation import validate_plan

# A serialized MRBTP tree per robot for the toy domain (A makes p(); B uses p()).
_TOY_TREES = {
    "A": {"node_type": "sequence", "cls_name": None, "args": [],
          "children": [{"node_type": "action", "cls_name": "make", "args": [], "children": []}]},
    "B": {"node_type": "sequence", "cls_name": None, "args": [],
          "children": [
              {"node_type": "condition", "cls_name": "p", "args": [], "children": []},
              {"node_type": "action", "cls_name": "use", "args": [], "children": []},
          ]},
}


# --- converter ------------------------------------------------------------- #

def test_converter_builds_rescorable_plan(toy_scenario):
    plan_dict = mrbtp_bt_to_plan(_TOY_TREES, toy_scenario)
    # Trees map to our composites/leaves.
    assert plan_dict["behavior_trees"]["A"]["type"] == "Sequence"
    assert plan_dict["behavior_trees"]["B"]["children"][0]["type"] == "Condition"
    # task_graph/assignments are derived from the Action leaves.
    actions = {
        (a["robot"], t["action"])
        for a, t in zip(plan_dict["assignments"], plan_dict["task_graph"], strict=True)
    }
    assert ("A", "make") in actions and ("B", "use") in actions
    # And the converted plan re-scores as valid + successful under our verifier.
    plan = parse_plan(plan_dict)
    validation = validate_plan(plan, toy_scenario)
    assert validation.valid
    assert simulate(plan, toy_scenario, max_ticks=20).success


def test_converter_maps_selector_to_fallback(toy_scenario):
    trees = {"A": {"node_type": "selector", "cls_name": None, "args": [],
                   "children": [{"node_type": "action", "cls_name": "make", "args": [], "children": []}]}}
    plan = mrbtp_bt_to_plan(trees, toy_scenario)
    assert plan["behavior_trees"]["A"]["type"] == "Fallback"


# --- porting --------------------------------------------------------------- #

def test_port_toy_scenario(toy_scenario):
    ported = port_scenario(toy_scenario)
    assert ported["goal"] == ["done()"]
    by_robot = {a["robot"]: a["actions"] for a in ported["agents"]}
    make = next(act for act in by_robot["A"] if act["name"] == "make")
    assert make["add"] == ["p()"] and make["pre"] == []
    use = next(act for act in by_robot["B"] if act["name"] == "use")
    assert use["pre"] == ["p()"] and use["add"] == ["done()"]


def test_port_expands_delete_patterns(gear_scenario):
    ported = port_scenario(gear_scenario)
    actions = [act for agent in ported["agents"] for act in agent["actions"]]
    open_drawer = next(act for act in actions if act["name"] == "open_drawer(parts_drawer)")
    # The prefix/pattern delete is expanded to the concrete fact it removes.
    assert "drawer_closed(parts_drawer)" in open_drawer["del_set"]
    # Robot-scoped predicates keep their literal robot id (it is not a parameter).
    pick = next(act for act in actions if act["name"] == "pick_gear(gear)")
    assert "holding(franka2, gear)" in pick["add"]


# --- ingest ---------------------------------------------------------------- #

def test_run_mrbtp_ingests_recorded_plan(toy_scenario, tmp_path):
    results = {"scenarios": {"toy": {
        "variant": "MAOBTP",
        "planning_time": 0.5,
        "feedback_rounds": 0,
        "plan": mrbtp_bt_to_plan(_TOY_TREES, toy_scenario),
    }}}
    path = tmp_path / "mrbtp_results.json"
    path.write_text(json.dumps(results), encoding="utf-8")

    result = run_mrbtp(toy_scenario, None, results_path=str(path))
    assert result.provider == "mrbtp"
    assert result.valid and result.success
    assert result.wall_seconds == 0.5


def test_run_mrbtp_native_metrics_path(toy_scenario, tmp_path):
    """When no plan is recorded (the scripts/run_mrbtp.py default), the adapter uses
    MRBTP's native found/time metrics directly."""
    results = {"scenarios": {"toy": {
        "variant": "MAOBTP", "valid": True, "success": True, "goal_success": True,
        "timed_out": False, "planning_time": 0.71, "expanded_count": 987, "feedback_rounds": 0,
    }}}
    path = tmp_path / "mrbtp_results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    result = run_mrbtp(toy_scenario, None, results_path=str(path))
    assert result.valid and result.success and result.goal_success
    assert result.wall_seconds == 0.71
    assert result.validation_errors == []


def test_run_mrbtp_native_timeout_is_failure(toy_scenario, tmp_path):
    results = {"scenarios": {"toy": {
        "variant": "MAOBTP", "valid": False, "success": False, "goal_success": False,
        "timed_out": True, "planning_time": 300.0, "feedback_rounds": 0,
    }}}
    path = tmp_path / "mrbtp_results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    result = run_mrbtp(toy_scenario, None, results_path=str(path))
    assert not result.valid and not result.success


def test_run_mrbtp_missing_file_raises(toy_scenario, tmp_path):
    with pytest.raises(FileNotFoundError):
        run_mrbtp(toy_scenario, None, results_path=str(tmp_path / "nope.json"))
