from __future__ import annotations

import json
from pathlib import Path

from llm_mr_bt_planner.artifacts import load_plan_file
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.inspection_recovery import (
    INSPECTION_ROBOTS,
    build_inspection_recovery_prompt,
    build_inspection_tool_recovery_scenario,
)
from llm_mr_bt_planner.mujoco_sim.camera_director import camera_director_for_task
from llm_mr_bt_planner.recovery import recovery_plan_json_schema
from llm_mr_bt_planner.simulation import simulate
from llm_mr_bt_planner.validation import validate_plan

ROOT = Path(__file__).resolve().parents[1]
INSPECTION_SCENARIO = ROOT / "examples" / "five_agent_solar_pipe_inspection.json"
INSPECTION_BT = ROOT / "examples" / "five_agent_solar_pipe_inspection.bt.json"
DROP_FAULT = ROOT / "examples" / "five_agent_solar_pipe_inspection_tool_drop.fault.json"
PIPE_SCENARIO = ROOT / "examples" / "five_agent_pipe_leak_repair.json"
PIPE_BT = ROOT / "examples" / "five_agent_pipe_leak_repair.bt.json"


def _failure_observation() -> dict[str, object]:
    return {
        "classification": "tool_dropped_and_location_unknown",
        "object": "inspection_kit",
        "object_usable": True,
        "requires_localization": True,
        "measured_position_m_for_audit": [-1.49, -0.79, 0.025],
    }


def test_pipe_repair_reference_contract_reaches_all_goals():
    scenario = load_scenario(PIPE_SCENARIO, strict=True)
    plan = load_plan_file(PIPE_BT)
    validation = validate_plan(plan, scenario, suggest_producers=True)
    assert validation.valid, validation.to_dicts()
    report = simulate(plan, scenario, max_ticks=400)
    assert report.success
    assert report.goal_success


def test_fault_is_sealed_outside_nominal_inspection_scenario():
    scenario_text = INSPECTION_SCENARIO.read_text(encoding="utf-8")
    fault = json.loads(DROP_FAULT.read_text(encoding="utf-8"))
    assert fault["trigger"]["after_action"] == "place_inspection_kit_handoff"
    assert fault["trigger"]["before_action"] == "load_inspection_kit"
    assert fault["fault_id"] not in scenario_text
    assert "recover_localized_tool" not in scenario_text


def test_runtime_recovery_reveals_only_measured_search_and_pickup_capabilities():
    scenario = load_scenario(INSPECTION_SCENARIO, strict=True)
    measured = tuple(
        fact
        for fact in scenario.initial_state
        if fact != "at(inspection_kit, kit_supply)"
    ) + (
        "fallen_tool_unlocalized(inspection_kit)",
        "arm_home(static_franka)",
        "gripper_empty(static_franka)",
    )
    runtime = build_inspection_tool_recovery_scenario(
        scenario,
        measured_initial_state=measured,
        failure_observation=_failure_observation(),
    )
    assert scenario.capability("husky_franka", "recover_localized_tool") is None
    assert runtime.capability("b2_base", "navigate_b2_tool_search") is not None
    assert runtime.capability("z1_thermal_arm", "localize_fallen_tool") is not None
    assert runtime.capability("husky_base", "navigate_husky_tool_recovery") is not None
    assert runtime.capability("husky_franka", "recover_localized_tool") is not None


def test_recovery_prompt_withholds_audit_coordinates_until_b2_search():
    scenario = load_scenario(INSPECTION_SCENARIO, strict=True)
    nominal = load_plan_file(INSPECTION_BT)
    runtime = build_inspection_tool_recovery_scenario(
        scenario,
        measured_initial_state=(
            *scenario.initial_state,
            "fallen_tool_unlocalized(inspection_kit)",
        ),
        failure_observation=_failure_observation(),
    )
    prompt = build_inspection_recovery_prompt(
        runtime,
        failure_observation=_failure_observation(),
        nominal_plan=nominal,
    )
    assert "measured_position_m_for_audit" not in prompt
    assert "-1.49" not in prompt
    assert "localize_fallen_tool" in prompt


def test_five_agent_recovery_schema_requires_exact_robot_team():
    schema = recovery_plan_json_schema(
        mission_id="five_agent_solar_pipe_inspection",
        robots=INSPECTION_ROBOTS,
    )
    behavior_trees = schema["properties"]["behavior_trees"]
    assert behavior_trees["required"] == list(INSPECTION_ROBOTS)
    assert set(behavior_trees["properties"]) == set(INSPECTION_ROBOTS)


def test_new_camera_programs_cover_search_recovery_and_pipe_repair():
    inspection = camera_director_for_task("five_agent_solar_pipe_inspection").program
    pipe = camera_director_for_task("five_agent_pipe_leak_repair").program
    assert inspection.action_cameras["localize_fallen_tool"] == "inspection_search"
    assert inspection.action_cameras["recover_localized_tool"] == "inspection_floor_recovery"
    assert pipe.action_cameras["detect_pipe_leak"] == "inspection_pipe"
    assert pipe.action_cameras["repair_pipe_leak"] == "inspection_service"
