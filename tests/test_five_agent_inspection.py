from __future__ import annotations

from pathlib import Path

from llm_mr_bt_planner.artifacts import load_plan_file
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.mujoco_sim.camera_director import camera_director_for_task
from llm_mr_bt_planner.mujoco_sim.inspection_assets import HUSKY_COMMIT, UNITREE_COMMIT
from llm_mr_bt_planner.simulation import simulate
from llm_mr_bt_planner.validation import validate_plan

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "examples" / "five_agent_solar_pipe_inspection.json"
BT = ROOT / "examples" / "five_agent_solar_pipe_inspection.bt.json"


def test_five_agent_inspection_reference_contract_reaches_all_goals():
    scenario = load_scenario(SCENARIO, strict=True)
    plan = load_plan_file(BT)

    assert {robot.id for robot in scenario.robots} == {
        "b2_base",
        "z1_thermal_arm",
        "husky_base",
        "husky_franka",
        "static_franka",
    }
    validation = validate_plan(plan, scenario, suggest_producers=True)
    assert validation.valid, validation.to_dicts()
    report = simulate(plan, scenario, max_ticks=300)
    assert report.success
    assert report.goal_success
    assert not report.errors


def test_inspection_camera_program_covers_each_physical_phase():
    director = camera_director_for_task("five_agent_solar_pipe_inspection")
    cameras = set(director.cameras)
    assert {
        "inspection_overview",
        "inspection_handoff",
        "inspection_convoy",
        "inspection_solar",
        "inspection_pipe",
        "inspection_service",
    } <= cameras
    assert director.program.action_cameras["scan_pipe_rig"] == "inspection_pipe"
    assert director.program.action_cameras["isolate_energy_rig"] == "inspection_service"


def test_inspection_sources_are_immutable_commit_pins():
    assert len(UNITREE_COMMIT) == 40
    assert len(HUSKY_COMMIT) == 40
    assert all(character in "0123456789abcdef" for character in UNITREE_COMMIT + HUSKY_COMMIT)


def test_symbolic_scenario_does_not_reveal_hidden_physical_anomaly_site():
    assert "pipe_joint_2" not in SCENARIO.read_text(encoding="utf-8")
    assert "pipe_joint_2" not in BT.read_text(encoding="utf-8")
