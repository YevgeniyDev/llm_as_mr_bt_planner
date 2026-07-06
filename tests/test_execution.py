from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_mr_bt_planner.execution import SymbolicExecutionBackend, get_backend

RESULTS = Path(__file__).resolve().parents[1] / "outputs"


# --- backend-agnostic tests --------------------------------------------------

def test_symbolic_backend_runs_a_correct_plan(toy_scenario, toy_plan):
    result = SymbolicExecutionBackend().execute(toy_plan, toy_scenario)
    assert result.backend == "symbolic"
    assert result.success and result.goal_success


def test_get_backend_unknown():
    with pytest.raises(ValueError):
        get_backend("warp_drive")


def test_ros_backend_is_an_actionable_scaffold(toy_plan, toy_scenario):
    backend = get_backend("ros")
    with pytest.raises(NotImplementedError) as info:
        backend.execute(toy_plan, toy_scenario)
    assert "export_behaviortree_cpp_xml" in str(info.value)


def test_mujoco_fidelity_is_validated():
    pytest.importorskip("mujoco")
    with pytest.raises(ValueError):
        get_backend("mujoco", fidelity="warp")


# --- MuJoCo scene realism ----------------------------------------------------

def _settle(model, data, steps: int) -> None:
    import mujoco

    for _ in range(steps):
        mujoco.mj_step(model, data)


def _scene(gear_scenario):
    """Build the gear-assembly scene or skip if the menagerie isn't checked out."""
    from llm_mr_bt_planner.execution.mujoco_scene import build_scene

    try:
        return build_scene(gear_scenario)
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not checked out")


def test_mujoco_scene_has_realistic_assets(gear_scenario):
    pytest.importorskip("mujoco")
    import mujoco

    model, info = _scene(gear_scenario)

    # Materials, textures and several lights make the workcell read realistically.
    assert model.nmat >= 8, "expected a palette of PBR-ish materials"
    assert model.ntex >= 2, "expected floor + skybox textures"
    assert model.nlight >= 3, "expected a key light plus fills"

    # Every manipulable part from the scenario becomes a shaped free body, and the
    # drawer becomes a real slide joint the skills can drive.
    for part in ("gear", "shaft", "gearbase", "gear_tray", "screwdriver"):
        assert part in info.object_bodies
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part) >= 0
    assert info.drawer_joint is not None
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, info.drawer_joint)
    assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE

    # Each robot has a discovered end-effector + actuated arm chain for IK.
    for robot in ("go2_z1", "franka1", "franka2"):
        arm = info.arms[robot]
        assert arm.ee_body and arm.joint_names


def test_mujoco_parts_rest_flat_on_the_table(gear_scenario):
    pytest.importorskip("mujoco")
    import mujoco

    model, info = _scene(gear_scenario)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _settle(model, data, 250)

    surface = info.surface_z
    for part in info.object_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
        z = float(data.xpos[bid][2])
        # Not fallen through the floor, not launched into the air: resting on top.
        assert surface - 0.05 <= z <= surface + 0.15, f"{part} settled at z={z:.3f}"


def test_mujoco_drawer_opens_and_closes(gear_scenario):
    pytest.importorskip("mujoco")
    import mujoco

    from llm_mr_bt_planner.execution.mujoco_skills import skill_close_drawer, skill_open_drawer

    model, info = _scene(gear_scenario)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, info.drawer_joint)
    qadr = model.jnt_qposadr[jid]

    skill_open_drawer(model, data, {"parameters": ["parts_drawer"]}, info)
    assert abs(float(data.qpos[qadr]) - info.drawer_open_pos) < 1e-6

    skill_close_drawer(model, data, {"parameters": ["parts_drawer"]}, info)
    assert abs(float(data.qpos[qadr])) < 1e-6


def test_mujoco_pick_lifts_and_place_sets_down(gear_scenario):
    pytest.importorskip("mujoco")
    import mujoco

    from llm_mr_bt_planner.execution.mujoco_skills import skill_pick, skill_place

    model, info = _scene(gear_scenario)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _settle(model, data, 60)

    part = "gear"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    resting = float(data.xpos[bid][2])

    # Picking raises the part clear of the table...
    event = {"robot": "franka2", "parameters": [part, "parts_zone"]}
    skill_pick(model, data, event, info)
    mujoco.mj_forward(model, data)
    assert float(data.xpos[bid][2]) > resting + 0.05

    # ...and placing sets it back down onto the surface at the target location.
    skill_place(model, data, event, info)
    mujoco.mj_forward(model, data)
    z = float(data.xpos[bid][2])
    assert info.surface_z - 0.03 <= z <= info.surface_z + 0.08


def test_mujoco_ik_reaches_and_grasps(gear_scenario):
    pytest.importorskip("mujoco")
    import mujoco
    import numpy as np

    from llm_mr_bt_planner.execution.mujoco_ik import ik_reach, snap_held_to_gripper

    model, info = _scene(gear_scenario)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    arm = info.arms["franka1"]
    assert arm.joint_names and arm.ee_body.endswith("hand")

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)
    start = np.array(data.xpos[ee_id]).copy()
    target = start + np.array([0.15, 0.0, -0.2])
    ik_reach(model, data, arm, target)
    assert np.linalg.norm(np.array(data.xpos[ee_id]) - target) < np.linalg.norm(start - target)

    # A held object is snapped onto its gripper (kinematic grasp).
    obj = info.object_bodies[0]
    info.held[obj] = "franka1"
    snap_held_to_gripper(model, data, info)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
    qadr = int(model.jnt_qposadr[model.body_jntadr[bid]])
    assert np.linalg.norm(np.array(data.qpos[qadr : qadr + 3]) - np.array(data.xpos[ee_id])) < 0.1


# --- full-plan execution in physics -----------------------------------------

def test_mujoco_backend_runs_a_correct_plan_in_physics(toy_plan, toy_scenario):
    pytest.importorskip("mujoco")
    backend = get_backend("mujoco", render=False, settle_steps=5)
    result = backend.execute(toy_plan, toy_scenario)
    assert result.backend == "mujoco"
    assert result.success and result.goal_success
    assert result.details["physics_actions"] == 2  # make() then use()
    assert result.details["sim_time"] > 0.0
    # The physical embodiment is a real multi-body workcell.
    assert result.details["model"]["nbody"] > 10
    assert result.details["fidelity"] == "settle"


@pytest.mark.parametrize("fidelity", ["settle", "ik"])
def test_mujoco_backend_executes_the_gear_plan(gear_scenario, fidelity):
    pytest.importorskip("mujoco")
    if fidelity == "ik":
        pytest.importorskip("mink")

    from llm_mr_bt_planner.execution.mujoco_scene import build_scene
    from llm_mr_bt_planner.plan import parse_plan

    saved = RESULTS / "run-gear-ik.json"
    if not saved.exists():
        pytest.skip("saved gear plan not available")
    try:
        build_scene(gear_scenario)
    except FileNotFoundError:
        pytest.skip("mujoco_menagerie not checked out")

    plan = parse_plan(json.loads(saved.read_text())["plan"])
    backend = get_backend("mujoco", render=False, fidelity=fidelity, settle_steps=5)
    result = backend.execute(plan, gear_scenario)

    assert result.success and result.goal_success
    assert result.details["fidelity"] == fidelity
    assert result.details["sim_time"] > 0.0
    # Every executed action drove a physical skill and left a trace entry.
    assert result.details["physics_actions"] >= 8
    assert result.details["physics_actions"] == len(result.details["physics_trace"])
    assert set(info_names := result.details["object_bodies"]) >= {"gear", "shaft", "gearbase"}
    assert info_names  # non-empty


def test_mujoco_scene_declares_grasp_welds_and_drawer_actuator(gear_scenario):
    pytest.importorskip("mujoco")

    model, info = _scene(gear_scenario)
    # A weld equality is pre-declared for every (robot, part) pair.
    assert model.neq >= len(info.object_bodies)
    assert ("franka2", "gear") in info.grasp_welds
    # The drawer is driven by a position actuator (arm pulls it, not by itself).
    assert info.drawer_actuator is not None


def test_mujoco_physics_grasp_carries_a_part(gear_scenario):
    pytest.importorskip("mujoco")
    pytest.importorskip("mink")
    import mujoco

    from llm_mr_bt_planner.execution.mujoco_physics import ArmController

    model, info = _scene(gear_scenario)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(80):
        mujoco.mj_step(model, data)  # let parts settle onto the table first
    ctrl = ArmController(model, data, info)

    part = "gear"
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, part)
    z0 = float(data.xpos[bid][2])
    ctrl._pick("franka2", part)

    # The part is attached to the gripper and carried up off the table (no teleport).
    assert info.held.get(part) == "franka2"
    assert float(data.xpos[bid][2]) > z0 + 0.03
    # It tracks the gripper (rigid kinematic grasp).
    ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, info.arms["franka2"].ee_body)
    import numpy as np
    assert np.linalg.norm(np.array(data.xpos[bid]) - np.array(data.xpos[ee])) < 0.2
