# ruff: noqa: E402 -- optional module-level skip must precede simulator imports.

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from llm_mr_bt_planner.artifacts import load_plan_file
from llm_mr_bt_planner.config import PROJECT_ROOT
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.mujoco_sim.assets import _is_valid, default_asset_root
from llm_mr_bt_planner.mujoco_sim.controllers import ContactGaitController, build_arm_controllers
from llm_mr_bt_planner.mujoco_sim.executor import (
    PANDA_GRASP_ROTATIONS,
    PhysicalExecutor,
)
from llm_mr_bt_planner.mujoco_sim.runner import _configure_free_camera
from llm_mr_bt_planner.mujoco_sim.world import (
    DESTINATION_DOCK_X,
    DOCK_Y,
    PACKAGING_STATION_POSES,
    PANDA_MOUNT_POSES,
    SOURCE_DOCK_X,
    STATION_PAD_HALF_EXTENTS,
    STATION_POSES,
    CourierWorld,
)


@pytest.fixture(scope="module")
def menagerie_assets() -> Path:
    configured = os.environ.get("LMRBTP_MUJOCO_ASSETS")
    root = Path(configured).expanduser().resolve() if configured else default_asset_root()
    if not _is_valid(root):
        pytest.skip("Pinned MuJoCo assets are not cached; run `lmrbtp mujoco --setup-only`.")
    return root


def test_composed_scene_has_three_isolated_robots_and_dynamic_payload(menagerie_assets: Path):
    world = CourierWorld.build(menagerie_assets)

    assert world.model.body("franka_a_link0").id >= 0
    assert world.model.body("franka_b_link0").id >= 0
    assert world.model.body("go2_base").id >= 0
    assert world.model.body("go2_z1_link00").id >= 0
    assert world.model.geom("source_worktable_top").id >= 0
    assert world.model.geom("destination_worktable_top").id >= 0
    assert world.model.nu == 35
    assert world.model.body("payload").jntnum == 1
    assert mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, "go2_transport_carrier") == -1
    assert len(world.z1_finger_pad_geoms["fixed"]) == 2
    assert len(world.z1_finger_pad_geoms["moving"]) == 2
    payload_joint = int(world.model.body("payload").jntadr[0])
    assert world.model.jnt_type[payload_joint] == mujoco.mjtJoint.mjJNT_FREE
    assert world.qpos_writes_after_reset == 0
    assert not any(world.equality_active(owner) for owner in world.grip_equalities)

    zone_delta = np.abs(STATION_POSES["destination_cradle"][:2] - STATION_POSES["target_fixture"][:2])
    zone_extent_sum = (
        STATION_PAD_HALF_EXTENTS["destination_cradle"][:2]
        + STATION_PAD_HALF_EXTENTS["target_fixture"][:2]
    )
    assert np.any(zone_delta > zone_extent_sum), "green destination pad overlaps the red target fixture"


def test_packaging_scene_contains_independent_parts_and_a_closed_dynamic_door(
    menagerie_assets: Path,
):
    world = CourierWorld.build(
        menagerie_assets,
        task_id="three_robot_packaging_delivery",
    )

    assert set(PACKAGING_STATION_POSES) - {"lid_seal_target"} <= set(world.station_sites)
    assert world.model.body("package_lid").jntnum == 1
    assert world.model.joint("room_door_hinge").type == mujoco.mjtJoint.mjJNT_HINGE
    assert world.door_closed()
    assert not world.door_open()
    assert not world.equality_active("package_seal")
    assert world.model.geom("room_door_panel").id >= 0
    assert world.model.geom("delivery_pedestal_top").id >= 0


def test_physical_blackboard_canonicalizes_llm_predicate_whitespace(
    menagerie_assets: Path,
):
    world = CourierWorld.build(
        menagerie_assets,
        task_id="three_robot_packaging_delivery",
    )
    arms = build_arm_controllers(world)
    gait = ContactGaitController(world)
    scenario = load_scenario(
        PROJECT_ROOT / "examples" / "three_robot_packaging_delivery.json",
        strict=True,
    )
    plan = load_plan_file(
        PROJECT_ROOT / "examples" / "three_robot_packaging_delivery.bt.json"
    )
    executor = PhysicalExecutor(
        world,
        scenario,
        plan,
        arms,
        gait,
        progress=lambda _message: None,
    )
    executor._add_signal("assembly_audited(package_base,package_lid)")

    assert executor.observe_literal("assembly_audited(package_base,package_lid)")
    assert executor.observe_literal("assembly_audited(package_base, package_lid)")


def test_pandas_are_bench_mounted_and_target_sites_are_hidden(menagerie_assets: Path):
    world = CourierWorld.build(menagerie_assets)

    for robot, station in (("franka_a", "source"), ("franka_b", "destination")):
        mount_pose = PANDA_MOUNT_POSES[robot]
        assert world.data.body(f"{robot}_link0").xpos == pytest.approx(mount_pose)

        plate_id = world.model.geom(f"{station}_panda_mounting_plate").id
        plate_top_z = world.data.geom_xpos[plate_id, 2] + world.model.geom_size[plate_id, 1]
        assert plate_top_z == pytest.approx(mount_pose[2])

        top_id = world.model.geom(f"{station}_worktable_top").id
        top_center = world.data.geom_xpos[top_id, :2]
        top_half_size = world.model.geom_size[top_id, :2]
        plate_radius = world.model.geom_size[plate_id, 0]
        assert np.all(np.abs(mount_pose[:2] - top_center) + plate_radius <= top_half_size)

        rotation = PANDA_GRASP_ROTATIONS[robot]
        assert rotation[:, 2] == pytest.approx([0.0, 0.0, -1.0])
        assert np.linalg.det(rotation) == pytest.approx(1.0)

    for station in STATION_POSES:
        site = world.model.site(f"station_{station}")
        assert world.model.site_rgba[site.id, 3] == 0.0
        assert mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_GEOM, site.name) == -1


def test_workcells_have_opposite_side_flow_and_a_long_clear_route(menagerie_assets: Path):
    world = CourierWorld.build(menagerie_assets)

    assert PANDA_MOUNT_POSES["franka_a"][0] < STATION_POSES["source_bin"][0]
    assert PANDA_MOUNT_POSES["franka_a"][0] < STATION_POSES["source_cradle"][0]
    assert PANDA_MOUNT_POSES["franka_b"][0] > STATION_POSES["destination_cradle"][0]
    assert PANDA_MOUNT_POSES["franka_b"][0] > STATION_POSES["target_fixture"][0]
    assert STATION_POSES["source_bin"][1] < STATION_POSES["source_cradle"][1]
    assert STATION_POSES["target_fixture"][1] < STATION_POSES["destination_cradle"][1]
    assert STATION_POSES["source_bin"][1] == pytest.approx(STATION_POSES["target_fixture"][1])
    assert STATION_POSES["source_cradle"][1] == pytest.approx(
        STATION_POSES["destination_cradle"][1]
    )

    assert SOURCE_DOCK_X == pytest.approx(STATION_POSES["source_cradle"][0])
    assert DESTINATION_DOCK_X == pytest.approx(STATION_POSES["destination_cradle"][0])
    assert DESTINATION_DOCK_X - SOURCE_DOCK_X > 2.5
    assert DOCK_Y > STATION_POSES["source_cradle"][1]

    station_to_bench = {
        "source_bin": "source",
        "source_cradle": "source",
        "destination_cradle": "destination",
        "target_fixture": "destination",
    }
    for station, bench in station_to_bench.items():
        top_id = world.model.geom(f"{bench}_worktable_top").id
        top_center = world.data.geom_xpos[top_id, :2]
        top_half_size = world.model.geom_size[top_id, :2]
        assert np.all(
            np.abs(STATION_POSES[station][:2] - top_center)
            + STATION_PAD_HALF_EXTENTS[station][:2]
            <= top_half_size
        )


def test_viewer_camera_is_free_and_starts_with_an_overview():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='1'/></worldbody></mujoco>")
    camera = mujoco.MjvCamera()

    _configure_free_camera(model, camera)

    assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert camera.fixedcamid == -1
    assert camera.distance == pytest.approx(6.0)
    assert camera.lookat == pytest.approx([1.5, 0.0, 0.5])
    assert camera.azimuth == pytest.approx(180.0)
    assert camera.elevation == pytest.approx(-35.0)


def test_z1_gripper_uses_measured_open_and_closed_joint_positions(menagerie_assets: Path):
    world = CourierWorld.build(menagerie_assets)
    arms = build_arm_controllers(world)
    gait = ContactGaitController(world)
    z1 = arms["unitree_go2_z1"]

    assert z1.gripper_opened()
    assert z1.gripper_position == pytest.approx(-0.75)

    z1.set_gripper(closed=True)
    for _ in range(500):
        z1.hold()
        gait.step()
        mujoco.mj_step(world.model, world.data)

    assert z1.gripper_closed()
    assert z1.gripper_position == pytest.approx(-0.50, abs=0.06)


@pytest.mark.skipif(
    os.environ.get("LMRBTP_RUN_MUJOCO_E2E") != "1",
    reason="Set LMRBTP_RUN_MUJOCO_E2E=1 to run the 10-20 second physical integration test.",
)
def test_reference_bt_reaches_all_measured_physical_goals(menagerie_assets: Path):
    world = CourierWorld.build(menagerie_assets)
    arms = build_arm_controllers(world)
    gait = ContactGaitController(world)
    scenario = load_scenario(PROJECT_ROOT / "examples" / "three_robot_courier.json", strict=True)
    plan = load_plan_file(PROJECT_ROOT / "examples" / "three_robot_courier.bt.json")
    executor = PhysicalExecutor(world, scenario, plan, arms, gait, progress=lambda _message: None)
    panda_q1_bounds = {
        robot: [float("inf"), float("-inf")] for robot in ("franka_a", "franka_b")
    }
    robot_worktable_contacts: set[tuple[str, str]] = set()

    for _ in range(round(1.0 / world.model.opt.timestep)):
        gait.step()
        for arm in arms.values():
            arm.hold()
        mujoco.mj_step(world.model, world.data)

    while not executor.complete and not executor.failed and world.data.time < 120.0:
        executor.step(float(world.model.opt.timestep))
        mujoco.mj_step(world.model, world.data)
        for robot in panda_q1_bounds:
            q1 = float(arms[robot].q[0])
            panda_q1_bounds[robot][0] = min(panda_q1_bounds[robot][0], q1)
            panda_q1_bounds[robot][1] = max(panda_q1_bounds[robot][1], q1)
        for index in range(world.data.ncon):
            contact = world.data.contact[index]
            body1 = world.model.body(int(world.model.geom_bodyid[contact.geom1])).name
            body2 = world.model.body(int(world.model.geom_bodyid[contact.geom2])).name
            if (
                body1.startswith(("franka_", "go2_"))
                and body2.endswith("_worktable")
            ) or (
                body2.startswith(("franka_", "go2_"))
                and body1.endswith("_worktable")
            ):
                robot_worktable_contacts.add(tuple(sorted((body1, body2))))
    report = executor.make_report()

    assert report.success, report.reason
    assert all(report.final_goals.values())
    assert report.locomotion["contact_driven_displacement_m"] > 2.0
    assert report.locomotion["ground_contact_steps"] > 100
    assert report.locomotion["direct_base_state_writes"] == 0
    assert report.locomotion["base_upright"] is True
    successful_actions = [event for event in report.action_events if event["kind"] == "action_success"]
    assert len(successful_actions) == 8
    navigation = next(event for event in successful_actions if event["message"].startswith("navigate_destination"))
    assert "payload remained in the closed Z1 grasp" in navigation["detail"]
    assert np.max(np.abs(arms["franka_a"].q - arms["franka_a"].home)) < 0.09
    assert np.max(np.abs(arms["franka_b"].q - arms["franka_b"].home)) < 0.09
    assert robot_worktable_contacts == set()
    for minimum, maximum in panda_q1_bounds.values():
        assert minimum < -0.15
        assert maximum > 0.15
        assert maximum - minimum > np.deg2rad(45.0)


@pytest.mark.skipif(
    os.environ.get("LMRBTP_RUN_MUJOCO_E2E") != "1",
    reason="Set LMRBTP_RUN_MUJOCO_E2E=1 to run the packaging physical integration test.",
)
def test_packaging_reference_assembles_opens_door_and_delivers(menagerie_assets: Path):
    task_id = "three_robot_packaging_delivery"
    world = CourierWorld.build(menagerie_assets, task_id=task_id)
    arms = build_arm_controllers(world)
    gait = ContactGaitController(world)
    scenario = load_scenario(
        PROJECT_ROOT / "examples" / "three_robot_packaging_delivery.json",
        strict=True,
    )
    plan = load_plan_file(
        PROJECT_ROOT / "examples" / "three_robot_packaging_delivery.bt.json"
    )
    executor = PhysicalExecutor(world, scenario, plan, arms, gait, progress=lambda _message: None)

    for _ in range(round(1.0 / world.model.opt.timestep)):
        gait.step()
        for arm in arms.values():
            arm.hold()
        mujoco.mj_step(world.model, world.data)

    while not executor.complete and not executor.failed and world.data.time < 120.0:
        executor.step(float(world.model.opt.timestep))
        mujoco.mj_step(world.model, world.data)
    report = executor.make_report()

    assert report.success, report.reason
    assert all(report.final_goals.values())
    assert report.physics["resources_released"] is True
    evidence = report.physics["packaging_delivery_evidence"]
    assert evidence["door_initially_closed"] is True
    assert evidence["door_physically_open"] is True
    assert evidence["final_door_angle_radians"] > 0.7
    assert evidence["package_seal_constraint_active"] is True
    assert evidence["parcel_physically_delivered"] is True
    door_event = next(
        event
        for event in report.action_events
        if event.get("message", "").startswith("push_open_door_and_cross")
        and event["kind"] == "action_success"
    )
    assert "sealed parcel remained in the Z1 grasp" in door_event["detail"]
    assert report.locomotion["contact_driven_displacement_m"] > 2.0
    assert report.locomotion["direct_base_state_writes"] == 0
