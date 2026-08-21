# ruff: noqa: E402 -- optional module-level skip must precede simulator imports.

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from llm_mr_bt_planner.artifacts import load_plan_file
from llm_mr_bt_planner.cli import _build_parser
from llm_mr_bt_planner.config import PROJECT_ROOT
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.mujoco_sim import adaptive_demo_runner
from llm_mr_bt_planner.mujoco_sim.adaptive_demo_runner import (
    _fault_blindness_evidence,
    validate_nominal_primary_only,
)
from llm_mr_bt_planner.mujoco_sim.assets import _is_valid, default_asset_root
from llm_mr_bt_planner.mujoco_sim.controllers import ContactGaitController, build_arm_controllers
from llm_mr_bt_planner.mujoco_sim.executor import (
    PANDA_GRASP_ROTATIONS,
    PhysicalExecutor,
)
from llm_mr_bt_planner.mujoco_sim.recovery_runner import run_recovery_cli
from llm_mr_bt_planner.mujoco_sim.runner import _configure_free_camera
from llm_mr_bt_planner.mujoco_sim.world import (
    DESTINATION_DOCK_X,
    DOCK_Y,
    PACKAGING_STATION_POSES,
    PANDA_MOUNT_POSES,
    RECOVERY_STATION_POSES,
    SOURCE_DOCK_X,
    STATION_PAD_HALF_EXTENTS,
    STATION_POSES,
    CourierWorld,
)
from llm_mr_bt_planner.prompts import build_prompt


def test_adaptive_demo_nominal_gate_rejects_a_preplanned_spare_branch():
    nominal = load_plan_file(
        PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.bt.json"
    )
    recovery = load_plan_file(
        PROJECT_ROOT
        / "examples"
        / "three_robot_spare_part_recovery.expected_recovery.bt.json"
    )

    actions = validate_nominal_primary_only(nominal)

    assert any("primary_part" in action["parameters"] for action in actions)
    assert not any("spare_part" in action["parameters"] for action in actions)
    with pytest.raises(ValueError, match="must not contain a preplanned recovery branch"):
        validate_nominal_primary_only(recovery)


def test_adaptive_demo_nominal_prompt_contains_no_fault_identifier():
    scenario = load_scenario(
        PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.json",
        strict=True,
    )
    fault_path = (
        PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.fault.json"
    )

    evidence = _fault_blindness_evidence(
        nominal_user_prompt=build_prompt(scenario),
        fault_path=fault_path,
        fault_id="drop_primary_after_handoff_placement",
        nominal_actions=validate_nominal_primary_only(
            load_plan_file(
                PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.bt.json"
            )
        ),
        nominal_completed=1.0,
        fault_loaded=2.0,
    )

    assert evidence["boundary_verified"] is True
    assert evidence["fault_disclosed_to_nominal_llm"] is False
    assert evidence["fault_configuration_loaded_after_nominal_bt_accepted"] is True
    assert evidence["fault_id_present_in_nominal_prompt"] is False
    assert len(evidence["nominal_request_prompt_sha256"]) == 64
    assert len(evidence["fault_specification_sha256"]) == 64


def test_adaptive_demo_orchestrates_generation_before_fault_and_publishes_audit_bundle(
    monkeypatch,
    tmp_path: Path,
):
    scenario_path = (
        PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.json"
    )
    fault_path = (
        PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.fault.json"
    )
    nominal = load_plan_file(
        PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.bt.json"
    )
    adapted = load_plan_file(
        PROJECT_ROOT
        / "examples"
        / "three_robot_spare_part_recovery.expected_recovery.bt.json"
    )
    scenario = load_scenario(scenario_path, strict=True)
    order: list[str] = []

    def fake_generate(self, document, **kwargs):  # noqa: ARG001
        assert "fault_loaded" not in order
        order.append("nominal_generated")
        kwargs["progress"]("Initial candidate: provider response received", 0.5)
        return SimpleNamespace(
            plan=nominal,
            validation=SimpleNamespace(valid=True),
            simulation=SimpleNamespace(success=True),
            planner_result=SimpleNamespace(
                plan=nominal.to_dict(),
                provider="openai",
                provider_responses=(
                    {"model_returned": "gpt-5.6-sol", "response_id": "chat_test"},
                ),
                correction_rounds=0,
            ),
        )

    original_load_fault = adaptive_demo_runner.load_fault_spec

    def tracked_load_fault(path):
        assert order == ["nominal_generated"]
        order.append("fault_loaded")
        return original_load_fault(path)

    def fake_plan_recovery(client, runtime_scenario, **kwargs):  # noqa: ARG001
        order.append("recovery_generated")
        kwargs["progress"]("Validated continuation BT is ready for MuJoCo", 1.0)
        return SimpleNamespace(
            plan=adapted,
            runtime_scenario=scenario,
            validation=SimpleNamespace(valid=True, to_dicts=lambda: []),
            simulation=SimpleNamespace(to_dict=lambda: {"success": True}),
            attempts=(
                {
                    "provenance": {
                        "model_returned": "gpt-5.6-sol",
                        "response_id": "resp_test",
                    }
                },
            ),
            provider="openai",
            model="gpt-5.6-sol",
            reasoning_effort="high",
        )

    def fake_trial(
        scenario_arg,
        nominal_arg,
        fault_arg,
        *,
        continue_after_failure,
        **kwargs,
    ):  # noqa: ARG001
        world = object()
        executor = SimpleNamespace(world=world, arms={}, gait=object())
        observation = {
            "classification": "dropped_to_floor",
            "object": "primary_part",
        }
        _, continuity = continue_after_failure(
            executor,
            ("usable(spare_part)", "at(spare_part,backup_bin)"),
            observation,
            {"world_identity": id(world)},
        )
        continuity["no_reset_through_completion"] = True
        return {
            "physical_execution": {"success": True},
            "failure_observation": observation,
            "failure_snapshot": {"state": "preserved"},
            "continuity": continuity,
            "recording": None,
            "live_viewer": {
                "enabled": True,
                "camera_mode": "action_directed",
                "realtime_factor": 1.0,
            },
        }

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(adaptive_demo_runner.PlannerService, "generate", fake_generate)
    monkeypatch.setattr(adaptive_demo_runner, "load_fault_spec", tracked_load_fault)
    monkeypatch.setattr(adaptive_demo_runner, "ensure_assets", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(adaptive_demo_runner, "plan_recovery", fake_plan_recovery)
    monkeypatch.setattr(adaptive_demo_runner, "_run_fault_trial", fake_trial)
    monkeypatch.setattr(
        adaptive_demo_runner,
        "_world_snapshot",
        lambda *args, **kwargs: {"state": "preserved"},
    )
    monkeypatch.setattr(
        adaptive_demo_runner,
        "_continuity_invariant",
        lambda world: {"world_identity": id(world)},
    )
    monkeypatch.setattr(
        adaptive_demo_runner,
        "PhysicalExecutor",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    args = _build_parser().parse_args(
        [
            "adaptive-demo",
            "--scenario",
            str(scenario_path),
            "--fault",
            str(fault_path),
            "--output",
            str(tmp_path / "runs"),
            "--heartbeat-seconds",
            "60",
            "--no-video",
        ]
    )

    assert adaptive_demo_runner.run_adaptive_demo_cli(args) == 0
    assert order == ["nominal_generated", "fault_loaded", "recovery_generated"]
    output_dir = next((tmp_path / "runs").iterdir())
    manifest = json.loads(
        (output_dir / "adaptive_demo_manifest.json").read_text(encoding="utf-8")
    )
    prompt_bytes = (output_dir / "nominal_user_prompt.txt").read_bytes()
    assert manifest["success"] is True
    assert manifest["fault_blindness"]["boundary_verified"] is True
    assert manifest["fault_blindness"]["nominal_prompt_file_sha256"] == sha256(
        prompt_bytes
    ).hexdigest()
    assert manifest["nominal_planner"]["model_verified"] is True
    assert manifest["recovery_planner"]["model_verified"] is True
    assert manifest["live_viewer"]["enabled"] is True
    assert "adaptive_demo.log" in manifest["files"]
    assert "adaptive_demo_events.json" in manifest["files"]


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


def test_recovery_scene_has_independent_primary_and_spare_parts(
    menagerie_assets: Path,
):
    world = CourierWorld.build(
        menagerie_assets,
        task_id="three_robot_spare_part_recovery",
    )

    assert set(world.object_body_ids) == {"primary_part", "spare_part"}
    assert world.model.body("primary_part").jntnum == 1
    assert world.model.body("spare_part").jntnum == 1
    assert world.object_position("primary_part") == pytest.approx(
        RECOVERY_STATION_POSES["primary_bin"]
    )
    assert world.object_position("spare_part") == pytest.approx(
        RECOVERY_STATION_POSES["backup_bin"]
    )
    assert world.active_payload_name == "primary_part"
    assert world.reset_count == 1
    assert "franka_a:primary_part" in world.grip_equalities
    assert "franka_a:spare_part" in world.grip_equalities
    assert "target_fixture:primary_part" in world.grip_equalities
    assert "target_fixture:spare_part" in world.grip_equalities
    assert not any(world.equality_active(key) for key in world.grip_equalities)

    world.activate_weld("franka_a:spare_part")
    assert world.active_payload_name == "spare_part"
    assert world.equality_active("franka_a:spare_part")
    assert not world.equality_active("franka_a:primary_part")


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


def test_composed_world_supports_publication_recording_settings(menagerie_assets: Path):
    recording_cameras = {
        "overview",
        "packaging_recording",
        "courier_source",
        "courier_route",
        "courier_destination",
        "packaging_assembly",
        "packaging_door",
        "packaging_route",
        "packaging_delivery",
        "recovery_source",
        "recovery_floor",
        "recovery_route",
        "recovery_destination",
    }
    for task_id in (
        "three_robot_courier",
        "three_robot_packaging_delivery",
        "three_robot_spare_part_recovery",
    ):
        world = CourierWorld.build(menagerie_assets, task_id=task_id)

        assert all(world.model.camera(name).id >= 0 for name in recording_cameras)
        assert world.model.vis.global_.offwidth >= 1920
        assert world.model.vis.global_.offheight >= 1080


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


@pytest.mark.skipif(
    os.environ.get("LMRBTP_RUN_MUJOCO_RECOVERY_E2E") != "1",
    reason=(
        "Set LMRBTP_RUN_MUJOCO_RECOVERY_E2E=1 to run the fault/control/same-state "
        "oracle recovery integration test."
    ),
)
def test_spare_part_recovery_continues_without_reset(
    menagerie_assets: Path,
    tmp_path: Path,
):
    args = _build_parser().parse_args(
        [
            "recovery-experiment",
            "--planner",
            "oracle",
            "--no-video",
            "--assets-dir",
            str(menagerie_assets),
            "--output",
            str(tmp_path),
        ]
    )

    assert run_recovery_cli(args) == 0
    run_dir = next(tmp_path.iterdir())
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    adaptive = json.loads(
        (run_dir / "adaptive_recovery_report.json").read_text(encoding="utf-8")
    )

    assert manifest["success"] is False
    assert manifest["planner"]["real_llm_evidence"] is False
    assert manifest["control"]["failed_as_expected"] is True
    assert manifest["adaptive"]["recovered"] is True
    assert adaptive["failure_observation"]["classification"] == "dropped_to_floor"
    assert adaptive["failure_observation"]["trigger_evidence"]["placed_at_location"] is True
    assert (
        adaptive["failure_observation"]["trigger_evidence"]["placement_event"]["kind"]
        == "action_success"
    )
    assert (
        adaptive["failure_observation"]["trigger_evidence"]["next_robot_holding_object"]
        is False
    )
    assert "unitree_go2_z1/pick_source_cradle" in adaptive["failure_observation"][
        "nominal_bt_failure"
    ]
    assert adaptive["continuity"]["state_hash_unchanged_while_replanning"] is True
    assert adaptive["continuity"]["no_reset_through_completion"] is True
    assert all(adaptive["physical_execution"]["final_goals"].values())
