"""Command runner for physical MuJoCo BT execution."""

from __future__ import annotations

import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import mujoco

from ..artifacts import load_plan_file
from ..bt import iter_nodes
from ..config import save_json
from ..domain import load_scenario
from ..validation import validate_plan
from .assets import MENAGERIE_COMMIT, ensure_assets
from .controllers import ContactGaitController, build_arm_controllers
from .executor import ExecutionReport, PhysicalExecutor
from .world import CourierWorld

SUPPORTED_ROBOTS = {"franka_a", "unitree_go2_z1", "franka_b"}
SUPPORTED_ACTIONS = {
    "three_robot_courier": {
        "pick_source",
        "place_source_cradle",
        "pick_source_cradle",
        "navigate_destination",
        "place_destination_cradle",
        "stow_arm_destination",
        "pick_destination_cradle",
        "install_target",
    },
    "three_robot_packaging_delivery": {
        "pick_loaded_package_base",
        "place_base_at_packing_station",
        "pick_package_lid",
        "fit_and_seal_package_lid",
        "verify_delivery_readiness",
        "pick_sealed_parcel",
        "approach_closed_room_door",
        "push_open_door_and_cross",
        "cross_already_open_door",
        "navigate_delivery_room",
        "place_parcel_at_delivery_station",
        "stow_after_delivery",
    },
}


def run_cli(args: Namespace) -> int:
    assets = ensure_assets(args.assets_dir, progress=print)
    if args.setup_only:
        print(f"Pinned asset commit: {MENAGERIE_COMMIT}")
        return 0
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be greater than zero.")

    scenario = load_scenario(args.scenario, strict=True)
    bt_path = Path(args.bt) if args.bt else Path(args.scenario).with_suffix(".bt.json")
    if not bt_path.is_file():
        raise ValueError(
            f"No default BT exists beside the selected scenario ({bt_path}). "
            "Pass --bt with an LLM-generated behavior_tree.json file."
        )
    plan = load_plan_file(bt_path)
    validation = validate_plan(plan, scenario, suggest_producers=True)
    if not validation.valid:
        first = "; ".join(error.message for error in validation.errors[:4])
        raise ValueError(f"The BT failed static validation and cannot enter MuJoCo: {first}")
    _check_adapter_scope(scenario, plan)

    print(
        f"Building one MuJoCo model for {scenario.task_id}: "
        "Panda A + Go2/Z1 + Panda B + dynamic task objects."
    )
    world = CourierWorld.build(assets, task_id=scenario.task_id)
    arms = build_arm_controllers(world)
    gait = ContactGaitController(world)
    executor = PhysicalExecutor(world, scenario, plan, arms, gait, progress=print)

    print("Settling actuator-controlled robots under gravity and contact for 1.0 simulated second.")
    settle_steps = round(1.0 / world.model.opt.timestep)
    for _ in range(settle_steps):
        gait.step()
        for arm in arms.values():
            arm.hold()
        mujoco.mj_step(world.model, world.data)
    if not gait.upright():
        raise RuntimeError("Go2 did not settle upright in the composed scene; physical BT execution was not started.")

    print(
        "Ticking the exact hierarchical BT against measured physics and explicit verified signals; "
        "capability effects are not blindly applied."
    )
    if args.headless:
        report = _loop(executor, max_seconds=args.max_seconds)
    else:
        report = _viewer_loop(executor, max_seconds=args.max_seconds, realtime_factor=args.realtime_factor)

    output_dir = _new_output_directory(Path(args.output), scenario.task_id)
    report_path = output_dir / "physical_execution_report.json"
    save_json(
        report_path,
        {
            "scenario": str(Path(args.scenario).resolve()),
            "behavior_tree": str(bt_path.resolve()),
            "asset_source": "google-deepmind/mujoco_menagerie",
            "asset_commit": MENAGERIE_COMMIT,
            **report.to_dict(),
        },
    )
    print(f"Physical execution report: {report_path.resolve()}")
    print(f"Result: {'SUCCESS' if report.success else 'FAILURE'} — {report.reason}")
    return 0 if report.success else 1


def _loop(executor: PhysicalExecutor, *, max_seconds: float) -> ExecutionReport:
    start_sim_time = float(executor.world.data.time)
    timestep = float(executor.world.model.opt.timestep)
    while not executor.complete and not executor.failed:
        if float(executor.world.data.time) - start_sim_time > max_seconds:
            return executor.make_report(f"Physical run exceeded the {max_seconds:.1f}s simulation limit.")
        executor.step(timestep)
        mujoco.mj_step(executor.world.model, executor.world.data)
    return executor.make_report()


def _viewer_loop(
    executor: PhysicalExecutor, *, max_seconds: float, realtime_factor: float
) -> ExecutionReport:
    if realtime_factor <= 0:
        raise ValueError("--realtime-factor must be greater than zero.")
    try:
        import mujoco.viewer
    except ImportError as error:
        raise RuntimeError("This MuJoCo installation does not include the passive viewer.") from error

    world = executor.world
    timestep = float(world.model.opt.timestep)
    start_sim_time = float(world.data.time)
    with mujoco.viewer.launch_passive(
        world.model,
        world.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        _configure_free_camera(world.model, viewer.cam)
        while viewer.is_running() and not executor.complete and not executor.failed:
            if float(world.data.time) - start_sim_time > max_seconds:
                return executor.make_report(f"Physical run exceeded the {max_seconds:.1f}s simulation limit.")
            wall_start = time.perf_counter()
            executor.step(timestep)
            mujoco.mj_step(world.model, world.data)
            viewer.sync()
            remaining = timestep / realtime_factor - (time.perf_counter() - wall_start)
            if remaining > 0:
                time.sleep(remaining)
        if not viewer.is_running() and not executor.complete and not executor.failed:
            return executor.make_report("The MuJoCo viewer was closed before BT execution completed.")
    return executor.make_report()


def _configure_free_camera(model: mujoco.MjModel, camera: mujoco.MjvCamera) -> None:
    """Start with a useful overview while preserving interactive camera controls."""
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.lookat[:] = [1.5, 0.0, 0.5]
    camera.distance = 6.0
    camera.azimuth = 180.0
    camera.elevation = -35.0


def _check_adapter_scope(scenario, plan) -> None:
    supported = SUPPORTED_ACTIONS.get(scenario.task_id)
    if supported is None or set(plan.behavior_trees) != SUPPORTED_ROBOTS:
        raise ValueError(
            "The physical adapter supports task_id 'three_robot_courier' and "
            "'three_robot_packaging_delivery' with robots franka_a, unitree_go2_z1, "
            "and franka_b. It does not silently reinterpret other scenarios."
        )
    for robot, root in plan.behavior_trees.items():
        for node in iter_nodes(root):
            if node.children and node.type not in {"Sequence", "Fallback"}:
                raise ValueError(
                    f"The physical adapter supports Sequence and Fallback composites; "
                    f"{robot}/{node.node_id} uses {node.type}. Symbolic validation remains available for that BT."
                )
            if node.type == "Action" and node.name not in supported:
                raise ValueError(f"No physical controller mapping exists for {robot}/{node.name}.")


def _new_output_directory(root: Path, task_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = root.resolve() / f"{task_id.replace('_', '-')}-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory
