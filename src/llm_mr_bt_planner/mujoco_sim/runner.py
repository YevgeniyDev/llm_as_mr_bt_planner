"""Command runner for physical MuJoCo BT execution."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mujoco

from ..artifacts import load_plan_file
from ..bt import iter_nodes
from ..config import PROJECT_ROOT, save_json
from ..domain import load_scenario
from ..validation import validate_plan
from .assets import MENAGERIE_COMMIT, ensure_assets
from .camera_director import (
    ActionCameraDirector,
    CameraDecision,
    camera_director_for_task,
)
from .controllers import ArmController, ContactGaitController, build_arm_controllers
from .executor import ExecutionReport, PhysicalExecutor
from .recording import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    RecordingConfig,
    RecordingMetadata,
    SimulationVideoRecorder,
)
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
    "three_robot_spare_part_recovery": {
        "pick_source_part",
        "place_source_cradle",
        "pick_source_cradle",
        "navigate_destination",
        "place_destination_cradle",
        "stow_arm_destination",
        "pick_destination_cradle",
        "install_target",
    },
}


def run_cli(args: Namespace) -> int:
    _validate_recording_cli_args(args)
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

    output_dir: Path | None = None
    recording_metadata: RecordingMetadata | None = None
    if args.record_video:
        output_dir = _new_output_directory(Path(args.output), scenario.task_id)
        _copy_recording_inputs(output_dir, Path(args.scenario), bt_path)
        camera_director = (
            camera_director_for_task(scenario.task_id) if args.video_camera is None else None
        )
        recording_camera = (
            camera_director.program.fallback
            if camera_director is not None
            else args.video_camera
        )
        assert recording_camera is not None
        config = RecordingConfig(
            path=output_dir / "simulation.mp4",
            fps=DEFAULT_VIDEO_FPS if args.video_fps is None else args.video_fps,
            width=DEFAULT_VIDEO_WIDTH if args.video_width is None else args.video_width,
            height=DEFAULT_VIDEO_HEIGHT if args.video_height is None else args.video_height,
            camera=recording_camera,
            camera_mode="action_directed" if camera_director is not None else "fixed",
            camera_sequence=camera_director.cameras if camera_director is not None else (),
        )
        recorder = SimulationVideoRecorder(world.model, config)
        with recorder:
            mujoco.mj_forward(world.model, world.data)
            initial_camera = _recording_camera_decision(executor, recorder, camera_director)
            recorder.capture_initial(
                world.data,
                camera=initial_camera.camera,
                reason=initial_camera.reason,
            )
            report = _execute(
                executor,
                arms,
                gait,
                args=args,
                recorder=recorder,
                camera_director=camera_director,
            )
            final_camera = _recording_camera_decision(executor, recorder, camera_director)
            recorder.finish(
                world.data,
                camera=final_camera.camera,
                reason=final_camera.reason,
            )
        recording_metadata = recorder.metadata
    else:
        report = _execute(executor, arms, gait, args=args)

    if output_dir is None:
        output_dir = _new_output_directory(Path(args.output), scenario.task_id)
    report_path = output_dir / "physical_execution_report.json"
    recording_payload: dict[str, Any] | None = None
    if recording_metadata is not None:
        recording_payload = recording_metadata.to_dict()
        recording_payload["sha256"] = _sha256_file(output_dir / recording_metadata.file)
        recording_payload["ffmpeg_version"] = _ffmpeg_version()
    save_json(
        report_path,
        {
            "scenario": "scenario.json" if recording_metadata else str(Path(args.scenario).resolve()),
            "behavior_tree": "behavior_tree.json" if recording_metadata else str(bt_path.resolve()),
            "asset_source": "google-deepmind/mujoco_menagerie",
            "asset_commit": MENAGERIE_COMMIT,
            **({"recording": recording_payload} if recording_payload else {}),
            **report.to_dict(),
        },
    )
    manifest_path: Path | None = None
    if recording_payload is not None:
        manifest_path = output_dir / "recording_manifest.json"
        _write_recording_manifest(
            manifest_path,
            task_id=scenario.task_id,
            report=report,
            recording=recording_payload,
            output_dir=output_dir,
            command=getattr(args, "invocation", list(sys.argv)),
        )
    print(f"Physical execution report: {report_path.resolve()}")
    if recording_metadata is not None:
        print(f"Video: {(output_dir / recording_metadata.file).resolve()}")
        print(f"Recording manifest: {manifest_path.resolve() if manifest_path else ''}")
    print(f"Result: {'SUCCESS' if report.success else 'FAILURE'} — {report.reason}")
    return 0 if report.success else 1


def _execute(
    executor: PhysicalExecutor,
    arms: dict[str, ArmController],
    gait: ContactGaitController,
    *,
    args: Namespace,
    recorder: SimulationVideoRecorder | None = None,
    camera_director: ActionCameraDirector | None = None,
) -> ExecutionReport:
    world = executor.world
    print("Settling actuator-controlled robots under gravity and contact for 1.0 simulated second.")
    settle_steps = round(1.0 / world.model.opt.timestep)
    for _ in range(settle_steps):
        gait.step()
        for arm in arms.values():
            arm.hold()
        mujoco.mj_step(world.model, world.data)
        if recorder is not None:
            _capture_recording_frame(executor, recorder, camera_director)
    if not gait.upright():
        raise RuntimeError(
            "Go2 did not settle upright in the composed scene; physical BT execution was not started."
        )

    print(
        "Ticking the exact hierarchical BT against measured physics and explicit verified signals; "
        "capability effects are not blindly applied."
    )
    if args.headless:
        return _loop(
            executor,
            max_seconds=args.max_seconds,
            recorder=recorder,
            camera_director=camera_director,
        )
    return _viewer_loop(
        executor,
        max_seconds=args.max_seconds,
        realtime_factor=args.realtime_factor,
        recorder=recorder,
        camera_director=camera_director,
    )


def _loop(
    executor: PhysicalExecutor,
    *,
    max_seconds: float,
    recorder: SimulationVideoRecorder | None = None,
    camera_director: ActionCameraDirector | None = None,
) -> ExecutionReport:
    start_sim_time = float(executor.world.data.time)
    timestep = float(executor.world.model.opt.timestep)
    while not executor.complete and not executor.failed:
        if float(executor.world.data.time) - start_sim_time > max_seconds:
            return executor.make_report(f"Physical run exceeded the {max_seconds:.1f}s simulation limit.")
        executor.step(timestep)
        mujoco.mj_step(executor.world.model, executor.world.data)
        if recorder is not None:
            _capture_recording_frame(executor, recorder, camera_director)
    return executor.make_report()


def _viewer_loop(
    executor: PhysicalExecutor,
    *,
    max_seconds: float,
    realtime_factor: float,
    recorder: SimulationVideoRecorder | None = None,
    camera_director: ActionCameraDirector | None = None,
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
            if recorder is not None:
                _capture_recording_frame(executor, recorder, camera_director)
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
            "The physical adapter supports the bundled courier, packaging-delivery, "
            "and spare-part-recovery task IDs with robots franka_a, unitree_go2_z1, "
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


def _default_recording_camera(task_id: str) -> str:
    return camera_director_for_task(task_id).program.fallback


def _recording_camera_decision(
    executor: PhysicalExecutor,
    recorder: SimulationVideoRecorder,
    camera_director: ActionCameraDirector | None,
) -> CameraDecision:
    if camera_director is None:
        return CameraDecision(recorder.config.camera, "fixed_camera_override")
    return camera_director.update(executor.events)


def _capture_recording_frame(
    executor: PhysicalExecutor,
    recorder: SimulationVideoRecorder,
    camera_director: ActionCameraDirector | None,
) -> None:
    decision = _recording_camera_decision(executor, recorder, camera_director)
    recorder.capture_after_step(
        executor.world.data,
        camera=decision.camera,
        reason=decision.reason,
    )


def _validate_recording_cli_args(args: Namespace) -> None:
    video_options = (args.video_fps, args.video_width, args.video_height, args.video_camera)
    if not args.record_video and any(value is not None for value in video_options):
        raise ValueError(
            "--video-fps, --video-width, --video-height, and --video-camera require --record-video."
        )
    if args.setup_only and args.record_video:
        raise ValueError("--record-video cannot be combined with --setup-only.")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("--video-fps must be greater than zero.")
    for option, value in (
        ("--video-width", args.video_width),
        ("--video-height", args.video_height),
    ):
        if value is not None and (value <= 0 or value % 2):
            raise ValueError(f"{option} must be a positive even integer.")


def _copy_recording_inputs(output_dir: Path, scenario_path: Path, bt_path: Path) -> None:
    shutil.copy2(scenario_path.resolve(), output_dir / "scenario.json")
    shutil.copy2(bt_path.resolve(), output_dir / "behavior_tree.json")


def _write_recording_manifest(
    path: Path,
    *,
    task_id: str,
    report: ExecutionReport,
    recording: dict[str, Any],
    output_dir: Path,
    command: list[str] | None = None,
) -> None:
    files = (
        output_dir / "simulation.mp4",
        output_dir / "scenario.json",
        output_dir / "behavior_tree.json",
        output_dir / "physical_execution_report.json",
    )
    source_commit, source_dirty = _source_revision()
    save_json(
        path,
        {
            "manifest_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "run_success": report.success,
            "command": list(sys.argv) if command is None else command,
            "recording": recording,
            "simulation": {
                "timestep_seconds": report.physics["timestep_seconds"],
                "simulated_seconds": report.simulated_seconds,
            },
            "software": {
                "project_version": _distribution_version("llm-mr-bt-planner"),
                "python_version": platform.python_version(),
                "mujoco_version": getattr(mujoco, "__version__", "unknown"),
                "menagerie_commit": MENAGERIE_COMMIT,
                "source_commit": source_commit,
                "source_worktree_dirty": source_dirty,
            },
            "files": {file.name: _sha256_file(file) for file in files},
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _ffmpeg_version() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_version()
    except (ImportError, RuntimeError):
        return "unknown"


def _source_revision() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return commit or None, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None
