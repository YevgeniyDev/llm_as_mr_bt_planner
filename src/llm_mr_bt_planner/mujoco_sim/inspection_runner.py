"""CLI runner for the five-agent inspection MuJoCo adapter."""

from __future__ import annotations

import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, cast

import mujoco

from ..artifacts import load_plan_file
from ..bt import iter_nodes
from ..config import save_json
from ..validation import validate_plan
from .assets import MENAGERIE_COMMIT, ensure_assets
from .camera_director import ActionCameraDirector, CameraDecision, camera_director_for_task
from .inspection_assets import HUSKY_COMMIT, UNITREE_COMMIT, ensure_inspection_assets
from .inspection_controllers import InspectionMotionController
from .inspection_executor import InspectionExecutor
from .inspection_world import InspectionWorld
from .recording import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    RecordingConfig,
    RecordingMetadata,
    SimulationVideoRecorder,
)
from .runner import (
    _execute,
    _ffmpeg_version,
    _new_output_directory,
    _sha256_file,
    _write_recording_manifest,
)

ROBOTS = {"b2_base", "z1_thermal_arm", "husky_base", "husky_franka", "static_franka"}


def run_inspection_cli(args: Namespace, *, scenario) -> int:
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be greater than zero.")
    menagerie = ensure_assets(args.assets_dir, progress=print)
    inspection_assets = ensure_inspection_assets(progress=print)
    if args.setup_only:
        print(f"Pinned Menagerie commit: {MENAGERIE_COMMIT}")
        print(f"Pinned Unitree B2 commit: {UNITREE_COMMIT}")
        print(f"Pinned Clearpath Husky commit: {HUSKY_COMMIT}")
        return 0

    bt_path = Path(args.bt) if args.bt else Path(args.scenario).with_suffix(".bt.json")
    if not bt_path.is_file():
        raise ValueError(f"BT file does not exist: {bt_path}. Generate it first or pass --bt.")
    plan = load_plan_file(bt_path)
    validation = validate_plan(plan, scenario, suggest_producers=True)
    if not validation.valid:
        first = "; ".join(error.message for error in validation.errors[:4])
        raise ValueError(f"The BT failed static validation and cannot enter MuJoCo: {first}")
    _check_scope(plan)

    print("Building the five-agent MuJoCo plant: official B2/Z1, Husky A200 geometry, and two Pandas.")
    world = InspectionWorld.build(menagerie, inspection_assets, task_id=scenario.task_id)
    motion = InspectionMotionController(world)
    executor = InspectionExecutor(world, scenario, plan, motion, progress=print)

    output_dir: Path | None = None
    recording_metadata: RecordingMetadata | None = None
    if args.record_video:
        output_dir = _new_output_directory(Path(args.output), scenario.task_id)
        shutil.copy2(Path(args.scenario).resolve(), output_dir / "scenario.json")
        shutil.copy2(bt_path.resolve(), output_dir / "behavior_tree.json")
        director = camera_director_for_task(scenario.task_id) if args.video_camera is None else None
        fallback = director.program.fallback if director else args.video_camera
        assert fallback is not None
        recorder = SimulationVideoRecorder(
            world.model,
            RecordingConfig(
                path=output_dir / "simulation.mp4",
                fps=DEFAULT_VIDEO_FPS if args.video_fps is None else args.video_fps,
                width=DEFAULT_VIDEO_WIDTH if args.video_width is None else args.video_width,
                height=DEFAULT_VIDEO_HEIGHT if args.video_height is None else args.video_height,
                camera=fallback,
                camera_mode="action_directed" if director else "fixed",
                camera_sequence=director.cameras if director else (),
            ),
        )
        with recorder:
            mujoco.mj_forward(world.model, world.data)
            decision = _decision(executor, recorder, director)
            recorder.capture_initial(world.data, camera=decision.camera, reason=decision.reason)
            report = _execute(
                executor,
                {},
                cast(Any, motion),
                args=args,
                recorder=recorder,
                camera_director=director,
            )
            decision = _decision(executor, recorder, director)
            recorder.finish(world.data, camera=decision.camera, reason=decision.reason)
        recording_metadata = recorder.metadata
    else:
        report = _execute(executor, {}, cast(Any, motion), args=args)

    if output_dir is None:
        output_dir = _new_output_directory(Path(args.output), scenario.task_id)
    recording_payload: dict[str, Any] | None = None
    if recording_metadata:
        recording_payload = recording_metadata.to_dict()
        recording_payload["sha256"] = _sha256_file(output_dir / recording_metadata.file)
        recording_payload["ffmpeg_version"] = _ffmpeg_version()
    report_path = output_dir / "physical_execution_report.json"
    save_json(
        report_path,
        {
            "scenario": "scenario.json" if recording_metadata else str(Path(args.scenario).resolve()),
            "behavior_tree": "behavior_tree.json" if recording_metadata else str(bt_path.resolve()),
            "asset_sources": {
                "mujoco_menagerie": MENAGERIE_COMMIT,
                "unitree_mujoco": UNITREE_COMMIT,
                "clearpath_husky": HUSKY_COMMIT,
            },
            **({"recording": recording_payload} if recording_payload else {}),
            **report.to_dict(),
        },
    )
    manifest_path = None
    if recording_payload:
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
    if recording_metadata:
        print(f"Video: {(output_dir / recording_metadata.file).resolve()}")
        print(f"Recording manifest: {manifest_path.resolve() if manifest_path else ''}")
    print(f"Result: {'SUCCESS' if report.success else 'FAILURE'} - {report.reason}")
    return 0 if report.success else 1


def _decision(executor, recorder, director: ActionCameraDirector | None) -> CameraDecision:
    if director is None:
        return CameraDecision(recorder.config.camera, "fixed_camera_override")
    return director.update(executor.events)


def _check_scope(plan) -> None:
    if set(plan.behavior_trees) != ROBOTS:
        raise ValueError(f"Inspection BT must contain exactly these five trees: {', '.join(sorted(ROBOTS))}.")
    for robot, root in plan.behavior_trees.items():
        for node in iter_nodes(root):
            if node.children and node.type not in {"Sequence", "Fallback"}:
                raise ValueError(f"Inspection physical adapter does not support {robot}/{node.type}.")
