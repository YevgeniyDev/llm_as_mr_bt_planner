"""End-to-end MuJoCo control/adaptation recovery experiment."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

from ..artifacts import load_plan_file
from ..config import PROJECT_ROOT, save_json, save_text
from ..domain import Scenario, load_scenario
from ..llm.base import redact_secrets
from ..recovery import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_RECOVERY_MODEL,
    OpenAIResponsesRecoveryClient,
    OracleRecoveryClient,
    RecoveryPlanningResult,
    plan_diff,
    plan_recovery,
)
from ..validation import validate_plan
from .assets import MENAGERIE_COMMIT, ensure_assets
from .camera_director import camera_director_for_task
from .controllers import ArmController, ContactGaitController, build_arm_controllers
from .executor import ExecutionReport, PhysicalExecutor
from .faults import DeterministicFaultInjector, FaultSpec, load_fault_spec
from .live_viewer import LiveViewerSession
from .recording import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    RecordingConfig,
    RecordingMetadata,
    SimulationVideoRecorder,
)
from .runner import _check_adapter_scope
from .world import RECOVERY_TASK_ID, CourierWorld

DEFAULT_RECOVERY_SCENARIO = PROJECT_ROOT / "examples" / "three_robot_component_installation.json"
DEFAULT_RECOVERY_BT = PROJECT_ROOT / "examples" / "three_robot_component_installation.bt.json"
DEFAULT_RECOVERY_FAULT = (
    PROJECT_ROOT / "examples" / "three_robot_component_installation.fault.json"
)
DEFAULT_RECOVERY_ORACLE = (
    PROJECT_ROOT / "examples" / "three_robot_component_installation.expected_recovery.bt.json"
)


def run_recovery_cli(args: Namespace) -> int:
    scenario_path = Path(args.scenario).resolve()
    nominal_path = Path(args.bt).resolve()
    fault_path = Path(args.fault).resolve()
    scenario = load_scenario(scenario_path, strict=True)
    if scenario.task_id != RECOVERY_TASK_ID:
        raise ValueError(
            f"Recovery experiment requires task_id {RECOVERY_TASK_ID!r}, got {scenario.task_id!r}."
        )
    nominal_plan = load_plan_file(nominal_path)
    validation = validate_plan(nominal_plan, scenario, suggest_producers=True)
    if not validation.valid:
        first = "; ".join(error.message for error in validation.errors[:4])
        raise ValueError(f"Nominal BT failed static validation: {first}")
    _check_adapter_scope(scenario, nominal_plan)
    fault_spec = load_fault_spec(fault_path)
    _validate_fault_scope(fault_spec, scenario)
    _validate_args(args)

    assets = ensure_assets(args.assets_dir, progress=print)
    output_dir = _new_experiment_directory(Path(args.output))
    shutil.copy2(scenario_path, output_dir / "scenario.json")
    shutil.copy2(nominal_path, output_dir / "nominal_behavior_tree.json")
    shutil.copy2(fault_path, output_dir / "fault_specification.json")

    print("Running fault-only control trial with the nominal Behavior Tree.")
    control = _run_fault_trial(
        scenario,
        nominal_plan,
        fault_spec,
        assets=assets,
        video_path=None if args.no_video else output_dir / "failure_only.mp4",
        args=args,
        continue_after_failure=None,
        progress=print,
    )
    save_json(output_dir / "failure_only_report.json", control)
    save_json(output_dir / "control_failure_snapshot.json", control["failure_snapshot"])

    client = (
        OracleRecoveryClient(args.oracle_bt)
        if args.planner == "oracle"
        else OpenAIResponsesRecoveryClient(
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
    )
    planning_holder: dict[str, RecoveryPlanningResult] = {}

    def adapt(
        executor: PhysicalExecutor,
        measured_facts: tuple[str, ...],
        observation: dict[str, Any],
        invariant_before: dict[str, Any],
    ) -> tuple[PhysicalExecutor, dict[str, Any]]:
        save_json(
            output_dir / "adaptive_failure_snapshot.json",
            _world_snapshot(executor, measured_facts, observation),
        )
        print(
            f"Requesting a complete continuation BT from {client.provider}/{client.model}; "
            "MuJoCo remains at the post-failure state."
        )
        planning_started = time.perf_counter()
        planning = plan_recovery(
            client,
            scenario,
            measured_initial_state=measured_facts,
            failure_observation=observation,
            nominal_plan=nominal_plan,
            max_corrections=args.max_corrections,
            max_ticks=args.max_ticks,
            progress=lambda message, fraction: print(
                f"[recovery {round(fraction * 100):3d}%] {message}"
            ),
        )
        planning_wall_seconds = time.perf_counter() - planning_started
        planning_holder["result"] = planning
        save_json(output_dir / "adapted_behavior_tree.json", planning.plan.to_dict())
        save_json(output_dir / "llm_recovery_attempts.json", list(planning.attempts))
        save_json(
            output_dir / "adapted_validation_report.json",
            {
                "valid": planning.validation.valid,
                "errors": planning.validation.to_dicts(),
                "contract_simulation": planning.simulation.to_dict(),
            },
        )
        save_text(output_dir / "behavior_tree_adaptation.diff", plan_diff(nominal_plan, planning.plan))

        recovery_executor = PhysicalExecutor(
            executor.world,
            planning.runtime_scenario,
            planning.plan,
            executor.arms,
            executor.gait,
            progress=print,
        )
        invariant_after = _continuity_invariant(executor.world)
        unchanged = invariant_before == invariant_after
        if not unchanged:
            raise RuntimeError(
                "Constructing the adapted executor changed MuJoCo state; same-simulation invariant failed."
            )
        return recovery_executor, {
            "same_model_and_data": True,
            "no_reset_during_adaptation": True,
            "state_hash_unchanged_while_replanning": unchanged,
            "before": invariant_before,
            "after": invariant_after,
            "replanning_wall_seconds": round(planning_wall_seconds, 4),
            "recovery_provider": planning.provider,
        }

    print("Running adaptive trial with the identical fault trigger and no post-failure reset.")
    try:
        adaptive = _run_fault_trial(
            scenario,
            nominal_plan,
            fault_spec,
            assets=assets,
            video_path=None if args.no_video else output_dir / "adaptive_recovery.mp4",
            args=args,
            continue_after_failure=adapt,
            progress=print,
        )
    except Exception as error:
        save_json(
            output_dir / "experiment_error.json",
            {
                "stage": "adaptive_trial_or_replanning",
                "error": redact_secrets(str(error)),
                "control_completed": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
    save_json(output_dir / "adaptive_recovery_report.json", adaptive)
    save_json(output_dir / "adaptive_failure_snapshot.json", adaptive["failure_snapshot"])
    if "result" not in planning_holder:
        raise RuntimeError("Adaptive trial ended without invoking the recovery planner.")
    planning = planning_holder["result"]

    comparison: dict[str, Any] | None = None
    if not args.no_video:
        comparison_path = output_dir / "comparison_side_by_side.mp4"
        _make_side_by_side(
            output_dir / "failure_only.mp4",
            output_dir / "adaptive_recovery.mp4",
            comparison_path,
            right_label=(
                "LLM-ADAPTED RECOVERY"
                if planning.provider == "openai"
                else "ORACLE DRY-RUN RECOVERY"
            ),
        )
        comparison = {
            "file": comparison_path.name,
            "sha256": _sha256_file(comparison_path),
            "layout": "fault-only control on left; adaptive recovery on right",
            "alignment": "same deterministic scene, settling, nominal BT, and fault trigger",
            "shorter_side": "last frame frozen until adaptive trial ends",
        }

    accepted_provenance = planning.attempts[-1]["provenance"]
    returned_model = accepted_provenance.get("model_returned")
    requested_model_returned = bool(
        isinstance(returned_model, str)
        and (
            returned_model == planning.model
            or returned_model.startswith(f"{planning.model}-")
        )
    )
    real_llm_evidence = planning.provider == "openai" and requested_model_returned
    integration_success = bool(
        not control["physical_execution"]["success"]
        and control["failure_observation"]["classification"] != "fault_not_physically_established"
        and adaptive["physical_execution"]["success"]
        and adaptive["continuity"]["no_reset_during_adaptation"]
        and adaptive["continuity"]["state_hash_unchanged_while_replanning"]
        and adaptive["continuity"]["no_reset_through_completion"]
    )
    success = integration_success and real_llm_evidence
    experiment = {
        "experiment_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": scenario.task_id,
        "success": success,
        "scientific_claim": (
            "A real LLM adapted the BT after a measured unexpected object failure, and the "
            "validated continuation completed in the same MuJoCo state without reset."
            if real_llm_evidence
            else (
                "Offline integration dry run only; deterministic oracle is not LLM evidence."
                if planning.provider == "deterministic_oracle"
                else "Requested OpenAI model identity was not verified; no research claim is made."
            )
        ),
        "planner": {
            "provider": planning.provider,
            "model": planning.model,
            "reasoning_effort": planning.reasoning_effort,
            "attempt_count": len(planning.attempts),
            "model_returned": returned_model,
            "requested_model_verified": requested_model_returned,
            "real_llm_evidence": real_llm_evidence,
        },
        "control": {
            "failed_as_expected": not control["physical_execution"]["success"],
            "failure_classification": control["failure_observation"]["classification"],
        },
        "adaptive": {
            "recovered": adaptive["physical_execution"]["success"],
            "continuity": adaptive["continuity"],
        },
        "comparison_video": comparison,
        "software": {
            "python": sys.version,
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "files": _artifact_hashes(output_dir),
    }
    save_json(output_dir / "experiment_manifest.json", experiment)
    print(f"Experiment artifacts: {output_dir}")
    if not args.no_video:
        print(f"Failure-only video: {output_dir / 'failure_only.mp4'}")
        print(f"Adaptive video: {output_dir / 'adaptive_recovery.mp4'}")
        print(f"Comparison video: {output_dir / 'comparison_side_by_side.mp4'}")
    if planning.provider != "openai":
        print(
            "Result: "
            f"{'DRY RUN SUCCESS' if integration_success else 'DRY RUN FAILURE'} — "
            "deterministic oracle used; no LLM claim is made."
        )
        return 0 if integration_success else 1
    print(f"Result: {'SUCCESS' if success else 'FAILURE'}")
    return 0 if success else 1


def _run_fault_trial(
    scenario: Scenario,
    nominal_plan,
    fault_spec: FaultSpec,
    *,
    assets: Path,
    video_path: Path | None,
    args: Namespace,
    continue_after_failure,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    world = CourierWorld.build(assets, task_id=scenario.task_id)
    arms = build_arm_controllers(world)
    gait = ContactGaitController(world)
    executor = PhysicalExecutor(world, scenario, nominal_plan, arms, gait, progress=progress)
    injector = DeterministicFaultInjector(fault_spec)
    recorder: SimulationVideoRecorder | None = None
    recording: RecordingMetadata | None = None
    config = None
    if video_path is not None:
        director = camera_director_for_task(scenario.task_id)
        config = RecordingConfig(
            path=video_path,
            fps=args.video_fps,
            width=args.video_width,
            height=args.video_height,
            camera=director.program.fallback,
            camera_mode="action_directed",
            camera_sequence=tuple(dict.fromkeys((*director.cameras, "recovery_floor"))),
        )
        recorder = SimulationVideoRecorder(world.model, config)

    live_viewer_enabled = not bool(getattr(args, "headless", True))
    realtime_factor = float(getattr(args, "realtime_factor", 1.0))

    def execute(
        open_recorder: SimulationVideoRecorder | None,
        live_viewer: LiveViewerSession | None,
    ) -> tuple[ExecutionReport, dict[str, Any]]:
        mujoco.mj_forward(world.model, world.data)
        if open_recorder is not None:
            open_recorder.capture_initial(world.data, camera="overview", reason="initial_overview")
        _settle(
            world,
            arms,
            gait,
            open_recorder,
            live_viewer,
            camera="overview",
            seconds=1.0,
        )
        start = float(world.data.time)
        while not executor.failed and not executor.complete:
            if float(world.data.time) - start > args.max_seconds:
                raise RuntimeError("Nominal fault trial exceeded its simulation-time limit.")
            executor.step(float(world.model.opt.timestep))
            injector.update(executor)
            mujoco.mj_step(world.model, world.data)
            _capture(
                open_recorder,
                live_viewer,
                world,
                "recovery_floor" if injector.triggered else "recovery_source",
                "fault_active" if injector.triggered else "nominal_handoff",
                phase=("FAILURE OCCURRING" if injector.triggered else "NOMINAL BT EXECUTION"),
            )
        injector.clear(executor)
        if not injector.triggered:
            raise RuntimeError("Nominal BT ended before the configured fault trigger was reached.")
        if not executor.failed:
            raise RuntimeError("Injected fault did not cause the nominal BT to fail.")
        _safe_stop(executor, open_recorder, live_viewer)
        observation = injector.observation(executor)
        if observation["classification"] == "fault_not_physically_established":
            raise RuntimeError(
                "Fault injection did not establish the required measured displacement."
            )
        if observation["object_usable"]:
            executor._add_signal(f"usable({observation['object']})")
        else:
            executor._discard_signal(f"usable({observation['object']})")
        measured_facts = _measured_recovery_facts(executor, observation)
        snapshot = _world_snapshot(executor, measured_facts, observation)

        if continue_after_failure is None:
            _settle(
                world,
                arms,
                gait,
                open_recorder,
                live_viewer,
                camera="recovery_floor",
                seconds=2.0,
            )
            return executor.make_report(
                "Nominal BT stopped safely after the injected primary-part failure."
            ), {
                "failure_observation": observation,
                "failure_snapshot": snapshot,
                "continuity": None,
                "combined_events": executor.events,
            }

        invariant_before = _continuity_invariant(world)

        def adapt() -> tuple[PhysicalExecutor, dict[str, Any]]:
            return continue_after_failure(
                executor,
                measured_facts,
                observation,
                invariant_before,
            )

        if live_viewer is None:
            recovery_executor, continuity = adapt()
        else:
            recovery_executor, continuity = live_viewer.run_while_frozen(
                adapt,
                camera="recovery_floor",
                title="FAILURE DETECTED - BT ADAPTATION IN PROGRESS",
                message=(
                    f"{observation['classification']}: {observation['object']} at "
                    f"{observation['recovery_location']}\n"
                    "The LLM is generating and validating a retrieval tree"
                ),
            )
        if open_recorder is not None:
            replanning_wall_seconds = float(continuity.get("replanning_wall_seconds", 0.0))
            provider = str(continuity.get("recovery_provider", "recovery_planner"))
            open_recorder.append_status_overlay(
                world.data,
                title="FAILURE DETECTED",
                message=(
                    "LLM IS ADAPTING THE BEHAVIOR TREE..."
                    if provider == "openai"
                    else "DRY-RUN RECOVERY PLANNER IS ADAPTING THE TREE..."
                ),
                detail=(
                    f"{observation['classification']}: {observation['object']} is intact at "
                    f"{observation['recovery_location']}. The robots are safely stopped while "
                    "a validated retrieval continuation is prepared."
                ),
                duration_seconds=min(max(replanning_wall_seconds, 3.0), 15.0),
                wall_seconds=replanning_wall_seconds,
                camera="recovery_floor",
            )
        recovery_director = camera_director_for_task(scenario.task_id)
        recovery_start = float(world.data.time)
        while not recovery_executor.complete and not recovery_executor.failed:
            if float(world.data.time) - recovery_start > args.max_seconds:
                _complete_continuity_evidence(continuity, world)
                return recovery_executor.make_report(
                    "Adapted continuation exceeded its simulation-time limit."
                ), {
                    "failure_observation": observation,
                    "failure_snapshot": snapshot,
                    "continuity": continuity,
                    "combined_events": [*executor.events, *recovery_executor.events],
                }
            recovery_executor.step(float(world.model.opt.timestep))
            mujoco.mj_step(world.model, world.data)
            decision = recovery_director.update(recovery_executor.events)
            _capture(
                open_recorder,
                live_viewer,
                world,
                decision.camera,
                decision.reason,
                phase="ADAPTED BT EXECUTION",
            )
        _complete_continuity_evidence(continuity, world)
        return recovery_executor.make_report(), {
            "failure_observation": observation,
            "failure_snapshot": snapshot,
            "continuity": continuity,
            "combined_events": [*executor.events, *recovery_executor.events],
        }

    def run_with_viewer(open_recorder: SimulationVideoRecorder | None) -> tuple[ExecutionReport, dict[str, Any]]:
        if not live_viewer_enabled:
            return execute(open_recorder, None)
        progress(
            "Opening the live MuJoCo viewer with action-directed camera cuts; "
            "closing it early stops the demo."
        )
        with LiveViewerSession(
            world.model,
            world.data,
            realtime_factor=realtime_factor,
        ) as live_viewer:
            report, extra = execute(open_recorder, live_viewer)
            final_camera = "recovery_destination" if report.success else "recovery_floor"
            live_viewer.set_camera(final_camera)
            live_viewer.set_status(
                "ADAPTIVE DEMO COMPLETE" if report.success else "ADAPTIVE DEMO STOPPED",
                "Behavior Tree goal reached" if report.success else report.reason,
            )
            live_viewer.refresh()
            live_viewer.hold_terminal_state()
            return report, extra

    if recorder is None:
        report, extra = run_with_viewer(None)
    else:
        with recorder:
            report, extra = run_with_viewer(recorder)
            final_camera = "recovery_destination" if report.success else "recovery_floor"
            recorder.finish(world.data, camera=final_camera, reason="terminal_state")
        recording = recorder.metadata

    result: dict[str, Any] = {
        "physical_execution": report.to_dict(),
        **extra,
        "recording": (
            {
                **recording.to_dict(),
                "sha256": _sha256_file(video_path),
            }
            if recording is not None and video_path is not None
            else None
        ),
        "live_viewer": {
            "enabled": live_viewer_enabled,
            "camera_mode": "action_directed" if live_viewer_enabled else None,
            "realtime_factor": realtime_factor if live_viewer_enabled else None,
        },
    }
    return result


def _settle(
    world: CourierWorld,
    arms: dict[str, ArmController],
    gait: ContactGaitController,
    recorder: SimulationVideoRecorder | None,
    live_viewer: LiveViewerSession | None,
    *,
    camera: str,
    seconds: float,
) -> None:
    steps = round(seconds / float(world.model.opt.timestep))
    for _ in range(steps):
        gait.step()
        for arm in arms.values():
            arm.hold()
        mujoco.mj_step(world.model, world.data)
        _capture(
            recorder,
            live_viewer,
            world,
            camera,
            "settling",
            phase="PREPARING SIMULATION",
        )


def _safe_stop(
    executor: PhysicalExecutor,
    recorder: SimulationVideoRecorder | None,
    live_viewer: LiveViewerSession | None,
) -> None:
    world = executor.world
    for key in world.grip_equalities:
        if key.startswith(("franka_a:", "franka_b:", "unitree_go2_z1:")):
            world.deactivate_weld(key)
    executor.resources.clear()
    start = float(world.data.time)
    all_home = False
    while float(world.data.time) - start <= 12.0:
        all_home = True
        for arm in executor.arms.values():
            arm.set_gripper(closed=False)
            all_home = arm.move_home(float(world.model.opt.timestep)) and all_home
        executor.gait.step()
        mujoco.mj_step(world.model, world.data)
        _capture(
            recorder,
            live_viewer,
            world,
            "recovery_floor",
            "safe_stop_after_failure",
            phase="FAILURE DETECTED - SAFE STOP",
        )
        primary_on_floor = bool(
            world.object_position("primary_part")[2] <= 0.08
        )
        if all_home and primary_on_floor and float(world.data.time) - start > 0.8:
            break
    if not all_home:
        raise RuntimeError("Robots did not reach their safe home postures after the fault.")


def _measured_recovery_facts(
    executor: PhysicalExecutor,
    observation: dict[str, Any],
) -> tuple[str, ...]:
    candidates = [
        "system_ready()",
        "robot_ready(franka_a)",
        "robot_ready(unitree_go2_z1)",
        "robot_ready(franka_b)",
        "usable(primary_part)",
        "at(primary_part,source_floor)",
        "gripper_empty(franka_a)",
        "gripper_empty(unitree_go2_z1)",
        "gripper_empty(franka_b)",
        "arm_stowed(unitree_go2_z1)",
        "base_stationary(unitree_go2_z1)",
        "docked(unitree_go2_z1,source_dock)",
        "docked(unitree_go2_z1,destination_dock)",
    ]
    facts = tuple(literal for literal in candidates if executor.observe_literal(literal))
    required = {
        "usable(primary_part)",
        "at(primary_part,source_floor)",
        "gripper_empty(franka_a)",
        "gripper_empty(unitree_go2_z1)",
        "gripper_empty(franka_b)",
        "arm_stowed(unitree_go2_z1)",
        "base_stationary(unitree_go2_z1)",
        "docked(unitree_go2_z1,source_dock)",
    }
    missing = sorted(required - set(facts))
    if missing:
        raise RuntimeError(
            "Safe same-object recovery snapshot is missing measured facts: "
            f"{', '.join(missing)}. Observation: "
            f"usable={observation.get('object_usable')}, "
            f"location={observation.get('recovery_location')}."
        )
    return facts


def _world_snapshot(
    executor: PhysicalExecutor,
    facts: tuple[str, ...],
    observation: dict[str, Any],
) -> dict[str, Any]:
    world = executor.world
    return {
        "simulation_time_seconds": round(float(world.data.time), 6),
        "reset_count": world.reset_count,
        "state_sha256": _state_hash(world),
        "measured_initial_state_for_continuation": list(facts),
        "diagnosed_failure": observation,
        "object_positions_m": {
            name: world.object_position(name).round(6).tolist()
            for name in world.object_body_ids
        },
        "go2_base_position_m": world.base_position.round(6).tolist(),
        "go2_base_velocity": world.base_velocity.round(6).tolist(),
        "active_equalities": {
            key: world.equality_active(key) for key in world.grip_equalities
        },
        "resources_after_safe_stop": dict(executor.resources),
    }


def _continuity_invariant(world: CourierWorld) -> dict[str, Any]:
    return {
        "model_identity": id(world.model),
        "data_identity": id(world.data),
        "reset_count": world.reset_count,
        "simulation_time_seconds": float(world.data.time),
        "state_sha256": _state_hash(world),
    }


def _complete_continuity_evidence(
    continuity: dict[str, Any],
    world: CourierWorld,
) -> None:
    final = _continuity_invariant(world)
    continuity["after_continuation"] = final
    continuity["no_reset_through_completion"] = (
        final["model_identity"] == continuity["before"]["model_identity"]
        and final["data_identity"] == continuity["before"]["data_identity"]
        and final["reset_count"] == continuity["before"]["reset_count"]
    )


def _state_hash(world: CourierWorld) -> str:
    digest = hashlib.sha256()
    for array in (
        world.data.qpos,
        world.data.qvel,
        world.data.ctrl,
        world.data.eq_active,
        world.data.xfrc_applied,
    ):
        digest.update(np.asarray(array).tobytes())
    digest.update(np.asarray([world.data.time], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _capture(
    recorder: SimulationVideoRecorder | None,
    live_viewer: LiveViewerSession | None,
    world: CourierWorld,
    camera: str,
    reason: str,
    *,
    phase: str = "MUJOCO EXECUTION",
) -> None:
    if recorder is not None:
        recorder.capture_after_step(world.data, camera=camera, reason=reason)
    if live_viewer is not None:
        live_viewer.after_step(camera=camera, phase=phase, detail=reason)


def _make_side_by_side(
    left: Path,
    right: Path,
    output: Path,
    *,
    right_label: str,
) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError("imageio-ffmpeg is required to create the comparison video.") from error
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    filter_graph = (
        "[0:v]tpad=stop_mode=clone:stop_duration=600,setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=ih/12:color=black@0.65:t=fill,"
        "drawtext=text='FAULT-ONLY CONTROL':fontcolor=white:fontsize=h/30:"
        "x=(w-text_w)/2:y=h/48[left];"
        "[1:v]setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=ih/12:color=black@0.65:t=fill,"
        f"drawtext=text='{right_label}':fontcolor=white:fontsize=h/30:"
        "x=(w-text_w)/2:y=h/48[right];"
        "[left][right]hstack=inputs=2:shortest=1[v]"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg comparison encode failed: {completed.stderr.strip()}")


def _validate_fault_scope(spec: FaultSpec, scenario: Scenario) -> None:
    if spec.trigger.robot not in scenario.robot_ids:
        raise ValueError(f"Fault trigger robot {spec.trigger.robot!r} is not declared.")
    if spec.trigger.object not in scenario.constants:
        raise ValueError(f"Fault object {spec.trigger.object!r} is not declared.")
    capability = scenario.capability(spec.trigger.robot, spec.trigger.action)
    if capability is None:
        raise ValueError("Fault trigger action is not a capability of the trigger robot.")
    if spec.trigger.event is not None and spec.trigger.event != "action_success":
        raise ValueError("Handoff recovery faults require trigger event 'action_success'.")
    if spec.trigger.location is not None and spec.trigger.location not in scenario.constants:
        raise ValueError(f"Fault trigger location {spec.trigger.location!r} is not declared.")
    if spec.trigger.before_robot is not None:
        if spec.trigger.before_robot not in scenario.robot_ids:
            raise ValueError(f"Fault before_robot {spec.trigger.before_robot!r} is not declared.")
        if (
            spec.trigger.before_action is None
            or scenario.capability(spec.trigger.before_robot, spec.trigger.before_action) is None
        ):
            raise ValueError("Fault before_action is not a capability of before_robot.")
    if not spec.recoverable:
        raise ValueError("Recovery experiment requires recoverable=true.")


def _validate_args(args: Namespace) -> None:
    if args.max_seconds <= 0 or args.max_ticks <= 0:
        raise ValueError("--max-seconds and --max-ticks must be positive.")
    if args.max_corrections < 0:
        raise ValueError("--max-corrections cannot be negative.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    for option, value in (("--video-width", args.video_width), ("--video-height", args.video_height)):
        if value <= 0 or value % 2:
            raise ValueError(f"{option} must be a positive even integer.")
    if args.planner == "oracle" and not Path(args.oracle_bt).is_file():
        raise ValueError("--oracle-bt must exist when --planner oracle is selected.")


def _new_experiment_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = root.resolve() / f"fallen-part-recovery-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.name: _sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "experiment_manifest.json"
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_recovery_args() -> dict[str, Any]:
    """Shared defaults kept explicit for CLI tests and programmatic use."""
    return {
        "scenario": str(DEFAULT_RECOVERY_SCENARIO),
        "bt": str(DEFAULT_RECOVERY_BT),
        "fault": str(DEFAULT_RECOVERY_FAULT),
        "oracle_bt": str(DEFAULT_RECOVERY_ORACLE),
        "model": DEFAULT_RECOVERY_MODEL,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "video_fps": DEFAULT_VIDEO_FPS,
        "video_width": DEFAULT_VIDEO_WIDTH,
        "video_height": DEFAULT_VIDEO_HEIGHT,
    }
