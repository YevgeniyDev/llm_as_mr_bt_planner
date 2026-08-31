"""Fault-blind LLM generation and same-state five-agent dropped-tool recovery."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from argparse import Namespace
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from ..config import PROJECT_ROOT, save_json, save_text
from ..inspection_recovery import (
    INSPECTION_ROBOTS,
    InspectionRecoveryPlanningResult,
    plan_inspection_tool_recovery,
)
from ..prompts import SYSTEM_PROMPT, build_prompt
from ..recovery import OpenAIResponsesRecoveryClient, plan_diff, recovery_plan_json_schema
from ..service import PlannerService
from .adaptive_demo_runner import DemoLogger, _heartbeat
from .assets import MENAGERIE_COMMIT, ensure_assets
from .camera_director import ActionCameraDirector, camera_director_for_task
from .inspection_assets import HUSKY_COMMIT, UNITREE_COMMIT, ensure_inspection_assets
from .inspection_controllers import InspectionMotionController
from .inspection_executor import InspectionExecutor
from .inspection_world import B2_DOCK_X, HUSKY_DOCK_X, TASK_ID, InspectionWorld
from .live_viewer import LiveViewerSession
from .recording import RecordingConfig, SimulationVideoRecorder

DEFAULT_SCENARIO = PROJECT_ROOT / "examples" / "five_agent_solar_pipe_inspection.json"
DEFAULT_FAULT = PROJECT_ROOT / "examples" / "five_agent_solar_pipe_inspection_tool_drop.fault.json"


def run_inspection_adaptive_cli(args: Namespace) -> int:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for the real five-agent adaptive demo.")
    if args.max_seconds <= 0 or args.realtime_factor <= 0:
        raise ValueError("--max-seconds and --realtime-factor must be greater than zero.")
    scenario_path = Path(args.scenario).resolve()
    fault_path = Path(args.fault).resolve()
    if not scenario_path.is_file() or not fault_path.is_file():
        raise FileNotFoundError("The five-agent scenario and sealed fault specification must exist.")

    output_dir = _new_output_directory(Path(args.output))
    logger = DemoLogger(output_dir / "inspection_adaptive.log")
    started = time.perf_counter()
    logger.event("START", f"Five-agent adaptive directory: {output_dir}")
    logger.event("BOUNDARY", "Fault configuration is sealed and has not been read.")
    try:
        success, manifest = _execute(
            args,
            api_key=api_key,
            scenario_path=scenario_path,
            fault_path=fault_path,
            output_dir=output_dir,
            logger=logger,
            started=started,
        )
    except Exception as error:
        logger.event("ERROR", str(error))
        save_json(
            output_dir / "inspection_adaptive_error.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": str(error),
                "events": logger.events,
            },
        )
        raise
    logger.event("DONE", f"Result: {'SUCCESS' if success else 'FAILURE'}")
    logger.event("DONE", f"Artifacts: {output_dir}")
    save_json(output_dir / "inspection_adaptive_events.json", logger.events)
    manifest["files"] = _artifact_hashes(output_dir, excluded={"inspection_adaptive_manifest.json"})
    save_json(output_dir / "inspection_adaptive_manifest.json", manifest)
    return 0 if success else 1


def _execute(
    args: Namespace,
    *,
    api_key: str,
    scenario_path: Path,
    fault_path: Path,
    output_dir: Path,
    logger: DemoLogger,
    started: float,
) -> tuple[bool, dict[str, Any]]:
    shutil.copy2(scenario_path, output_dir / "scenario.json")
    scenario_document = PlannerService.load_json(scenario_path)
    service = PlannerService(output_dir / "nominal_generation")
    scenario = service.parse_scenario_document(scenario_document)
    if scenario.task_id != TASK_ID:
        raise ValueError(f"Five-agent adaptive demo requires task_id {TASK_ID!r}.")

    nominal_prompt = build_prompt(scenario)
    save_text(output_dir / "nominal_system_prompt.txt", SYSTEM_PROMPT)
    save_text(output_dir / "nominal_user_prompt.txt", nominal_prompt)
    nominal_started = time.perf_counter()
    logger.event("PHASE 1/5", f"Generating five fault-blind BTs with openai/{args.model}.")
    with _heartbeat(logger, "NOMINAL LLM", "Waiting for five-agent nominal BT generation", interval_seconds=args.heartbeat_seconds):
        outcome = service.generate(
            scenario_document,
            provider="openai",
            api_key=api_key,
            model=args.model,
            max_corrections=args.generation_max_corrections,
            max_ticks=args.max_ticks,
            progress=lambda message, fraction: logger.progress("NOMINAL LLM", message, fraction),
        )
    if not outcome.validation.valid or not outcome.simulation.success:
        raise RuntimeError("Nominal five-agent LLM BT was not accepted.")
    forbidden = {
        "navigate_b2_tool_search",
        "localize_fallen_tool",
        "navigate_husky_tool_recovery",
        "recover_localized_tool",
    }
    nominal_actions = sorted(
        node.name
        for root in outcome.plan.behavior_trees.values()
        for node in _iter_nodes(root)
        if node.type == "Action" and node.name
    )
    if forbidden.intersection(nominal_actions):
        raise RuntimeError("Nominal plan leaked a fault-recovery capability before fault disclosure.")
    save_json(output_dir / "nominal_behavior_tree.json", outcome.plan.to_dict())
    save_json(output_dir / "nominal_provider_responses.json", list(outcome.planner_result.provider_responses))
    nominal_completed = time.perf_counter()

    logger.event("BOUNDARY", "Nominal plan is fixed. Reading the sealed fault for the first time.")
    fault_loaded = time.perf_counter()
    fault = json.loads(fault_path.read_text(encoding="utf-8"))
    _validate_fault(fault)
    shutil.copy2(fault_path, output_dir / "fault_specification.json")
    blindness = {
        "boundary_verified": nominal_completed <= fault_loaded,
        "fault_absent_from_nominal_prompt": fault["fault_id"] not in nominal_prompt,
        "recovery_actions_absent_from_nominal_bt": not forbidden.intersection(nominal_actions),
        "nominal_completed_monotonic_s": nominal_completed,
        "fault_loaded_monotonic_s": fault_loaded,
        "nominal_prompt_sha256": hashlib.sha256(nominal_prompt.encode("utf-8")).hexdigest(),
    }
    save_json(output_dir / "fault_blindness_evidence.json", blindness)
    if not all(
        blindness[key]
        for key in (
            "boundary_verified",
            "fault_absent_from_nominal_prompt",
            "recovery_actions_absent_from_nominal_bt",
        )
    ):
        raise RuntimeError("Fault-blind generation boundary was not verified.")

    logger.event("PHASE 2/5", "Building the five-agent MuJoCo plant and dynamic dropped-tool proxy.")
    menagerie = ensure_assets(args.assets_dir, progress=lambda message: logger.event("ASSETS", message))
    inspection_assets = ensure_inspection_assets(progress=lambda message: logger.event("ASSETS", message))
    world = InspectionWorld.build(menagerie, inspection_assets, task_id=TASK_ID)
    motion = InspectionMotionController(world)
    executor = InspectionExecutor(
        world,
        scenario,
        outcome.plan,
        motion,
        progress=lambda message: logger.event("MUJOCO", message),
    )
    director = camera_director_for_task(TASK_ID)
    recovery_director: ActionCameraDirector | None = None
    planning_holder: dict[str, InspectionRecoveryPlanningResult] = {}
    recovery_wall_seconds = 0.0

    with ExitStack() as stack:
        recorder: SimulationVideoRecorder | None = None
        if not args.no_video:
            recorder = stack.enter_context(
                SimulationVideoRecorder(
                    world.model,
                    RecordingConfig(
                        path=output_dir / "inspection_adaptive.mp4",
                        fps=args.video_fps,
                        width=args.video_width,
                        height=args.video_height,
                        camera=director.program.fallback,
                        camera_mode="action_directed",
                        camera_sequence=director.cameras,
                    ),
                )
            )
            mujoco.mj_forward(world.model, world.data)
            recorder.capture_initial(world.data, camera=director.program.fallback, reason="initial_overview")
        viewer: LiveViewerSession | None = None
        if not args.headless:
            viewer = stack.enter_context(
                LiveViewerSession(
                    world.model,
                    world.data,
                    realtime_factor=args.realtime_factor,
                    initial_camera="inspection_overview",
                    initial_title="FIVE-AGENT ADAPTIVE INSPECTION",
                )
            )

        _settle(world, motion, executor, recorder, viewer, director)
        logger.event("PHASE 3/5", "Executing the exact nominal LLM BT until the post-handoff tool drop.")
        observation = _run_until_fault(
            executor,
            fault,
            args=args,
            recorder=recorder,
            viewer=viewer,
            director=director,
        )
        save_json(output_dir / "failure_snapshot.json", observation)
        measured_facts = _measured_failure_state(executor)
        invariant_before = _state_invariant(world)
        logger.event(
            "PHASE 4/5",
            "Tool loss detected; physics is frozen while the LLM creates a five-agent continuation.",
        )

        client = OpenAIResponsesRecoveryClient(
            model=args.model,
            api_key=api_key,
            reasoning_effort=args.reasoning_effort,
            response_schema=recovery_plan_json_schema(
                mission_id=scenario.task_id,
                robots=INSPECTION_ROBOTS,
            ),
            structured_output_name="five_agent_tool_recovery_bt",
        )

        def adapt() -> InspectionRecoveryPlanningResult:
            with _heartbeat(logger, "RECOVERY LLM", "B2/Husky failure handling in progress", interval_seconds=args.heartbeat_seconds):
                return plan_inspection_tool_recovery(
                    client,
                    scenario,
                    measured_initial_state=measured_facts,
                    failure_observation=observation,
                    nominal_plan=outcome.plan,
                    max_corrections=args.recovery_max_corrections,
                    max_ticks=args.max_ticks,
                    progress=lambda message, fraction: logger.progress("RECOVERY LLM", message, fraction),
                )

        recovery_started = time.perf_counter()
        planning = (
            viewer.run_while_frozen(
                adapt,
                camera="inspection_floor_recovery",
                title="DROPPED TOOL DETECTED",
                message="LLM is assigning B2 search and Husky recovery",
            )
            if viewer is not None
            else adapt()
        )
        recovery_wall_seconds = time.perf_counter() - recovery_started
        planning_holder["result"] = planning
        invariant_after = _state_invariant(world)
        if invariant_before != invariant_after:
            raise RuntimeError("MuJoCo state changed while the LLM was adapting the BT.")
        if recorder is not None:
            recorder.append_status_overlay(
                world.data,
                title="DROPPED TOOL DETECTED",
                message="LLM recovery planning completed",
                detail="B2/Z1 will localize the tool; Husky and its Panda will recover it.",
                duration_seconds=4.0,
                wall_seconds=recovery_wall_seconds,
                camera="inspection_floor_recovery",
            )

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
        save_text(output_dir / "behavior_tree_adaptation.diff", plan_diff(outcome.plan, planning.plan))

        executor = InspectionExecutor(
            world,
            planning.runtime_scenario,
            planning.plan,
            motion,
            progress=lambda message: logger.event("MUJOCO", message),
        )
        recovery_director = camera_director_for_task(TASK_ID)
        logger.event("PHASE 4/5", "Adapted BT accepted; resuming the same MuJoCo model/data state.")
        report = _run_to_completion(
            executor,
            args=args,
            recorder=recorder,
            viewer=viewer,
            director=recovery_director,
        )
        if recorder is not None:
            decision = recovery_director.update(executor.events)
            recorder.finish(world.data, camera=decision.camera, reason="terminal_state")
        if viewer is not None:
            viewer.set_status("MISSION COMPLETE" if report.success else "MISSION FAILED", report.reason)
            viewer.hold_terminal_state(2.0)

    planning = planning_holder["result"]
    save_json(output_dir / "physical_execution_report.json", report.to_dict())
    logger.event("PHASE 5/5", "Checking physical completion, provenance, and same-state continuity.")
    nominal_model = _accepted_nominal_model(outcome)
    recovery_model = planning.attempts[-1]["provenance"].get("model_returned")
    success = bool(
        report.success
        and _model_matches(args.model, nominal_model)
        and _model_matches(args.model, recovery_model)
        and planning.validation.valid
        and planning.simulation.success
    )
    video = None
    if not args.no_video:
        video = recorder.metadata.to_dict() if recorder is not None else None
    return success, {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": scenario.task_id,
        "experiment": "fault-blind five-agent dropped-tool localization and recovery",
        "success": success,
        "fault_blindness": blindness,
        "nominal_planner": {
            "provider": outcome.planner_result.provider,
            "model_requested": args.model,
            "model_returned": nominal_model,
            "model_verified": _model_matches(args.model, nominal_model),
            "correction_rounds": outcome.planner_result.correction_rounds,
            "wall_seconds": round(nominal_completed - nominal_started, 4),
        },
        "recovery_planner": {
            "provider": planning.provider,
            "model_requested": planning.model,
            "model_returned": recovery_model,
            "model_verified": _model_matches(args.model, recovery_model),
            "reasoning_effort": planning.reasoning_effort,
            "attempt_count": len(planning.attempts),
            "wall_seconds": round(recovery_wall_seconds, 4),
        },
        "failure_observation": observation,
        "continuity": {
            "same_model_and_data": True,
            "no_reset_during_adaptation": True,
            "state_hash_unchanged_while_replanning": True,
        },
        "physical_execution": {"success": report.success, "reason": report.reason},
        "video": video,
        "software": {
            "python": sys.version,
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "menagerie_commit": MENAGERIE_COMMIT,
            "unitree_commit": UNITREE_COMMIT,
            "husky_commit": HUSKY_COMMIT,
        },
        "elapsed_wall_seconds": round(time.perf_counter() - started, 4),
    }


def _run_until_fault(
    executor: InspectionExecutor,
    fault: dict[str, Any],
    *,
    args: Namespace,
    recorder: SimulationVideoRecorder | None,
    viewer: LiveViewerSession | None,
    director: ActionCameraDirector,
) -> dict[str, Any]:
    world = executor.inspection_world
    dt = float(world.model.opt.timestep)
    start = float(world.data.time)
    event_index = 0
    triggered = False
    while float(world.data.time) - start <= args.max_seconds:
        executor.step(dt)
        new_events = executor.events[event_index:]
        event_index = len(executor.events)
        if any(
            event.get("robot") == fault["trigger"]["after_robot"]
            and event.get("kind") == "action_success"
            and str(event.get("message", "")).startswith(fault["trigger"]["after_action"] + "(")
            for event in new_events
        ):
            injection = fault["injection"]
            velocity = tuple(float(value) for value in injection["horizontal_velocity_m_s"])
            world.drop_handoff_tool(
                horizontal_velocity=(velocity[0], velocity[1]),
                vertical_velocity=float(injection["vertical_velocity_m_s"]),
            )
            executor.signals.discard("at(inspection_kit, handoff_tray)")
            executor.signals.discard("kit_ready(inspection_kit)")
            executor.signals.add("fallen_tool_unlocalized(inspection_kit)")
            executor.resources.clear()
            executor.motion.targets["b2_base"] = world.robot_x("b2_base")
            executor.motion.targets["husky_base"] = world.robot_x("husky_base")
            for cursor in executor.cursors.values():
                cursor.action = None
                cursor.action_node_id = None
            triggered = True
        mujoco.mj_step(world.model, world.data)
        _after_step(executor, recorder, viewer, director, "NOMINAL EXECUTION")
        if triggered:
            break
        if executor.failed or executor.complete:
            raise RuntimeError("Nominal execution ended before the sealed dropped-tool fault was triggered.")
    if not triggered:
        raise RuntimeError("Dropped-tool trigger was not reached before the simulation limit.")

    settle_seconds = float(fault["injection"]["settle_seconds"])
    deadline = float(world.data.time) + max(settle_seconds, 0.1) + 10.0
    minimum = float(world.data.time) + settle_seconds
    while float(world.data.time) < deadline:
        executor.motion.step()
        mujoco.mj_step(world.model, world.data)
        _after_step(executor, recorder, viewer, director, "FAULT ESTABLISHMENT")
        if (
            float(world.data.time) >= minimum
            and world.fallen_tool_settled()
            and world.robot_speed("b2_base") < 0.10
            and world.robot_speed("husky_base") < 0.10
        ):
            break
    if not world.fallen_tool_settled():
        raise RuntimeError("The dynamically dropped inspection tool did not settle on the floor.")
    if world.robot_speed("b2_base") >= 0.10 or world.robot_speed("husky_base") >= 0.10:
        raise RuntimeError("The mobile bases did not reach a measured safe stop after the fault.")
    position = world.fallen_tool_position()
    return {
        "fault_id": fault["fault_id"],
        "classification": "tool_dropped_and_location_unknown",
        "object": "inspection_kit",
        "object_usable": True,
        "requires_localization": True,
        "position_withheld_from_recovery_prompt_until_search": True,
        "measured_position_m_for_audit": position.round(5).tolist(),
        "settled": True,
        "sim_time_seconds": round(float(world.data.time), 4),
        "localization_team": ["b2_base", "z1_thermal_arm"],
        "recovery_team": ["husky_base", "husky_franka"],
    }


def _run_to_completion(
    executor: InspectionExecutor,
    *,
    args: Namespace,
    recorder: SimulationVideoRecorder | None,
    viewer: LiveViewerSession | None,
    director: ActionCameraDirector,
):
    world = executor.inspection_world
    dt = float(world.model.opt.timestep)
    start = float(world.data.time)
    while not executor.complete and not executor.failed:
        if float(world.data.time) - start > args.max_seconds:
            return executor.make_report(f"Recovery exceeded {args.max_seconds:.1f} simulated seconds.")
        executor.step(dt)
        mujoco.mj_step(world.model, world.data)
        _after_step(executor, recorder, viewer, director, "ADAPTED BT EXECUTION")
    return executor.make_report()


def _settle(world, motion, executor, recorder, viewer, director) -> None:
    steps = round(1.0 / float(world.model.opt.timestep))
    for _ in range(steps):
        motion.step()
        mujoco.mj_step(world.model, world.data)
        _after_step(executor, recorder, viewer, director, "SCENE SETTLING")
    if not motion.upright():
        raise RuntimeError("B2 did not remain upright during scene settling.")


def _after_step(executor, recorder, viewer, director, phase: str) -> None:
    decision = director.update(executor.events)
    if recorder is not None:
        recorder.capture_after_step(executor.inspection_world.data, camera=decision.camera, reason=decision.reason)
    if viewer is not None:
        viewer.after_step(camera=decision.camera, phase=phase, detail=decision.reason)


def _measured_failure_state(executor: InspectionExecutor) -> tuple[str, ...]:
    signals = set(executor.signals)
    signals.discard("at(inspection_kit, handoff_tray)")
    signals.discard("kit_ready(inspection_kit)")
    signals.discard("holding(husky_franka, inspection_kit)")
    signals.discard("kit_secured(husky_franka, inspection_kit)")
    signals.add("fallen_tool_unlocalized(inspection_kit)")
    signals.add("gripper_empty(husky_franka)")
    signals.add("stowed(husky_franka)")
    signals = {
        fact
        for fact in signals
        if not fact.startswith(("docked(b2_base, ", "docked(husky_base, "))
    }
    signals.add("stationary(b2_base)")
    signals.add("stowed(z1_thermal_arm)")
    signals.add("stationary(husky_base)")
    signals.add("arm_home(static_franka)")
    signals.add("gripper_empty(static_franka)")
    world = executor.inspection_world
    for dock in B2_DOCK_X:
        if world.at_dock("b2_base", dock):
            signals.add(f"docked(b2_base, {dock})")
    for dock in HUSKY_DOCK_X:
        if world.at_dock("husky_base", dock):
            signals.add(f"docked(husky_base, {dock})")
    return tuple(sorted(signals))


def _state_invariant(world: InspectionWorld) -> str:
    digest = hashlib.sha256()
    for array in (world.data.qpos, world.data.qvel, world.data.ctrl, world.data.mocap_pos):
        digest.update(np.asarray(array).tobytes())
    digest.update(np.asarray([world.data.time], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _validate_fault(fault: dict[str, Any]) -> None:
    if fault.get("fault_type") != "drop_tool":
        raise ValueError("Five-agent adaptive demo requires a drop_tool fault.")
    trigger = fault.get("trigger", {})
    expected = {
        "after_robot": "static_franka",
        "after_action": "place_inspection_kit_handoff",
        "before_robot": "husky_franka",
        "before_action": "load_inspection_kit",
        "object": "inspection_kit",
    }
    if any(trigger.get(key) != value for key, value in expected.items()):
        raise ValueError("Dropped-tool fault trigger does not match the guarded handoff boundary.")


def _accepted_nominal_model(outcome) -> Any:
    responses = outcome.planner_result.provider_responses
    return responses[-1].get("model_returned") if responses else None


def _model_matches(requested: str, returned: Any) -> bool:
    return isinstance(returned, str) and returned.lower().startswith(requested.lower())


def _iter_nodes(root):
    from ..bt import iter_nodes

    return iter_nodes(root)


def _new_output_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = root.resolve() / f"inspection-adaptive-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _artifact_hashes(output_dir: Path, *, excluded: set[str]) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
