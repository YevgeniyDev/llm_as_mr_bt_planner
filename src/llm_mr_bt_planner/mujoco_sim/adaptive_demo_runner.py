"""One-command, fault-blind LLM planning and same-state MuJoCo recovery demo."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
from argparse import Namespace
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import mujoco

from ..bt import iter_nodes
from ..config import PROJECT_ROOT, save_json, save_text
from ..llm.base import redact_secrets
from ..plan import Plan
from ..prompts import SYSTEM_PROMPT, build_prompt
from ..recovery import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_RECOVERY_MODEL,
    OpenAIResponsesRecoveryClient,
    RecoveryPlanningResult,
    plan_diff,
    plan_recovery,
)
from ..service import PipelineOutcome, PlannerService
from .assets import MENAGERIE_COMMIT, ensure_assets
from .executor import PhysicalExecutor
from .faults import load_fault_spec
from .recovery_runner import (
    DEFAULT_RECOVERY_FAULT,
    DEFAULT_RECOVERY_SCENARIO,
    _check_adapter_scope,
    _continuity_invariant,
    _run_fault_trial,
    _validate_fault_scope,
    _world_snapshot,
)
from .world import RECOVERY_TASK_ID


def run_adaptive_demo_cli(args: Namespace) -> int:
    """Generate a nominal BT, reveal a runtime fault, adapt, resume, and record."""
    _validate_args(args)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to .env or the environment before running "
            "the real-LLM adaptive demo."
        )

    scenario_path = Path(args.scenario).resolve()
    fault_path = Path(args.fault).resolve()
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario file does not exist: {scenario_path}")
    # Checking existence is intentionally the only pre-generation access to the fault path.
    if not fault_path.is_file():
        raise FileNotFoundError(f"Fault file does not exist: {fault_path}")

    output_dir = _new_demo_directory(Path(args.output))
    logger = DemoLogger(output_dir / "adaptive_demo.log")
    logger.event("START", f"Adaptive demo directory: {output_dir}")
    logger.event("BOUNDARY", "Fault configuration is sealed and has not been loaded.")
    started = time.perf_counter()
    try:
        result, manifest = _execute_demo(
            args,
            scenario_path=scenario_path,
            fault_path=fault_path,
            output_dir=output_dir,
            logger=logger,
            api_key=api_key,
            started=started,
        )
    except Exception as error:
        clean_error = redact_secrets(str(error))
        logger.event("ERROR", clean_error)
        save_json(
            output_dir / "adaptive_demo_error.json",
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "error": clean_error,
                "elapsed_wall_seconds": round(time.perf_counter() - started, 4),
                "events": logger.events,
            },
        )
        raise

    logger.event("DONE", f"Result: {'SUCCESS' if result else 'FAILURE'}")
    logger.event("DONE", f"Artifacts: {output_dir}")
    if not args.no_video:
        logger.event("DONE", f"Video: {output_dir / 'adaptive_demo.mp4'}")
    save_json(output_dir / "adaptive_demo_events.json", logger.events)
    manifest["files"] = _recursive_artifact_hashes(
        output_dir,
        excluded={"adaptive_demo_manifest.json"},
    )
    save_json(output_dir / "adaptive_demo_manifest.json", manifest)
    return 0 if result else 1


def _execute_demo(
    args: Namespace,
    *,
    scenario_path: Path,
    fault_path: Path,
    output_dir: Path,
    logger: DemoLogger,
    api_key: str,
    started: float,
) -> tuple[bool, dict[str, Any]]:
    shutil.copy2(scenario_path, output_dir / "scenario.json")
    scenario_document = PlannerService.load_json(scenario_path)
    service = PlannerService(output_dir / "nominal_generation")
    scenario = service.parse_scenario_document(scenario_document)
    if scenario.task_id != RECOVERY_TASK_ID:
        raise ValueError(
            f"Adaptive demo requires task_id {RECOVERY_TASK_ID!r}, got {scenario.task_id!r}."
        )

    nominal_user_prompt = build_prompt(scenario)
    nominal_system_prompt_path = output_dir / "nominal_system_prompt.txt"
    nominal_user_prompt_path = output_dir / "nominal_user_prompt.txt"
    save_text(nominal_system_prompt_path, SYSTEM_PROMPT)
    save_text(nominal_user_prompt_path, nominal_user_prompt)
    nominal_started = time.perf_counter()
    logger.event(
        "PHASE 1/5",
        f"Generating a nominal BT with openai/{args.model}; only the scenario is visible.",
    )
    with _heartbeat(
        logger,
        "NOMINAL LLM",
        "Waiting for nominal BT generation/repair",
        interval_seconds=args.heartbeat_seconds,
    ):
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
        raise RuntimeError("Nominal LLM generation did not produce an accepted Behavior Tree.")
    nominal_actions = validate_nominal_primary_only(outcome.plan)
    _check_adapter_scope(scenario, outcome.plan)
    save_json(output_dir / "nominal_behavior_tree.json", outcome.planner_result.plan)
    save_json(output_dir / "nominal_provider_responses.json", list(outcome.planner_result.provider_responses))
    logger.event(
        "PHASE 1/5",
        "Nominal BT accepted: it operates on primary_part and contains no spare_part branch.",
    )

    nominal_completed = time.perf_counter()
    logger.event(
        "BOUNDARY",
        "Nominal BT is complete. Loading the fault configuration for the first time now.",
    )
    fault_loaded = time.perf_counter()
    fault_spec = load_fault_spec(fault_path)
    _validate_fault_scope(fault_spec, scenario)
    shutil.copy2(fault_path, output_dir / "fault_specification.json")
    blindness = _fault_blindness_evidence(
        nominal_user_prompt=nominal_user_prompt,
        fault_path=fault_path,
        fault_id=fault_spec.fault_id,
        nominal_actions=nominal_actions,
        nominal_completed=nominal_completed,
        fault_loaded=fault_loaded,
    )
    blindness["nominal_prompt_file_sha256"] = _sha256_file(nominal_user_prompt_path)
    blindness["nominal_system_prompt_file_sha256"] = _sha256_file(
        nominal_system_prompt_path
    )
    save_json(output_dir / "fault_blindness_evidence.json", blindness)
    if not blindness["boundary_verified"]:
        raise RuntimeError("Fault-blind nominal-planning boundary could not be verified.")

    logger.event("PHASE 2/5", "Preparing pinned MuJoCo assets and the multi-object scene.")
    assets = ensure_assets(args.assets_dir, progress=lambda message: logger.event("ASSETS", message))
    logger.event(
        "PHASE 2/5",
        (
            "Starting headless MuJoCo; primary_part and spare_part are both present initially."
            if args.headless
            else (
                "Opening the live MuJoCo window automatically; primary_part and spare_part "
                "are both visible initially."
            )
        ),
    )

    recovery_client = OpenAIResponsesRecoveryClient(
        model=args.model,
        api_key=api_key,
        reasoning_effort=args.reasoning_effort,
    )
    planning_holder: dict[str, RecoveryPlanningResult] = {}

    def adapt(
        executor: PhysicalExecutor,
        measured_facts: tuple[str, ...],
        observation: dict[str, Any],
        invariant_before: dict[str, Any],
    ) -> tuple[PhysicalExecutor, dict[str, Any]]:
        logger.event(
            "PHASE 3/5",
            f"Unexpected {observation['classification']} detected for {observation['object']}.",
        )
        logger.event(
            "PHASE 4/5",
            "Robots safely stopped; MuJoCo state is frozen while the LLM adapts the BT.",
        )
        save_json(
            output_dir / "failure_snapshot.json",
            _world_snapshot(executor, measured_facts, observation),
        )
        recovery_started = time.perf_counter()
        with _heartbeat(
            logger,
            "RECOVERY LLM",
            "Failure handling in progress; MuJoCo state remains preserved",
            interval_seconds=args.heartbeat_seconds,
        ):
            planning = plan_recovery(
                recovery_client,
                scenario,
                measured_initial_state=measured_facts,
                failure_observation=observation,
                nominal_plan=outcome.plan,
                max_corrections=args.recovery_max_corrections,
                max_ticks=args.max_ticks,
                progress=lambda message, fraction: logger.progress(
                    "RECOVERY LLM", message, fraction
                ),
            )
        recovery_wall_seconds = time.perf_counter() - recovery_started
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
        save_text(
            output_dir / "behavior_tree_adaptation.diff",
            plan_diff(outcome.plan, planning.plan),
        )
        recovery_executor = PhysicalExecutor(
            executor.world,
            planning.runtime_scenario,
            planning.plan,
            executor.arms,
            executor.gait,
            progress=lambda message: logger.event("MUJOCO", message),
        )
        invariant_after = _continuity_invariant(executor.world)
        unchanged = invariant_before == invariant_after
        if not unchanged:
            raise RuntimeError("LLM planning changed MuJoCo state; same-state recovery was aborted.")
        logger.event(
            "PHASE 4/5",
            "Adapted BT passed validation; resuming the same MuJoCo model/data state.",
        )
        return recovery_executor, {
            "same_model_and_data": True,
            "no_reset_during_adaptation": True,
            "state_hash_unchanged_while_replanning": unchanged,
            "before": invariant_before,
            "after": invariant_after,
            "replanning_wall_seconds": round(recovery_wall_seconds, 4),
            "recovery_provider": planning.provider,
        }

    logger.event(
        "PHASE 3/5",
        "Executing the nominal BT. The fault will occur only after placement and release.",
    )
    trial = _run_fault_trial(
        scenario,
        outcome.plan,
        fault_spec,
        assets=assets,
        video_path=None if args.no_video else output_dir / "adaptive_demo.mp4",
        args=args,
        continue_after_failure=adapt,
        progress=lambda message: logger.event("MUJOCO", message),
    )
    save_json(output_dir / "adaptive_demo_report.json", trial)
    if "result" not in planning_holder:
        raise RuntimeError("Physical trial ended without invoking LLM recovery.")
    planning = planning_holder["result"]

    logger.event("PHASE 5/5", "Checking physical completion, continuity, and LLM provenance.")
    accepted_recovery_provenance = planning.attempts[-1]["provenance"]
    nominal_returned_model = _accepted_nominal_model(outcome)
    recovery_returned_model = accepted_recovery_provenance.get("model_returned")
    nominal_model_verified = _model_matches(args.model, nominal_returned_model)
    recovery_model_verified = _model_matches(args.model, recovery_returned_model)
    continuity = trial["continuity"]
    success = bool(
        trial["physical_execution"]["success"]
        and continuity["no_reset_during_adaptation"]
        and continuity["state_hash_unchanged_while_replanning"]
        and continuity["no_reset_through_completion"]
        and nominal_model_verified
        and recovery_model_verified
        and blindness["boundary_verified"]
    )
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": scenario.task_id,
        "success": success,
        "workflow": [
            "fault-blind nominal LLM generation",
            "independent BT validation and contract simulation",
            "automatic MuJoCo execution with initially visible spare inventory",
            "post-placement physical fault and safe stop",
            "failure-informed LLM BT adaptation",
            "same-state MuJoCo continuation with action-directed cameras",
        ],
        "fault_blindness": blindness,
        "nominal_planner": {
            "provider": outcome.planner_result.provider,
            "model_requested": args.model,
            "model_returned": nominal_returned_model,
            "model_verified": nominal_model_verified,
            "correction_rounds": outcome.planner_result.correction_rounds,
            "wall_seconds": round(nominal_completed - nominal_started, 4),
        },
        "recovery_planner": {
            "provider": planning.provider,
            "model_requested": planning.model,
            "model_returned": recovery_returned_model,
            "model_verified": recovery_model_verified,
            "reasoning_effort": planning.reasoning_effort,
            "attempt_count": len(planning.attempts),
            "wall_seconds": continuity["replanning_wall_seconds"],
        },
        "fault": trial["failure_observation"],
        "physical_execution": {
            "success": trial["physical_execution"]["success"],
            "continuity": continuity,
        },
        "live_viewer": trial["live_viewer"],
        "video": trial["recording"],
        "software": {
            "python": sys.version,
            "mujoco": getattr(mujoco, "__version__", "unknown"),
            "menagerie_commit": MENAGERIE_COMMIT,
        },
        "elapsed_wall_seconds": round(time.perf_counter() - started, 4),
    }
    return success, manifest


class DemoLogger:
    """Thread-safe, human-readable console/file logger with a machine-readable mirror."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def event(self, stage: str, message: str, *, fraction: float | None = None) -> None:
        clean = redact_secrets(message)
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        elapsed = time.perf_counter() - self.started
        percent = "" if fraction is None else f" {round(fraction * 100):3d}%"
        line = f"{now} | {elapsed:8.3f}s | {stage:<12}{percent} | {clean}"
        event = {
            "timestamp": now,
            "elapsed_wall_seconds": round(elapsed, 4),
            "stage": stage,
            "fraction": fraction,
            "message": clean,
        }
        with self._lock:
            self.events.append(event)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(line + "\n")
            print(line, flush=True)

    def progress(self, stage: str, message: str, fraction: float) -> None:
        self.event(stage, message, fraction=max(0.0, min(1.0, fraction)))


@contextmanager
def _heartbeat(
    logger: DemoLogger,
    stage: str,
    message: str,
    *,
    interval_seconds: float,
) -> Iterator[None]:
    stopped = threading.Event()
    heartbeat_started = time.perf_counter()

    def run() -> None:
        while not stopped.wait(interval_seconds):
            waiting = time.perf_counter() - heartbeat_started
            logger.event(stage, f"{message} ({waiting:.0f}s elapsed)")

    thread = threading.Thread(target=run, name="adaptive-demo-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=min(interval_seconds, 1.0))


def validate_nominal_primary_only(plan: Plan) -> list[dict[str, Any]]:
    """Reject pre-adapted/fallback plans so the initial LLM cannot consume the spare."""
    actions: list[dict[str, Any]] = []
    all_parameters: list[str] = []
    for robot, tree in plan.behavior_trees.items():
        for node in iter_nodes(tree):
            all_parameters.extend(node.parameters)
            if node.type == "Action":
                actions.append(
                    {"robot": robot, "action": node.name, "parameters": list(node.parameters)}
                )
    if "spare_part" in all_parameters:
        raise ValueError(
            "Nominal LLM BT mentions spare_part. The initial tree must remain nominal and "
            "must not contain a preplanned recovery branch."
        )
    if "primary_part" not in all_parameters:
        raise ValueError("Nominal LLM BT does not operate on primary_part.")
    required = {
        ("franka_a", "place_source_cradle", ("primary_part",)),
        ("unitree_go2_z1", "pick_source_cradle", ("primary_part",)),
    }
    actual = {
        (item["robot"], item["action"], tuple(item["parameters"])) for item in actions
    }
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"Nominal LLM BT is missing required handoff actions: {missing}")
    return actions


def _fault_blindness_evidence(
    *,
    nominal_user_prompt: str,
    fault_path: Path,
    fault_id: str,
    nominal_actions: list[dict[str, Any]],
    nominal_completed: float,
    fault_loaded: float,
) -> dict[str, Any]:
    prompt_contains_fault_id = fault_id in nominal_user_prompt
    loaded_after = fault_loaded >= nominal_completed
    return {
        "boundary_verified": loaded_after and not prompt_contains_fault_id,
        "fault_disclosed_to_nominal_llm": False,
        "fault_configuration_loaded_after_nominal_bt_accepted": loaded_after,
        "nominal_request_inputs": ["nominal system prompt", "scenario document"],
        "nominal_request_excluded_inputs": [
            "fault specification",
            "runtime failure observation",
            "recovery prompt",
        ],
        "nominal_request_prompt_sha256": _sha256_text(nominal_user_prompt),
        "nominal_system_request_sha256": _sha256_text(SYSTEM_PROMPT),
        "fault_specification_sha256": _sha256_file(fault_path),
        "fault_id_present_in_nominal_prompt": prompt_contains_fault_id,
        "nominal_actions": nominal_actions,
    }


def _accepted_nominal_model(outcome: PipelineOutcome) -> Any:
    responses = outcome.planner_result.provider_responses
    return responses[-1].get("model_returned") if responses else None


def _model_matches(requested: str, returned: Any) -> bool:
    return bool(
        isinstance(returned, str)
        and (returned == requested or returned.startswith(f"{requested}-"))
    )


def _validate_args(args: Namespace) -> None:
    if args.generation_max_corrections < 0 or args.recovery_max_corrections < 0:
        raise ValueError("Correction limits cannot be negative.")
    if args.max_ticks <= 0 or args.max_seconds <= 0:
        raise ValueError("--max-ticks and --max-seconds must be positive.")
    if args.heartbeat_seconds <= 0:
        raise ValueError("--heartbeat-seconds must be positive.")
    if args.realtime_factor <= 0:
        raise ValueError("--realtime-factor must be positive.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    for option, value in (("--video-width", args.video_width), ("--video-height", args.video_height)):
        if value <= 0 or value % 2:
            raise ValueError(f"{option} must be a positive even integer.")


def _new_demo_directory(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = root.resolve() / f"adaptive-demo-{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _recursive_artifact_hashes(output_dir: Path, *, excluded: set[str]) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): _sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def default_adaptive_demo_args() -> dict[str, Any]:
    return {
        "scenario": str(DEFAULT_RECOVERY_SCENARIO),
        "fault": str(DEFAULT_RECOVERY_FAULT),
        "output": str(PROJECT_ROOT / "outputs" / "adaptive_demo"),
        "model": DEFAULT_RECOVERY_MODEL,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "generation_max_corrections": 4,
        "recovery_max_corrections": 2,
        "max_ticks": 160,
        "max_seconds": 160.0,
        "heartbeat_seconds": 5.0,
        "headless": False,
        "realtime_factor": 1.0,
        "video_fps": 30,
        "video_width": 1920,
        "video_height": 1080,
    }
