"""Clean-room common-domain reproduction of Wang et al.'s LLM-HBT.

The implementation retains the paper's task-initialization, failure queue,
Alex allocation, robot action selection, and local/delegated BT update loop.
No executable author code, original prompt, response grammar, model identifier,
or decoding configuration was publicly available when this runner was built.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from ..artifacts import canonical_json
from ..bt import iter_nodes
from ..config import save_json, save_text
from ..domain import Scenario, scenario_to_dict
from ..llm import get_client
from ..plan import Plan, parse_plan
from ..predicates import canonical_predicate
from ..simulation import SimulationReport, simulate, skipped_simulation
from ..validation import ValidationReport, validate_plan
from .llm_bt_native import GroundAction, ground_action_templates
from .llm_hbt_native import (
    Assignment,
    DecisionInterface,
    HBTConstruction,
    condition_library,
    construct_forest,
    native_forest_document,
    parse_action_response,
    parse_assignment_response,
    parse_initialization_response,
)
from .llm_hbt_prompts import (
    ALEX_SYSTEM_PROMPT,
    INITIALIZATION_SYSTEM_PROMPT,
    ROBOT_SYSTEM_PROMPT,
    build_action_prompt,
    build_assignment_prompt,
    build_initialization_prompt,
)
from .llm_hbt_source import (
    ARXIV_SOURCE_SHA256,
    LLM_HBT_PAPER_URL,
    LLM_HBT_PROJECT_COMMIT,
    LLM_HBT_PROJECT_URL,
)

LLM_HBT_METHOD_ID = "llm-hbt"
PAPER_MODEL = "not reported"
DEFAULT_REPRODUCTION_MODEL = "gpt-4o-2024-08-06"
REPRODUCTION_TEMPERATURE = 0.0
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class LLMHBTError(RuntimeError):
    """Raised when an LLM-HBT run cannot be configured or replayed."""


@dataclass(frozen=True)
class StageResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMHBTGenerator(Protocol):
    provider: str
    model: str
    real_model_inference: bool
    temperature: float | None
    seed: int | None

    def generate(
        self,
        stage: str,
        system: str,
        user: str,
        context: dict[str, Any],
    ) -> StageResult:
        ...


class ProviderGenerator:
    """JSON-mode provider for the three paper-defined LLM decision boundaries."""

    real_model_inference = True

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str,
        *,
        seed: int | None = 42,
    ) -> None:
        provider = provider.lower()
        if provider not in {"openai", "anthropic"}:
            raise LLMHBTError("Provider must be 'openai' or 'anthropic'.")
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self.temperature = REPRODUCTION_TEMPERATURE if provider == "openai" else None
        self.seed = seed if provider == "openai" else None

    def generate(
        self,
        stage: str,
        system: str,
        user: str,
        context: dict[str, Any],  # noqa: ARG002
    ) -> StageResult:
        options: dict[str, Any] = {
            "model": self.model,
            "api_key": self._api_key,
            "max_tokens": 1200,
        }
        if self.provider == "openai":
            options.update(
                temperature=self.temperature,
                seed=self.seed,
                json_mode=True,
            )
        client = get_client(self.provider, **options)
        text = client.complete(system, user)
        if not isinstance(text, str) or not text.strip():
            raise LLMHBTError(f"LLM-HBT {stage} response contained no text.")
        metadata: dict[str, Any] = {
            "mode": "provider",
            "stage": stage,
            "provider": self.provider,
            "model": self.model,
            "real_model_inference": True,
            "temperature": self.temperature,
            "seed": self.seed,
            "output_characters": len(text),
        }
        response_metadata = getattr(client, "response_metadata", None)
        if isinstance(response_metadata, list) and response_metadata:
            metadata.update(response_metadata[-1])
            usage = metadata.pop("usage", None)
            if isinstance(usage, dict):
                metadata["input_tokens"] = usage.get("prompt_tokens")
                metadata["output_tokens"] = usage.get("completion_tokens")
                metadata["total_tokens"] = usage.get("total_tokens")
        return StageResult(text=text, metadata=metadata)


class ReplayGenerator:
    """Consume ordered archived responses and verify their native call context."""

    provider = "replay"
    real_model_inference = False
    temperature = None
    seed = None

    def __init__(
        self,
        responses: list[dict[str, Any]],
        *,
        model: str = "archived-llm-hbt-responses",
    ) -> None:
        if not responses or not all(isinstance(item, dict) for item in responses):
            raise LLMHBTError("LLM-HBT replay requires an ordered non-empty response array.")
        self.model = model
        self._responses = responses
        self._index = 0

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayGenerator:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        responses = document.get("responses") if isinstance(document, dict) else None
        if not isinstance(responses, list):
            raise LLMHBTError("LLM-HBT replay JSON requires a 'responses' object array.")
        return cls(responses, model=f"replay:{Path(path).name}")

    @property
    def remaining(self) -> int:
        return len(self._responses) - self._index

    def generate(
        self,
        stage: str,
        system: str,  # noqa: ARG002
        user: str,  # noqa: ARG002
        context: dict[str, Any],
    ) -> StageResult:
        if self._index >= len(self._responses):
            raise LLMHBTError(f"LLM-HBT replay exhausted before stage '{stage}'.")
        item = self._responses[self._index]
        self._index += 1
        if item.get("stage") != stage:
            raise LLMHBTError(
                f"LLM-HBT replay response {self._index} is for '{item.get('stage')}', "
                f"expected '{stage}'."
            )
        for key in ("condition", "robot", "requester", "track"):
            if key in item and not _replay_context_matches(
                key,
                item[key],
                context.get(key),
            ):
                raise LLMHBTError(
                    f"LLM-HBT replay response {self._index} has {key}={item[key]!r}, "
                    f"expected {context.get(key)!r}."
                )
        response = item.get("response")
        if isinstance(response, dict | list):
            text = json.dumps(response)
        elif isinstance(response, str) and response.strip():
            text = response
        else:
            raise LLMHBTError(
                f"LLM-HBT replay response {self._index} has no JSON/string response."
            )
        return StageResult(
            text=text,
            metadata={
                "mode": "replay",
                "stage": stage,
                "provider": self.provider,
                "real_model_inference": False,
                "output_characters": len(text),
            },
        )


def _replay_context_matches(key: str, archived: Any, current: Any) -> bool:
    """Compare predicate context canonically while keeping identifiers exact."""
    if key != "condition" or not isinstance(archived, str) or not isinstance(current, str):
        return archived == current
    try:
        return canonical_predicate(archived) == canonical_predicate(current)
    except ValueError:
        return False


@dataclass(frozen=True)
class LLMHBTRun:
    directory: Path
    canonical_plan: Path
    accepted_plan: Path | None
    validation_report: Path
    simulation_trace: Path
    metrics: Path
    manifest: Path
    plan_generation_success: bool
    static_validity: bool
    symbolic_goal_success: bool


class _RecordedDecisions(DecisionInterface):
    def __init__(
        self,
        scenario: Scenario,
        generator: LLMHBTGenerator,
        prompts_dir: Path,
        native_dir: Path,
        *,
        track: str,
        failure_observation: dict[str, Any] | None,
    ) -> None:
        self.scenario = scenario
        self.generator = generator
        self.prompts_dir = prompts_dir
        self.native_dir = native_dir
        self.track = track
        self.failure_observation = failure_observation
        self.actions = ground_action_templates(scenario)
        self.calls: list[dict[str, Any]] = []
        self.artifact_files: list[Path] = []

    def initialize(self) -> tuple[str, ...]:
        library = condition_library(self.scenario)
        user = build_initialization_prompt(self.scenario, library)
        result = self._call(
            "initialize",
            INITIALIZATION_SYSTEM_PROMPT,
            user,
            {"track": self.track},
        )
        return parse_initialization_response(result.text, self.scenario)

    def assign(
        self,
        failed_condition: str,
        requester: str | None,
        observed_state: set[str],
    ) -> Assignment:
        user = build_assignment_prompt(
            self.scenario,
            self.actions,
            failed_condition=failed_condition,
            requester=requester,
            observed_state=observed_state,
            failure_observation=self.failure_observation,
        )
        result = self._call(
            "assign",
            ALEX_SYSTEM_PROMPT,
            user,
            {
                "track": self.track,
                "condition": failed_condition,
                "requester": requester,
            },
        )
        return parse_assignment_response(
            result.text,
            self.scenario,
            requester=requester,
        )

    def select_action(
        self,
        failed_condition: str,
        assignment: Assignment,
        observed_state: set[str],
    ) -> GroundAction:
        user = build_action_prompt(
            self.scenario,
            self.actions,
            failed_condition=failed_condition,
            selected_robot=assignment.robot,
            task=assignment.task,
            observed_state=observed_state,
            failure_observation=self.failure_observation,
        )
        result = self._call(
            "select_action",
            ROBOT_SYSTEM_PROMPT,
            user,
            {
                "track": self.track,
                "condition": failed_condition,
                "requester": None,
                "robot": assignment.robot,
            },
        )
        return parse_action_response(
            result.text,
            self.actions,
            robot=assignment.robot,
            failed_condition=failed_condition,
        )

    def _call(
        self,
        stage: str,
        system: str,
        user: str,
        context: dict[str, Any],
    ) -> StageResult:
        index = len(self.calls) + 1
        stem = f"{index:03d}.{stage}"
        system_path = self.prompts_dir / f"{stem}.system.txt"
        user_path = self.prompts_dir / f"{stem}.user.txt"
        save_text(system_path, system)
        save_text(user_path, user)
        self.artifact_files.extend([system_path, user_path])
        started = time.perf_counter()
        result = self.generator.generate(stage, system, user, context)
        elapsed = round(time.perf_counter() - started, 4)
        response_path = self.native_dir / f"{stem}.response.txt"
        save_text(response_path, result.text)
        self.artifact_files.append(response_path)
        metadata = {"elapsed_wall_seconds": elapsed, **result.metadata}
        self.calls.append(
            {
                "index": index,
                "stage": stage,
                "context": context,
                "system_prompt": system_path.name,
                "user_prompt": user_path.name,
                "response": response_path.name,
                "metadata": metadata,
            }
        )
        return StageResult(result.text, metadata)


def run_llm_hbt(
    scenario: Scenario,
    generator: LLMHBTGenerator,
    output_root: str | Path,
    *,
    max_extensions: int = 100,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> LLMHBTRun:
    """Generate initial conditions and execute LLM-HBT's nominal extension loop."""
    started = time.perf_counter()
    base = _new_run_directory(Path(output_root), scenario.task_id, "nominal")
    prompts_dir = base / "prompts"
    native_dir = base / "native"
    prompts_dir.mkdir(parents=True, exist_ok=False)
    native_dir.mkdir(parents=True, exist_ok=False)
    decisions = _RecordedDecisions(
        scenario,
        generator,
        prompts_dir,
        native_dir,
        track="nominal",
        failure_observation=None,
    )
    generation_errors: list[dict[str, Any]] = []
    construction: HBTConstruction | None = None
    conditions: tuple[str, ...] = ()
    try:
        conditions = decisions.initialize()
        construction = construct_forest(
            scenario,
            conditions,
            decisions,
            namespace="llmhbt.nominal",
            max_extensions=max_extensions,
        )
        _reject_unused_replay(generator)
    except Exception as error:
        generation_errors.append({"type": "llm_hbt_nominal_error", "message": str(error)})
    artifact_files = _write_native_artifacts(
        scenario,
        native_dir,
        decisions,
        conditions,
        construction,
        generation_errors,
    )
    behavior_trees = (
        {robot: tree.to_dict() for robot, tree in construction.trees.items()}
        if construction is not None
        else {}
    )
    plan_document = {
        "schema_version": "2.0",
        "mission_id": scenario.task_id,
        "behavior_trees": behavior_trees,
    }
    return _finish_run(
        base,
        scenario,
        plan_document,
        [*decisions.artifact_files, *artifact_files],
        generation_errors,
        decisions.calls,
        generator,
        max_ticks=max_ticks,
        wall_seconds=time.perf_counter() - started,
        invocation=invocation,
        track="nominal",
        nominal_run=None,
        recovery_details=None,
    )


def run_llm_hbt_recovery(
    runtime_scenario: Scenario,
    nominal_run: str | Path,
    failure_snapshot: dict[str, Any],
    generator: LLMHBTGenerator,
    output_root: str | Path,
    *,
    max_extensions: int = 100,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> LLMHBTRun:
    """Detect the runtime failure and construct local/delegated recovery insertions."""
    started = time.perf_counter()
    nominal_directory = Path(nominal_run).resolve()
    nominal_manifest = nominal_directory / "manifest.json"
    conditions_path = nominal_directory / "native" / "initial_conditions.json"
    forest_path = nominal_directory / "native" / "native_forest.json"
    if not nominal_manifest.is_file() or not conditions_path.is_file() or not forest_path.is_file():
        raise LLMHBTError(
            "Nominal LLM-HBT run lacks manifest, initial conditions, or native forest."
        )
    manifest_document = json.loads(nominal_manifest.read_text(encoding="utf-8"))
    if manifest_document.get("method", {}).get("id") != LLM_HBT_METHOD_ID:
        raise LLMHBTError("--nominal-run is not an LLM-HBT run.")
    conditions_document = json.loads(conditions_path.read_text(encoding="utf-8"))
    if not isinstance(conditions_document, list) or not all(
        isinstance(item, str) for item in conditions_document
    ):
        raise LLMHBTError("Nominal LLM-HBT initial conditions are malformed.")
    conditions = tuple(conditions_document)
    observation = failure_snapshot.get("failure_observation")
    measured = failure_snapshot.get("measured_initial_state")
    if not isinstance(observation, dict) or not isinstance(measured, list):
        raise LLMHBTError(
            "Failure snapshot requires failure_observation and measured_initial_state."
        )

    base = _new_run_directory(Path(output_root), runtime_scenario.task_id, "recovery")
    prompts_dir = base / "prompts"
    native_dir = base / "native"
    prompts_dir.mkdir(parents=True, exist_ok=False)
    native_dir.mkdir(parents=True, exist_ok=False)
    decisions = _RecordedDecisions(
        runtime_scenario,
        generator,
        prompts_dir,
        native_dir,
        track="recovery",
        failure_observation=observation,
    )
    generation_errors: list[dict[str, Any]] = []
    construction: HBTConstruction | None = None
    try:
        construction = construct_forest(
            runtime_scenario,
            conditions,
            decisions,
            namespace="llmhbt.recovery",
            max_extensions=max_extensions,
        )
        recoveries = [
            selected
            for selected in construction.actions
            if selected.action.name == "recover_fallen_part"
            and selected.action.parameters == ("primary_part", "source_floor")
        ]
        if not recoveries:
            raise LLMHBTError(
                "LLM-HBT recovery did not insert recover_fallen_part(primary_part,source_floor)."
            )
        _reject_unused_replay(generator)
    except Exception as error:
        generation_errors.append({"type": "llm_hbt_recovery_error", "message": str(error)})

    snapshot_path = native_dir / "failure_snapshot.json"
    before_path = native_dir / "native_forest_before.json"
    save_json(snapshot_path, failure_snapshot)
    save_json(before_path, json.loads(forest_path.read_text(encoding="utf-8")))
    artifact_files = [snapshot_path, before_path]
    artifact_files.extend(
        _write_native_artifacts(
            runtime_scenario,
            native_dir,
            decisions,
            conditions,
            construction,
            generation_errors,
        )
    )
    update_path = native_dir / "online_update.json"
    save_json(
        update_path,
        {
            "failure_detection": observation,
            "nominal_tree_preserved_as": before_path.name,
            "operations": (
                [selected.to_dict() for selected in construction.actions]
                if construction is not None
                else []
            ),
            "native_semantics": "BT_Extention plus BT_Melt with delegated root-priority insertion",
            "common_observation": (
                "remaining continuation materialized from the measured post-failure state because "
                "the common simulator does not persist native per-node tick memory"
            ),
        },
    )
    artifact_files.append(update_path)
    behavior_trees = (
        {robot: tree.to_dict() for robot, tree in construction.trees.items()}
        if construction is not None
        else {}
    )
    plan_document = {
        "schema_version": "2.0",
        "mission_id": runtime_scenario.task_id,
        "behavior_trees": behavior_trees,
    }
    selected_names = (
        [selected.action.name for selected in construction.actions]
        if construction is not None
        else []
    )
    return _finish_run(
        base,
        runtime_scenario,
        plan_document,
        [*decisions.artifact_files, *artifact_files],
        generation_errors,
        decisions.calls,
        generator,
        max_ticks=max_ticks,
        wall_seconds=time.perf_counter() - started,
        invocation=invocation,
        track="recovery",
        nominal_run=nominal_directory,
        recovery_details={
            "failure_detected_before_recovery_calls": True,
            "real_llm_recalled_after_failure": generator.real_model_inference and bool(decisions.calls),
            "post_failure_call_count": len(decisions.calls),
            "same_nominal_conditions_reused": True,
            "same_object_recovery": "recover_fallen_part" in selected_names,
            "native_update": "online extension/insertion",
            "nominal_manifest_sha256": _sha256_file(nominal_manifest),
        },
    )


def _write_native_artifacts(
    scenario: Scenario,
    native_dir: Path,
    decisions: _RecordedDecisions,
    conditions: tuple[str, ...],
    construction: HBTConstruction | None,
    generation_errors: list[dict[str, Any]],
) -> list[Path]:
    paths = {
        "condition_library": native_dir / "condition_library.json",
        "action_libraries": native_dir / "action_libraries.json",
        "initial_conditions": native_dir / "initial_conditions.json",
        "calls": native_dir / "llm_calls.json",
        "failure_queue": native_dir / "failure_queue.json",
        "trace": native_dir / "update_trace.json",
        "forest": native_dir / "native_forest.json",
        "canonical_observation": native_dir / "canonical_observation.json",
    }
    save_json(paths["condition_library"], condition_library(scenario))
    save_json(
        paths["action_libraries"],
        {
            robot.id: [
                action.to_dict()
                for action in decisions.actions
                if action.robot == robot.id
            ]
            for robot in scenario.robots
        },
    )
    save_json(paths["initial_conditions"], list(conditions))
    save_json(paths["calls"], decisions.calls)
    if construction is None:
        save_json(paths["failure_queue"], [])
        save_json(paths["trace"], {"events": [], "errors": generation_errors})
        save_json(paths["forest"], {})
        save_json(paths["canonical_observation"], {})
    else:
        save_json(paths["failure_queue"], construction.failure_queue)
        save_json(
            paths["trace"],
            {
                "events": construction.trace,
                "selected_actions": [item.to_dict() for item in construction.actions],
                "final_planning_state": list(construction.final_planning_state),
                "unresolved": construction.unresolved,
            },
        )
        save_json(paths["forest"], native_forest_document(construction))
        save_json(
            paths["canonical_observation"],
            {robot: tree.to_dict() for robot, tree in construction.trees.items()},
        )
    return list(paths.values())


def _finish_run(
    base: Path,
    scenario: Scenario,
    plan_document: dict[str, Any],
    artifact_files: list[Path],
    generation_errors: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    generator: LLMHBTGenerator,
    *,
    max_ticks: int,
    wall_seconds: float,
    invocation: list[str] | None,
    track: str,
    nominal_run: Path | None,
    recovery_details: dict[str, Any] | None,
) -> LLMHBTRun:
    plan = parse_plan(plan_document)
    validation = validate_plan(plan, scenario, suggest_producers=False)
    simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
    plan_generation_success = not generation_errors and set(plan.behavior_trees) == scenario.robot_ids
    canonical_path = base / "canonical_plan.json"
    accepted_path = (
        base / "accepted_plan.json"
        if plan_generation_success and validation.valid and simulation.success
        else None
    )
    validation_path = base / "validation_report.json"
    simulation_path = base / "simulation_trace.json"
    metrics_path = base / "metrics.json"
    scenario_path = base / "scenario.json"
    manifest_path = base / "manifest.json"
    save_json(canonical_path, plan_document)
    if accepted_path is not None:
        save_json(accepted_path, plan_document)
    save_json(
        validation_path,
        {
            "valid": validation.valid,
            "errors": validation.to_dicts(),
            "native_generation_errors": generation_errors,
        },
    )
    save_json(simulation_path, simulation.to_dict())
    save_json(scenario_path, scenario_to_dict(scenario))
    metrics = _metrics_payload(
        plan,
        validation,
        simulation,
        plan_generation_success,
        wall_seconds,
        generator,
        calls,
        generation_errors,
        track,
    )
    save_json(metrics_path, metrics)
    artifact_files.extend(
        [canonical_path, validation_path, simulation_path, metrics_path, scenario_path]
    )
    if accepted_path is not None:
        artifact_files.append(accepted_path)
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "id": LLM_HBT_METHOD_ID,
            "name": "LLM-HBT",
            "paper": LLM_HBT_PAPER_URL,
            "official_project_page": LLM_HBT_PROJECT_URL,
            "official_project_commit": LLM_HBT_PROJECT_COMMIT,
            "official_executable_code_found": False,
            "software_license": "not declared (project page contains no software)",
            "implementation": "paper-based clean-room common-domain reproduction",
            "paper_model": PAPER_MODEL,
            "selected_model": generator.model,
            "track": track,
        },
        "generator": {
            "provider": generator.provider,
            "model": generator.model,
            "real_model_inference": generator.real_model_inference,
            "calls": calls,
        },
        "invocation": invocation or [],
        "scenario_sha256": _sha256_json(scenario_to_dict(scenario)),
        "canonical_plan_sha256": _sha256_json(plan_document),
        "nominal_run": str(nominal_run) if nominal_run is not None else None,
        "recovery": recovery_details,
        "fidelity": {
            "native_architecture": [
                "LLM task initialization into ordered condition nodes",
                "continuous failure-node detection and queueing",
                "Alex LLM assignment over heterogeneous robot action libraries",
                "second LLM selection of one producing action for the assigned robot",
                "independent extension or delegated high-priority insertion with requester monitoring",
                "iterative precondition/postcondition-based BT extension",
            ],
            "paper_inference": {
                "model": PAPER_MODEL,
                "prompt": "not published",
                "response_grammar": "not published",
                "decoding_parameters": "not published",
            },
            "reproduction_inference": {
                "model": generator.model,
                "temperature": generator.temperature,
                "seed": generator.seed,
                "response_grammar": "strict JSON",
            },
            "input_adaptations": [
                "paper action and condition libraries are instantiated from common grounded capability contracts",
                "common symbolic state replaces robot-specific partial-observation middleware",
                "the shared post-drop snapshot is exposed only in the recovery track",
            ],
            "output_adaptations": [
                "paper Selector/Sequence/Condition/Action notation is observed as common BT JSON",
                "delegated requester monitoring is represented by bounded common WaitFor leaves",
                "exclusive-resource operations required by a selected low-level capability are explicit common leaves",
                "recovery materializes the remaining continuation because common simulation does not persist native tick memory",
            ],
            "semantic_rewrites": [],
            "validator_feedback_to_model": False,
            "known_reproduction_limits": [
                "the author-owned repository contains only a project page and says code is coming soon",
                "the paper does not identify the LLM model, original prompts, output parser, temperature, or seed",
                "Behavior-1K simulation, quadruped/drone/arm middleware, and cafe hardware are replaced by common evaluators",
                "the common runner uses an explicit safety bound on tree extensions",
            ],
            "provenance": {
                "arxiv_v1_source_sha256": ARXIV_SOURCE_SHA256,
                "project_commit": LLM_HBT_PROJECT_COMMIT,
            },
        },
        "results": metrics,
        "files": {
            _relative(base, path): _sha256_file(path)
            for path in artifact_files
            if path.is_file()
        },
    }
    save_json(manifest_path, manifest)
    return LLMHBTRun(
        directory=base,
        canonical_plan=canonical_path,
        accepted_plan=accepted_path,
        validation_report=validation_path,
        simulation_trace=simulation_path,
        metrics=metrics_path,
        manifest=manifest_path,
        plan_generation_success=plan_generation_success,
        static_validity=validation.valid,
        symbolic_goal_success=simulation.goal_success,
    )


def _metrics_payload(
    plan: Plan,
    validation: ValidationReport,
    simulation: SimulationReport,
    plan_generation_success: bool,
    wall_seconds: float,
    generator: LLMHBTGenerator,
    calls: list[dict[str, Any]],
    generation_errors: list[dict[str, Any]],
    track: str,
) -> dict[str, Any]:
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    input_tokens = _sum_call_metadata(calls, "input_tokens")
    output_tokens = _sum_call_metadata(calls, "output_tokens")
    common: dict[str, Any] = {
        "track": track,
        "wall_seconds": round(wall_seconds, 4),
        "model_calls": len(calls) if generator.real_model_inference else 0,
        "archived_response_count": len(calls) if not generator.real_model_inference else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "monetary_cost": None,
        "bt_node_count": len(nodes),
        "action_count": sum(node.type == "Action" for node in nodes),
        "maximum_tree_depth": max(
            (_tree_depth(tree) for tree in plan.behavior_trees.values()),
            default=0,
        ),
        "validation_error_count": len(validation.errors),
        "simulation_error_count": len(simulation.errors),
        "native_generation_errors": generation_errors,
    }
    if track == "recovery":
        common.update(
            recovery_plan_success=plan_generation_success,
            recovery_validity=validation.valid,
            recovery_goal_success=simulation.goal_success,
            plan_generation_success=None,
            static_validity=None,
            symbolic_goal_success=None,
        )
    else:
        common.update(
            plan_generation_success=plan_generation_success,
            static_validity=validation.valid,
            symbolic_goal_success=simulation.goal_success,
            nominal_execution_success=None,
        )
    return common


def _reject_unused_replay(generator: LLMHBTGenerator) -> None:
    remaining = getattr(generator, "remaining", 0)
    if isinstance(remaining, int) and remaining:
        raise LLMHBTError(f"LLM-HBT replay has {remaining} unused ordered response(s).")


def _sum_call_metadata(calls: list[dict[str, Any]], key: str) -> int | None:
    values = [call.get("metadata", {}).get(key) for call in calls]
    integers = [value for value in values if isinstance(value, int)]
    return sum(integers) if integers else None


def _tree_depth(node) -> int:
    return 1 + max((_tree_depth(child) for child in node.children), default=0)


def _new_run_directory(output_root: Path, task_id: str, variant: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / f"{_safe_name(task_id)}-{variant}-{stamp}"
    counter = 1
    while base.exists():
        base = output_root / f"{_safe_name(task_id)}-{variant}-{stamp}-{counter}"
        counter += 1
    base.mkdir(parents=True, exist_ok=False)
    return base


def _safe_name(value: str) -> str:
    return _SAFE_ID.sub("-", value).strip("-") or "item"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
