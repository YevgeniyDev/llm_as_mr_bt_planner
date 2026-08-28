"""BETR-XP-LLM goal interpretation, reactive planning, and failure resolution."""

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
from ..llm.openai_client import OpenAIClient
from ..plan import Plan, parse_plan
from ..simulation import SimulationReport, simulate, skipped_simulation
from ..validation import ValidationReport, validate_plan
from .betr_xp_native import (
    BetrXPNativeError,
    ConditionSchema,
    EntityAlias,
    PlannedAlternative,
    build_condition_schemas,
    build_entity_aliases,
    encode_predicate,
    grounded_skill_library,
    native_flat_policy,
    parse_goal_response,
    parse_parameter_response,
    pickup_binding,
    plan_formula,
    plan_predicates,
    select_lowest_cost,
)
from .betr_xp_source import (
    BETR_XP_DOI,
    BETR_XP_PAPER_URL,
    BETR_XP_REPOSITORY_COMMIT,
    BETR_XP_REPOSITORY_URL,
)
from .llm_bt_native import ground_action_templates

BETR_XP_METHOD_ID = "betr-xp-llm"
PAPER_MODEL = "gpt-4-1106-preview"
PAPER_TEMPERATURE = 0.1
PAPER_TOP_P = 0.1
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class BetrXPError(RuntimeError):
    """Raised when a BETR-XP-LLM comparison run cannot be configured."""


@dataclass(frozen=True)
class StageResponse:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BetrXPCaller(Protocol):
    provider: str
    model: str
    real_model_inference: bool

    def generate(self, prompt: str, *, stage: str) -> StageResponse:
        ...


class ProviderCaller:
    """Use the paper's one-user-message Chat Completions configuration."""

    provider = "openai"
    real_model_inference = True

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self._client = OpenAIClient(
            model=model,
            api_key=api_key,
            temperature=PAPER_TEMPERATURE,
            top_p=PAPER_TOP_P,
            seed=seed,
            json_mode=False,
            max_tokens=1200,
        )
        self._seed = seed

    def generate(self, prompt: str, *, stage: str) -> StageResponse:
        result = self._client.create_chat_completion(
            [{"role": "user", "content": prompt}]
        )
        content = result.get("choices", [{}])[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise BetrXPError(f"BETR-XP-LLM {stage} response contained no text.")
        metadata: dict[str, Any] = {
            "mode": "provider",
            "stage": stage,
            "provider": self.provider,
            "model": self.model,
            "real_model_inference": True,
            "temperature": PAPER_TEMPERATURE,
            "top_p": PAPER_TOP_P,
            "seed": self._seed,
            "message_roles": ["user"],
            "output_characters": len(content),
        }
        if self._client.response_metadata:
            metadata.update(self._client.response_metadata[-1])
            usage = metadata.pop("usage", None)
            if isinstance(usage, dict):
                metadata["input_tokens"] = usage.get("prompt_tokens")
                metadata["output_tokens"] = usage.get("completion_tokens")
                metadata["total_tokens"] = usage.get("total_tokens")
        return StageResponse(content, metadata)


class ReplayCaller:
    provider = "replay"
    real_model_inference = False

    def __init__(
        self,
        responses: dict[str, str],
        *,
        model: str = "archived-betr-xp-responses",
    ) -> None:
        if not responses or any(not isinstance(value, str) or not value.strip() for value in responses.values()):
            raise BetrXPError("BETR-XP-LLM replay responses must be non-empty strings.")
        self.responses = responses
        self.model = model
        self.calls: list[str] = []

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayCaller:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in document.items()
        ):
            raise BetrXPError("BETR-XP-LLM replay JSON must map stage names to strings.")
        return cls(document, model=f"replay:{Path(path).name}")

    def generate(self, prompt: str, *, stage: str) -> StageResponse:
        response = self.responses.get(stage)
        if response is None:
            raise BetrXPError(f"BETR-XP-LLM replay has no '{stage}' response.")
        self.calls.append(stage)
        return StageResponse(
            response,
            {
                "mode": "replay",
                "stage": stage,
                "provider": self.provider,
                "real_model_inference": False,
                "input_characters": len(prompt),
                "output_characters": len(response),
            },
        )


@dataclass(frozen=True)
class BetrXPRun:
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


def build_goal_prompt(
    scenario: Scenario,
    schemas: list[ConditionSchema],
    entities: list[EntityAlias],
) -> str:
    """Adapt the released strict, description-rich goal prompt to common symbols."""
    condition_lines = "\n".join(
        f"{schema.native}_{'_'.join(f'<{item}>' for item in schema.argument_types)}: "
        f"{schema.description}"
        if schema.argument_types
        else f"{schema.native}: {schema.description}"
        for schema in schemas
    )
    object_groups: dict[str, list[str]] = {}
    for entity in entities:
        object_groups.setdefault(entity.entity_type, []).append(entity.native)
    object_lines = "\n".join(
        f"<{entity_type}>=[{', '.join(sorted(values))}]"
        for entity_type, values in sorted(object_groups.items())
    )
    facts = "\n".join(f"- {fact}" for fact in scenario.initial_state)
    examples = _goal_examples(scenario, schemas, entities)
    return "\n".join(
        [
            "You are a helpful assistant that translates robot task instructions to one or more ",
            "formal goal conditions as a well-formed first-order-logic formula. Use only the ",
            "condition and object identifiers below.",
            "",
            "Logical operators:",
            "& combines conditions with AND.",
            "| combines alternatives with OR.",
            "~ negates one condition.",
            "",
            "Conditions:",
            condition_lines,
            "",
            "Objects:",
            object_lines,
            "",
            "Scene information:",
            facts,
            "",
            "Examples:",
            examples,
            "",
            "Use every relevant part of the instruction. Only use listed identifiers and the exact ",
            "underscore-separated Condition_Object syntax. Do not include reasoning or commentary.",
            "RESPONSE FORMAT:",
            "Goal: one well-formed formula",
            "",
            "Now translate this task instruction:",
            scenario.instruction,
        ]
    )


def build_parameter_recovery_prompt(
    scenario: Scenario,
    failure_snapshot: dict[str, Any],
) -> str:
    observation = failure_snapshot.get("failure_observation")
    if not isinstance(observation, dict):
        raise BetrXPError("Failure snapshot has no failure_observation object.")
    part = observation.get("object")
    location = observation.get("recovery_location")
    facts = "\n".join(f"- {fact}" for fact in scenario.initial_state)
    locations = sorted(entity.id for entity in scenario.entities if entity.type == "location")
    return "\n".join(
        [
            "You are a helpful assistant that resolves robot errors by figuring out what value is ",
            "appropriate for a missing or invalid action parameter.",
            "",
            f"Task: {scenario.instruction}",
            f"Available location values: {', '.join(locations)}",
            "Scene information:",
            facts,
            "",
            f"Failing action: Pick {part} from source_cradle",
            (
                "Error message: The pickup location parameter is no longer valid. "
                f"The intact object was observed at {location} after falling before pickup."
            ),
            "",
            "Respond only in this format:",
            "Reasoning: Briefly explain which observed location should configure the pickup skill.",
            "Parameter value: Use exactly one available location value.",
            "",
            "Now resolve this error:",
        ]
    )


def run_betr_xp(
    scenario: Scenario,
    caller: BetrXPCaller,
    output_root: str | Path,
    *,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> BetrXPRun:
    """Formalize goals once and deterministically generate the nominal reactive forest."""
    started = time.perf_counter()
    base = _new_run_directory(Path(output_root), scenario.task_id, "nominal")
    prompts_dir = base / "prompts"
    native_dir = base / "native"
    prompts_dir.mkdir(parents=True, exist_ok=False)
    native_dir.mkdir(parents=True, exist_ok=False)
    schemas = build_condition_schemas(scenario)
    entities = build_entity_aliases(scenario)
    prompt = build_goal_prompt(scenario, schemas, entities)
    artifact_files: list[Path] = []
    generation_errors: list[dict[str, Any]] = []
    call_metadata: dict[str, Any] = {}
    selected: PlannedAlternative | None = None
    formula_document: dict[str, Any] = {}
    alternatives_document: list[dict[str, Any]] = []

    for path, value in (
        (prompts_dir / "goal.user.txt", prompt),
        (native_dir / "condition_schemas.json", [item.to_dict() for item in schemas]),
        (native_dir / "entity_aliases.json", [item.to_dict() for item in entities]),
        (native_dir / "skill_database.json", grounded_skill_library(scenario)),
    ):
        if isinstance(value, str):
            save_text(path, value)
        else:
            save_json(path, value)
        artifact_files.append(path)

    try:
        call_started = time.perf_counter()
        stage = caller.generate(prompt, stage="goal_response")
        call_metadata = {
            "elapsed_wall_seconds": round(time.perf_counter() - call_started, 4),
            **stage.metadata,
        }
        response_path = native_dir / "goal_response.txt"
        save_text(response_path, stage.text)
        artifact_files.append(response_path)
        formula = parse_goal_response(stage.text, schemas, entities)
        formula_document = formula.to_dict()
        planned = plan_formula(scenario, formula)
        selected = select_lowest_cost(planned)
        alternatives_document = [
            {
                "index": item.index,
                "estimated_cost": item.estimated_cost,
                "selected": item.index == selected.index,
                "goals": [goal.to_dict() for goal in item.parsed_goals],
                "unresolved": item.expansion.unresolved,
            }
            for item in planned
        ]
        if selected.expansion.unresolved:
            generation_errors.extend(
                {"type": "unresolved_planner_condition", **item}
                for item in selected.expansion.unresolved
            )
    except Exception as error:
        generation_errors.append(
            {"type": "goal_interpretation_or_planning_error", "message": str(error)}
        )

    formula_path = native_dir / "formal_goal.json"
    alternatives_path = native_dir / "goal_alternatives.json"
    selected_goals_path = native_dir / "selected_goals.json"
    initial_path = native_dir / "initial_forest.json"
    trace_path = native_dir / "planner_trace.json"
    native_policy_path = native_dir / "native_policy.bt.json"
    expanded_path = native_dir / "expanded_forest.json"
    save_json(formula_path, formula_document)
    save_json(alternatives_path, alternatives_document)
    if selected is None:
        goals_document: list[dict[str, Any]] = []
        initial_document: dict[str, Any] = {}
        trace_document: dict[str, Any] = {"events": [], "unresolved": []}
        native_policy: dict[str, Any] = {}
        behavior_trees: dict[str, Any] = {}
    else:
        goals_document = [goal.to_dict() for goal in selected.parsed_goals]
        initial_document = {
            robot: tree.to_dict() for robot, tree in selected.expansion.initial_trees.items()
        }
        trace_document = {
            "events": selected.expansion.trace,
            "unresolved": selected.expansion.unresolved,
            "selected_alternative": selected.index,
            "estimated_cost": selected.estimated_cost,
        }
        native_policy = native_flat_policy(selected.expansion.trees)
        behavior_trees = {
            robot: tree.to_dict() for robot, tree in selected.expansion.trees.items()
        }
    save_json(selected_goals_path, goals_document)
    save_json(initial_path, initial_document)
    save_json(trace_path, trace_document)
    save_json(native_policy_path, native_policy)
    save_json(expanded_path, behavior_trees)
    artifact_files.extend(
        [
            formula_path,
            alternatives_path,
            selected_goals_path,
            initial_path,
            trace_path,
            native_policy_path,
            expanded_path,
        ]
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
        artifact_files,
        generation_errors,
        call_metadata,
        caller,
        max_ticks=max_ticks,
        wall_seconds=time.perf_counter() - started,
        invocation=invocation,
        track="nominal",
        nominal_run=None,
        recovery_details=None,
    )


def run_betr_xp_recovery(
    runtime_scenario: Scenario,
    nominal_run: str | Path,
    failure_snapshot: dict[str, Any],
    caller: BetrXPCaller,
    output_root: str | Path,
    *,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> BetrXPRun:
    """Resolve the failed pickup location, update the policy, and plan the continuation."""
    started = time.perf_counter()
    nominal_directory = Path(nominal_run).resolve()
    goals_path = nominal_directory / "native" / "selected_goals.json"
    policy_path = nominal_directory / "native" / "native_policy.bt.json"
    nominal_manifest = nominal_directory / "manifest.json"
    if not goals_path.is_file() or not policy_path.is_file() or not nominal_manifest.is_file():
        raise BetrXPError("Nominal BETR-XP-LLM run lacks selected goals, native policy, or manifest.")
    manifest_document = json.loads(nominal_manifest.read_text(encoding="utf-8"))
    if manifest_document.get("method", {}).get("id") != BETR_XP_METHOD_ID:
        raise BetrXPError("--nominal-run is not a BETR-XP-LLM run.")
    goals_document = json.loads(goals_path.read_text(encoding="utf-8"))
    if not isinstance(goals_document, list):
        raise BetrXPError("Nominal BETR-XP-LLM selected goals must be an array.")
    predicates: list[str] = []
    for item in goals_document:
        if isinstance(item, dict) and isinstance(item.get("predicate"), str):
            predicates.append(item["predicate"])
    if len(predicates) != len(goals_document) or not predicates:
        raise BetrXPError("Nominal BETR-XP-LLM selected goals are incomplete.")
    observation = failure_snapshot.get("failure_observation")
    if not isinstance(observation, dict):
        raise BetrXPError("Failure snapshot has no failure_observation object.")
    part = observation.get("object")
    expected_location = observation.get("recovery_location")
    if not isinstance(part, str) or not isinstance(expected_location, str):
        raise BetrXPError("Failure observation lacks object or recovery_location.")

    base = _new_run_directory(Path(output_root), runtime_scenario.task_id, "recovery")
    prompts_dir = base / "prompts"
    native_dir = base / "native"
    prompts_dir.mkdir(parents=True, exist_ok=False)
    native_dir.mkdir(parents=True, exist_ok=False)
    prompt = build_parameter_recovery_prompt(runtime_scenario, failure_snapshot)
    prompt_path = prompts_dir / "parameter_recovery.user.txt"
    snapshot_path = native_dir / "failure_snapshot.json"
    policy_before_path = native_dir / "native_policy_before.bt.json"
    save_text(prompt_path, prompt)
    save_json(snapshot_path, failure_snapshot)
    save_json(policy_before_path, json.loads(policy_path.read_text(encoding="utf-8")))
    artifact_files = [prompt_path, snapshot_path, policy_before_path]
    generation_errors: list[dict[str, Any]] = []
    call_metadata: dict[str, Any] = {}
    response_text = ""
    reasoning = ""
    selected_location: str | None = None
    binding_document: dict[str, Any] | None = None
    selected_plan: PlannedAlternative | None = None
    locations = [entity.id for entity in runtime_scenario.entities if entity.type == "location"]

    try:
        call_started = time.perf_counter()
        stage = caller.generate(prompt, stage="recovery_response")
        response_text = stage.text
        call_metadata = {
            "elapsed_wall_seconds": round(time.perf_counter() - call_started, 4),
            **stage.metadata,
        }
        reasoning, selected_location = parse_parameter_response(stage.text, locations)
        if selected_location != expected_location:
            raise BetrXPNativeError(
                f"The LLM selected '{selected_location}', but the measured object location is "
                f"'{expected_location}'."
            )
        binding = pickup_binding(
            runtime_scenario,
            part=part,
            location=selected_location,
        )
        if binding.name != "recover_fallen_part":
            raise BetrXPNativeError(
                "The resolved pickup parameter did not bind the measured floor-recovery skill."
            )
        binding_document = binding.to_dict()
        grounded_actions = [
            action
            for action in ground_action_templates(runtime_scenario)
            if action.name != "recover_fallen_part"
        ]
        grounded_actions.append(binding)
        selected_plan = plan_predicates(
            runtime_scenario,
            predicates,
            grounded_actions=grounded_actions,
        )
        producers = [
            action["name"]
            for event in selected_plan.expansion.trace
            for action in event.get("producer_templates", [])
        ]
        if "recover_fallen_part" not in producers:
            raise BetrXPNativeError(
                "Reactive replanning did not propagate the resolved floor location into recovery."
            )
        if selected_plan.expansion.unresolved:
            generation_errors.extend(
                {"type": "unresolved_recovery_condition", **item}
                for item in selected_plan.expansion.unresolved
            )
    except Exception as error:
        generation_errors.append(
            {"type": "parameter_resolution_or_replanning_error", "message": str(error)}
        )

    response_path = native_dir / "recovery_response.txt"
    update_path = native_dir / "parameter_update.json"
    goals_output = native_dir / "nominal_selected_goals.json"
    trace_path = native_dir / "planner_trace.json"
    policy_after_path = native_dir / "native_policy_after.bt.json"
    expanded_path = native_dir / "expanded_forest.json"
    save_text(response_path, response_text)
    save_json(goals_output, goals_document)
    if selected_plan is None:
        behavior_trees: dict[str, Any] = {}
        policy_after: dict[str, Any] = {}
        trace_document: dict[str, Any] = {"events": [], "unresolved": []}
    else:
        behavior_trees = {
            robot: tree.to_dict()
            for robot, tree in selected_plan.expansion.trees.items()
        }
        policy_after = native_flat_policy(selected_plan.expansion.trees)
        trace_document = {
            "events": selected_plan.expansion.trace,
            "unresolved": selected_plan.expansion.unresolved,
        }
    update_document = {
        "mode": "missing_parameter_resolution",
        "failing_native_skill": {
            "name": "Pick",
            "parameters": {"part": part, "location": "source_cradle"},
        },
        "error_message": (
            f"pickup location invalid after {part} fell to {expected_location}"
        ),
        "reasoning": reasoning,
        "resolved_parameter": {
            "name": "location",
            "before": "source_cradle",
            "after": selected_location,
        },
        "common_skill_binding": binding_document,
        "permanent_policy_update": selected_plan is not None,
        "llm_output_used_as_planner_input": selected_plan is not None,
    }
    save_json(update_path, update_document)
    save_json(trace_path, trace_document)
    save_json(policy_after_path, policy_after)
    save_json(expanded_path, behavior_trees)
    artifact_files.extend(
        [response_path, goals_output, update_path, trace_path, policy_after_path, expanded_path]
    )
    plan_document = {
        "schema_version": "2.0",
        "mission_id": runtime_scenario.task_id,
        "behavior_trees": behavior_trees,
    }
    return _finish_run(
        base,
        runtime_scenario,
        plan_document,
        artifact_files,
        generation_errors,
        call_metadata,
        caller,
        max_ticks=max_ticks,
        wall_seconds=time.perf_counter() - started,
        invocation=invocation,
        track="recovery",
        nominal_run=nominal_directory,
        recovery_details={
            "failure_resolver_invoked_after_failure": bool(call_metadata),
            "real_llm_recalled_after_failure": (
                caller.real_model_inference and bool(call_metadata)
            ),
            "failure_resolution_branch": "missing_parameter",
            "failed_skill": "Pick(primary_part, source_cradle)",
            "resolved_parameter": selected_location,
            "expected_measured_parameter": expected_location,
            "nominal_manifest_sha256": _sha256_file(nominal_manifest),
            "same_nominal_goals_reused": True,
            "same_object_recovery": part == "primary_part",
        },
    )


def _finish_run(
    base: Path,
    scenario: Scenario,
    plan_document: dict[str, Any],
    artifact_files: list[Path],
    generation_errors: list[dict[str, Any]],
    call_metadata: dict[str, Any],
    caller: BetrXPCaller,
    *,
    max_ticks: int,
    wall_seconds: float,
    invocation: list[str] | None,
    track: str,
    nominal_run: Path | None,
    recovery_details: dict[str, Any] | None,
) -> BetrXPRun:
    plan = parse_plan(plan_document)
    validation = validate_plan(plan, scenario, suggest_producers=False)
    simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
    plan_generation_success = not generation_errors and set(plan.behavior_trees) == scenario.robot_ids
    canonical_path = base / "canonical_plan.json"
    accepted_path = base / "accepted_plan.json" if validation.valid and simulation.success else None
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
        caller,
        call_metadata,
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
            "id": BETR_XP_METHOD_ID,
            "name": "BETR-XP-LLM",
            "paper": BETR_XP_PAPER_URL,
            "doi": BETR_XP_DOI,
            "official_repository": BETR_XP_REPOSITORY_URL,
            "official_repository_commit": BETR_XP_REPOSITORY_COMMIT,
            "official_executable_code_found": True,
            "software_license": "BSD-3-Clause",
            "implementation": "official-source-informed common-domain compatibility runner",
            "paper_model": PAPER_MODEL,
            "selected_model": caller.model,
            "track": track,
        },
        "caller": {
            "provider": caller.provider,
            "model": caller.model,
            "real_model_inference": caller.real_model_inference,
            "call": call_metadata or None,
        },
        "invocation": invocation or [],
        "scenario_sha256": _sha256_json(scenario_to_dict(scenario)),
        "canonical_plan_sha256": _sha256_json(plan_document),
        "nominal_run": str(nominal_run) if nominal_run is not None else None,
        "recovery": recovery_details,
        "fidelity": {
            "native_architecture": [
                "one GPT-4-1106 goal-formalization call with no reflective feedback",
                "strict described-condition and scene-object vocabulary",
                "reactive PDDL-style failed-condition backchaining into Behavior Trees",
                "lowest-cost native DNF goal-alternative selection",
                "failure-time LLM resolution of missing preconditions or parameters",
                "resolved knowledge permanently fed back into the planner policy",
            ],
            "paper_inference": {
                "model": PAPER_MODEL,
                "temperature": PAPER_TEMPERATURE,
                "top_p": PAPER_TOP_P,
                "message_roles": ["user"],
            },
            "input_adaptations": [
                "released prompt structure is instantiated with common predicate descriptions and typed entities",
                "underscore-free aliases represent common predicate and entity identifiers in the released grammar",
                "common symbolic facts replace the authors' text/vision scene description",
                "the dropped-object trial uses the published missing-parameter resolution branch for pickup location",
            ],
            "output_adaptations": [
                "the authors' single-YuMi reactive policy is partitioned by declared common robot ownership",
                "cross-robot dependencies and exclusive zones use explicit common WaitFor and resource leaves",
                "native PyTrees/list syntax is archived beside the canonical representation",
                "the post-failure common artifact is the continuation policy planned from the unchanged snapshot",
            ],
            "semantic_rewrites": [],
            "validator_feedback_to_model": False,
            "known_reproduction_limits": [
                "ABB YuMi, RWS, collision-free planning, Azure Kinect, YOLO-World, NanoSAM, and VLM perception are replaced by common evaluators",
                "the official planner's custom py_trees fork and domain behaviors are represented by equivalent common reactive-expansion semantics",
                "capability duration and relaxed causal distance stand in for unavailable task-specific skill costs",
                "the original Azure OpenAI endpoint is replaced by the selected standard OpenAI-compatible endpoint",
                "the common drop resolves an invalid pickup-location parameter; it is not one of the paper's ten precondition scenarios",
            ],
        },
        "results": metrics,
        "files": {_relative(base, path): _sha256_file(path) for path in artifact_files},
    }
    save_json(manifest_path, manifest)
    return BetrXPRun(
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
    caller: BetrXPCaller,
    call_metadata: dict[str, Any],
    generation_errors: list[dict[str, Any]],
    track: str,
) -> dict[str, Any]:
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    common = {
        "track": track,
        "wall_seconds": round(wall_seconds, 4),
        "model_calls": 1 if caller.real_model_inference and call_metadata else 0,
        "archived_response_count": 1 if not caller.real_model_inference and call_metadata else 0,
        "input_tokens": call_metadata.get("input_tokens") if call_metadata else None,
        "output_tokens": call_metadata.get("output_tokens") if call_metadata else None,
        "monetary_cost": None,
        "bt_node_count": len(nodes),
        "action_count": sum(node.type == "Action" for node in nodes),
        "maximum_tree_depth": max((_tree_depth(tree) for tree in plan.behavior_trees.values()), default=0),
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


def _goal_examples(
    scenario: Scenario,
    schemas: list[ConditionSchema],
    entities: list[EntityAlias],
) -> str:
    examples: list[str] = []
    for predicate in scenario.initial_state:
        name, _arguments = _parse(predicate)
        if name in {"system_ready", "at", "docked"}:
            try:
                encoded = encode_predicate(predicate, schemas, entities)
            except BetrXPNativeError:
                continue
            examples.append(f"Instruction: Preserve the observed condition {predicate}.\nGoal: {encoded}")
        if len(examples) == 2:
            break
    return "\n\n".join(examples) or "Instruction: Keep the system ready.\nGoal: SystemReady"


def _parse(predicate: str) -> tuple[str, tuple[str, ...]]:
    from ..predicates import parse_predicate

    name, arguments = parse_predicate(predicate)
    return name, tuple(arguments)


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
    return path.relative_to(root).as_posix()
