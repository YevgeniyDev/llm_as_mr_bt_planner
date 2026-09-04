"""LLM-BT reasoning, released-parser, and dynamic-ATL comparison runner."""

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
from ..simulation import SimulationReport, simulate, skipped_simulation
from ..validation import ValidationReport, validate_plan
from .llm_bt_native import (
    AliasEntry,
    ParsedGoal,
    build_alias_catalog,
    expand_initial_trees,
    ground_action_templates,
    map_moves_to_goals,
    semantic_map_xml,
)
from .llm_bt_parser import KeywordParser, ParserResult, ReplayKeywordParser
from .llm_bt_source import (
    LLM_BT_PAPER_URL,
    LLM_BT_REPOSITORY_COMMIT,
    LLM_BT_REPOSITORY_URL,
)

LLM_BT_METHOD_ID = "llm-bt"
PAPER_CHATGPT_MODEL = "not reported"
DEFAULT_REPRODUCTION_MODEL = "gpt-3.5-turbo"
REASONING_SYSTEM_PROMPT = (
    "You are the reasoning module of a robot task system. Convert the instruction and "
    "semantic map into concise descriptive steps. Return exactly one numbered sentence for "
    "each required final condition, copied exactly from the supplied move-phrase catalog. "
    "Choose only a phrase whose postcondition is that final condition; do not select "
    "intermediate effects or multiple producer aliases for the same condition. Do not add "
    "explanations, headings, conditions, XML, or phrases outside that catalog."
)
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


class LLMBTError(RuntimeError):
    """Raised when LLM-BT reasoning, parsing, or expansion cannot run."""


@dataclass(frozen=True)
class StageResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Reasoner(Protocol):
    provider: str
    model: str
    real_model_inference: bool

    def generate(self, system: str, user: str) -> StageResult:
        ...


class ProviderReasoner:
    """Call ChatGPT once, matching the paper's pre-execution reasoning boundary."""

    provider = "openai"
    real_model_inference = True

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        seed: int | None = 42,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._seed = seed

    def generate(self, system: str, user: str) -> StageResult:
        client = get_client(
            "openai",
            model=self.model,
            api_key=self._api_key,
            temperature=0.0,
            seed=self._seed,
            json_mode=False,
            max_tokens=4000,
        )
        text = client.complete(system, user)
        metadata: dict[str, Any] = {
            "mode": "provider",
            "provider": self.provider,
            "real_model_inference": True,
            "model": self.model,
            "temperature": 0.0,
            "seed": self._seed,
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


class ReplayReasoner:
    provider = "replay"
    real_model_inference = False

    def __init__(self, response: str, *, model: str = "archived-chatgpt-response") -> None:
        if not isinstance(response, str) or not response.strip():
            raise LLMBTError("LLM-BT replay reasoning response must be a non-empty string.")
        self.model = model
        self._response = response

    def generate(self, system: str, user: str) -> StageResult:  # noqa: ARG002
        return StageResult(
            text=self._response,
            metadata={
                "mode": "replay",
                "provider": self.provider,
                "real_model_inference": False,
                "output_characters": len(self._response),
            },
        )


@dataclass(frozen=True)
class LLMBTRun:
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


def load_replay_bundle(path: str | Path) -> tuple[ReplayReasoner, ReplayKeywordParser]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise LLMBTError("LLM-BT replay root must be an object.")
    response = document.get("reasoning_response")
    predictions = document.get("ner_predictions")
    if not isinstance(response, str) or not isinstance(predictions, list) or not all(
        isinstance(item, dict) for item in predictions
    ):
        raise LLMBTError(
            "LLM-BT replay requires 'reasoning_response' and an 'ner_predictions' object array."
        )
    return (
        ReplayReasoner(response, model=f"replay:{Path(path).name}"),
        ReplayKeywordParser(predictions, model=f"replay:{Path(path).name}:ner"),
    )


def build_reasoning_prompt(
    scenario: Scenario,
    semantic_map: str,
    catalog: list[AliasEntry],
) -> str:
    """Build the common semantic-map input using the released parser's move grammar."""
    entries = "\n".join(
        f"- {entry.phrase} => robot={entry.robot}; postcondition={entry.predicate}; "
        f"ATL action={entry.action}({', '.join(entry.parameters)})"
        for entry in catalog
    )
    goals = "\n".join(f"- {goal}" for goal in scenario.goal_state)
    return "\n".join(
        [
            "User instruction:",
            scenario.instruction,
            "",
            "Current semantic map:",
            semantic_map,
            "",
            "Required final conditions:",
            goals,
            "",
            "Released-parser-compatible move-phrase catalog:",
            entries,
            "",
            f"Return exactly {len(scenario.goal_state)} phrases: one for each required final condition, ",
            "with no intermediate-effect phrases and no duplicate postconditions. ",
            "Order them so that a condition which may be consumed is restored after the consuming step. ",
            "The BT Update module, not you, will select actions needed to establish those conditions.",
        ]
    )


def run_llm_bt(
    scenario: Scenario,
    reasoner: Reasoner,
    keyword_parser: KeywordParser,
    output_root: str | Path,
    *,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> LLMBTRun:
    """Run one ChatGPT reasoning call, released BERT parsing, and deterministic BT expansion."""
    started = time.perf_counter()
    base = _new_run_directory(Path(output_root), scenario.task_id, "nominal")
    prompts_dir = base / "prompts"
    native_dir = base / "native"
    prompts_dir.mkdir(parents=True, exist_ok=False)
    native_dir.mkdir(parents=True, exist_ok=False)
    catalog = build_alias_catalog(scenario)
    semantic_map = semantic_map_xml(scenario)
    prompt = build_reasoning_prompt(scenario, semantic_map, catalog)
    artifact_files: list[Path] = []
    generation_errors: list[dict[str, Any]] = []
    call_metadata: dict[str, Any] = {}
    parser_metadata: dict[str, Any] = {}
    parser_result: ParserResult | None = None
    parsed_goals: list[ParsedGoal] = []
    expansion = None

    for path, content in (
        (prompts_dir / "reasoning.system.txt", REASONING_SYSTEM_PROMPT),
        (prompts_dir / "reasoning.user.txt", prompt),
        (native_dir / "semantic_map.xml", semantic_map),
    ):
        save_text(path, content)
        artifact_files.append(path)
    catalog_path = native_dir / "alias_catalog.json"
    atl_path = native_dir / "action_template_library.json"
    save_json(catalog_path, [entry.to_dict() for entry in catalog])
    save_json(atl_path, [action.to_dict() for action in ground_action_templates(scenario)])
    artifact_files.extend([catalog_path, atl_path])

    try:
        call_started = time.perf_counter()
        stage = reasoner.generate(REASONING_SYSTEM_PROMPT, prompt)
        call_metadata = {
            "elapsed_wall_seconds": round(time.perf_counter() - call_started, 4),
            **stage.metadata,
        }
        response_path = native_dir / "reasoning_response.txt"
        save_text(response_path, stage.text)
        artifact_files.append(response_path)
        parser_started = time.perf_counter()
        parser_result = keyword_parser.parse(stage.text)
        parser_metadata = {
            "elapsed_wall_seconds": round(time.perf_counter() - parser_started, 4),
            **parser_result.metadata,
        }
        predictions_path = native_dir / "ner_predictions.json"
        moves_path = native_dir / "parsed_moves.json"
        save_json(predictions_path, parser_result.predictions)
        save_json(moves_path, [move.to_dict() for move in parser_result.moves])
        artifact_files.extend([predictions_path, moves_path])
        parsed_goals = map_moves_to_goals(parser_result.moves, catalog)
        expansion = expand_initial_trees(scenario, parsed_goals)
    except Exception as error:
        generation_errors.append(
            {"type": "reasoning_parser_or_expansion_error", "message": str(error)}
        )

    parsed_goals_path = native_dir / "parsed_goals.json"
    initial_path = native_dir / "initial_forest.json"
    assigned_path = native_dir / "assigned_goals.json"
    expansion_path = native_dir / "expansion_trace.json"
    expanded_path = native_dir / "expanded_forest.json"
    save_json(parsed_goals_path, [goal.to_dict() for goal in parsed_goals])
    if expansion is None:
        initial_document: dict[str, Any] = {}
        assigned_document: list[dict[str, Any]] = []
        expansion_document: dict[str, Any] = {"events": [], "unresolved": []}
        behavior_trees: dict[str, Any] = {}
    else:
        initial_document = {
            robot: tree.to_dict() for robot, tree in expansion.initial_trees.items()
        }
        assigned_document = [goal.to_dict() for goal in expansion.assigned_goals]
        expansion_document = {
            "events": expansion.trace,
            "unresolved": expansion.unresolved,
            "relaxed_reachable": list(expansion.relaxed_reachable),
        }
        behavior_trees = {robot: tree.to_dict() for robot, tree in expansion.trees.items()}
    save_json(initial_path, initial_document)
    save_json(assigned_path, assigned_document)
    save_json(expansion_path, expansion_document)
    save_json(expanded_path, behavior_trees)
    artifact_files.extend(
        [parsed_goals_path, initial_path, assigned_path, expansion_path, expanded_path]
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
        parser_metadata,
        reasoner,
        keyword_parser,
        max_ticks=max_ticks,
        wall_seconds=time.perf_counter() - started,
        invocation=invocation,
        track="nominal",
        nominal_run=None,
        recovery_details=None,
    )


def run_llm_bt_recovery(
    runtime_scenario: Scenario,
    nominal_run: str | Path,
    failure_snapshot: dict[str, Any],
    output_root: str | Path,
    *,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> LLMBTRun:
    """Re-expand nominal parsed goals from a post-failure snapshot without recalling the LLM."""
    started = time.perf_counter()
    nominal_directory = Path(nominal_run).resolve()
    goals_path = nominal_directory / "native" / "parsed_goals.json"
    nominal_manifest = nominal_directory / "manifest.json"
    if not goals_path.is_file() or not nominal_manifest.is_file():
        raise LLMBTError("Nominal LLM-BT run lacks parsed_goals.json or manifest.json.")
    goals_document = json.loads(goals_path.read_text(encoding="utf-8"))
    if not isinstance(goals_document, list):
        raise LLMBTError("Nominal LLM-BT parsed goals must be an array.")
    goals = [_parsed_goal_from_dict(item, index) for index, item in enumerate(goals_document)]
    expansion = expand_initial_trees(runtime_scenario, goals)
    base = _new_run_directory(Path(output_root), runtime_scenario.task_id, "recovery")
    native_dir = base / "native"
    native_dir.mkdir(parents=True, exist_ok=False)
    artifact_files: list[Path] = []
    snapshot_path = native_dir / "failure_snapshot.json"
    goals_output = native_dir / "nominal_parsed_goals.json"
    assigned_path = native_dir / "assigned_goals.json"
    expansion_path = native_dir / "expansion_trace.json"
    expanded_path = native_dir / "expanded_forest.json"
    save_json(snapshot_path, failure_snapshot)
    save_json(goals_output, [goal.to_dict() for goal in goals])
    save_json(assigned_path, [goal.to_dict() for goal in expansion.assigned_goals])
    save_json(
        expansion_path,
        {
            "events": expansion.trace,
            "unresolved": expansion.unresolved,
            "relaxed_reachable": list(expansion.relaxed_reachable),
        },
    )
    behavior_trees = {robot: tree.to_dict() for robot, tree in expansion.trees.items()}
    save_json(expanded_path, behavior_trees)
    artifact_files.extend(
        [snapshot_path, goals_output, assigned_path, expansion_path, expanded_path]
    )
    plan_document = {
        "schema_version": "2.0",
        "mission_id": runtime_scenario.task_id,
        "behavior_trees": behavior_trees,
    }
    errors = [
        {"type": "unresolved_recovery_condition", **item}
        for item in expansion.unresolved
    ]
    return _finish_run(
        base,
        runtime_scenario,
        plan_document,
        artifact_files,
        errors,
        {},
        {},
        ReplayReasoner("not called during LLM-BT recovery", model="not-recalled"),
        ReplayKeywordParser(
            [{"entity": "B-Action", "word": "move"}],
            model="not-recalled",
        ),
        max_ticks=max_ticks,
        wall_seconds=time.perf_counter() - started,
        invocation=invocation,
        track="recovery",
        nominal_run=nominal_directory,
        recovery_details={
            "llm_recalled_after_failure": False,
            "parser_recalled_after_failure": False,
            "nominal_manifest_sha256": _sha256_file(nominal_manifest),
            "same_parsed_goals_reused": True,
            "runtime_atl_capability_count": sum(
                len(robot.capabilities) for robot in runtime_scenario.robots
            ),
        },
    )


def _finish_run(
    base: Path,
    scenario: Scenario,
    plan_document: dict[str, Any],
    artifact_files: list[Path],
    generation_errors: list[dict[str, Any]],
    call_metadata: dict[str, Any],
    parser_metadata: dict[str, Any],
    reasoner: Reasoner,
    keyword_parser: KeywordParser,
    *,
    max_ticks: int,
    wall_seconds: float,
    invocation: list[str] | None,
    track: str,
    nominal_run: Path | None,
    recovery_details: dict[str, Any] | None,
) -> LLMBTRun:
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
        reasoner,
        keyword_parser,
        call_metadata,
        parser_metadata,
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
            "id": LLM_BT_METHOD_ID,
            "name": "LLM-BT",
            "paper": LLM_BT_PAPER_URL,
            "official_repository": LLM_BT_REPOSITORY_URL,
            "official_repository_commit": LLM_BT_REPOSITORY_COMMIT,
            "official_executable_code_found": True,
            "implementation": "paper/source-faithful common-domain compatibility runner",
            "paper_chatgpt_model": PAPER_CHATGPT_MODEL,
            "selected_reasoning_model": reasoner.model,
            "track": track,
        },
        "reasoner": {
            "provider": reasoner.provider,
            "model": reasoner.model,
            "real_model_inference": reasoner.real_model_inference,
            "call": call_metadata or None,
        },
        "keyword_parser": {
            "model": keyword_parser.model,
            "real_model_inference": keyword_parser.real_model_inference,
            "call": parser_metadata or None,
        },
        "invocation": invocation or [],
        "scenario_sha256": _sha256_json(scenario_to_dict(scenario)),
        "canonical_plan_sha256": _sha256_json(plan_document),
        "nominal_run": str(nominal_run) if nominal_run is not None else None,
        "recovery": recovery_details,
        "fidelity": {
            "native_architecture": [
                "one ChatGPT descriptive-step reasoning call before nominal execution",
                "released DistilBERT NER parser with the original eight-label vocabulary",
                "initial sequence of postcondition goal nodes",
                "deterministic failed-condition expansion through a manually supplied Action Template Library",
                "Fallback(condition, Sequence(preconditions, action)) expansion structure",
                "no LLM or BERT call during runtime recovery",
            ],
            "input_adaptations": [
                "symbolic common state is serialized as the paper's XML semantic-map medium",
                "protocol goals receive first-choice move/object/position aliases on a compact numeric grid compatible with the released parser",
                "remaining unique grounded postconditions receive secondary parser-compatible aliases",
                "common capability contracts instantiate the manually supplied Action Template Library",
                "numbered LLM instructions are classified independently to prevent cross-instruction BIO-label bleed",
            ],
            "output_adaptations": [
                "native tick-wise failed-node updates are materialized as a state-aware causal ATL fixpoint before common static validation",
                "single-robot ATL dependencies are assigned to the declared common-domain robot owner",
                "an external producer dependency becomes an explicit bounded consumer WaitFor",
                "resource acquire/release leaves wrap resource-bearing common capability templates",
                "an idle ready-condition tree represents a team member with no remaining assigned goal",
                "the paper's central-tree type-conflict priority move is recorded but not applied across partitioned per-robot trees",
                "common memory sequences replace the released core's reticked sequences so completed multi-robot handoffs are not re-executed",
            ],
            "semantic_rewrites": [],
            "validator_feedback_to_model": False,
            "known_reproduction_limits": [
                "the paper and source do not release the ChatGPT prompt, model version, or decoding settings",
                "the released parser supports only the move/target/destination grammar from its original domains",
                "the project repository declares no project-wide software or parser-model license",
                "V-REP 3.6.2, the original scene, Qt editor, and physical perception stack are replaced by common evaluators",
                "when several equally short ATL actions can establish one condition, the pinned source-style stable first match is used",
            ],
        },
        "results": metrics,
        "files": {_relative(base, path): _sha256_file(path) for path in artifact_files},
    }
    save_json(manifest_path, manifest)
    return LLMBTRun(
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
    reasoner: Reasoner,
    keyword_parser: KeywordParser,
    call_metadata: dict[str, Any],
    parser_metadata: dict[str, Any],
    generation_errors: list[dict[str, Any]],
    track: str,
) -> dict[str, Any]:
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    common = {
        "track": track,
        "wall_seconds": round(wall_seconds, 4),
        "model_calls": 1 if reasoner.real_model_inference and call_metadata else 0,
        "archived_response_count": 1 if not reasoner.real_model_inference and call_metadata else 0,
        "parser_inference_count": 1 if keyword_parser.real_model_inference and parser_metadata else 0,
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


def _parsed_goal_from_dict(document: Any, index: int) -> ParsedGoal:
    if not isinstance(document, dict):
        raise LLMBTError(f"Nominal parsed goal {index} must be an object.")
    alias = document.get("alias")
    if not isinstance(alias, dict):
        raise LLMBTError(f"Nominal parsed goal {index} has no alias object.")
    order = document.get("order")
    robot = document.get("robot")
    predicate = document.get("predicate")
    target = alias.get("target")
    destination = alias.get("destination")
    if not isinstance(order, int) or not all(
        isinstance(value, str) for value in (robot, predicate, target, destination)
    ):
        raise LLMBTError(f"Nominal parsed goal {index} has invalid fields.")
    assert isinstance(robot, str)
    assert isinstance(predicate, str)
    assert isinstance(target, str)
    assert isinstance(destination, str)
    return ParsedGoal(order, robot, predicate, target, destination)


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


def _relative(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix()
