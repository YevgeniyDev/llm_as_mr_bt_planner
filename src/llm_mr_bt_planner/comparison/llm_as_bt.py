"""Common-domain reproduction of Ao et al.'s LLM-as-BT-Planner.

The implementation preserves the paper's assembly-planner boundary and four
in-context generation schemes.  Model output remains native KIOS JSON until a
strict, non-repairing observer translates it for the shared evaluation stack.
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
from ..bt import BTNode, iter_nodes
from ..config import save_json, save_text
from ..domain import Scenario, scenario_to_dict
from ..llm import get_client
from ..plan import parse_plan
from ..predicates import canonical_predicate, parse_predicate
from ..simulation import SimulationReport, simulate, skipped_simulation
from ..validation import ValidationReport, validate_plan
from .llm_as_bt_native import (
    KiosNode,
    NativeExecution,
    action_sequence,
    native_forest_to_plan,
    parse_kios_tree,
    simulate_kios_tree,
    validate_unit_subtrees,
)
from .llm_as_bt_prompts import (
    SYSTEM_PROMPT,
    build_decomposition_prompt,
    build_human_tree_prompt,
    build_iterative_prompt,
    build_make_plan_prompt,
    build_make_tree_prompt,
    build_one_step_prompt,
    build_predict_state_prompt,
    build_sequential_plan_prompt,
)
from .llm_as_bt_source import KIOS_REPOSITORY_COMMIT, KIOS_REPOSITORY_URL

LLM_AS_BT_METHOD_ID = "llm-as-bt-planner"
LLM_AS_BT_PAPER_URL = "https://arxiv.org/abs/2409.10444"
PAPER_MODEL = "gpt-4"
SCHEMES = ("one-step", "iterative", "human", "recursive")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class LLMAsBTError(RuntimeError):
    """Raised when a native planning stage is invalid."""


@dataclass(frozen=True)
class StageResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMAsBTGenerator(Protocol):
    provider: str
    model: str
    real_model_inference: bool

    def generate(
        self,
        stage: str,
        system: str,
        user: str,
        context: dict[str, Any],
    ) -> StageResult:
        ...


class ProviderGenerator:
    """JSON-mode provider surface for every paper stage."""

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
            raise LLMAsBTError("Provider must be 'openai' or 'anthropic'.")
        self.provider = provider
        self.model = model
        self._api_key = api_key
        self._seed = seed

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
            "max_tokens": 3000,
        }
        if self.provider == "openai":
            options.update(temperature=0.0, seed=self._seed, json_mode=True)
        client = get_client(self.provider, **options)
        text = client.complete(system, user)
        metadata: dict[str, Any] = {
            "mode": "provider",
            "provider": self.provider,
            "real_model_inference": True,
            "stage": stage,
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
    """Consume ordered archived native responses with stage/context checks."""

    provider = "replay"
    real_model_inference = False

    def __init__(self, responses: list[dict[str, Any]], *, model: str = "archived-kios-responses") -> None:
        if not responses:
            raise LLMAsBTError("KIOS replay requires at least one ordered response.")
        self.model = model
        self._responses = responses
        self._index = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayGenerator":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        responses = document.get("responses") if isinstance(document, dict) else None
        if not isinstance(responses, list) or not all(isinstance(item, dict) for item in responses):
            raise LLMAsBTError("Replay JSON must contain an ordered 'responses' object array.")
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
            raise LLMAsBTError(f"Replay exhausted before stage '{stage}'.")
        item = self._responses[self._index]
        self._index += 1
        if item.get("stage") != stage:
            raise LLMAsBTError(
                f"Replay response {self._index} is for '{item.get('stage')}', expected '{stage}'."
            )
        for key in ("subgoal", "attempt", "depth"):
            if key in item and item[key] != context.get(key):
                raise LLMAsBTError(
                    f"Replay response {self._index} has {key}={item[key]!r}, "
                    f"expected {context.get(key)!r}."
                )
        response = item.get("response")
        if isinstance(response, dict | list):
            text = json.dumps(response)
        elif isinstance(response, str):
            text = response
        else:
            raise LLMAsBTError(f"Replay response {self._index} has no JSON/string 'response'.")
        return StageResult(
            text=text,
            metadata={
                "mode": "replay",
                "provider": self.provider,
                "real_model_inference": False,
                "output_characters": len(text),
            },
        )


@dataclass(frozen=True)
class Subgoal:
    id: str
    robot: str
    target: str
    instruction: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "robot": self.robot,
            "target": self.target,
            "instruction": self.instruction,
        }


@dataclass(frozen=True)
class LLMAsBTRun:
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


@dataclass
class _RunContext:
    generator: LLMAsBTGenerator
    prompt_dir: Path
    response_dir: Path
    calls: list[dict[str, Any]] = field(default_factory=list)
    counter: int = 0

    def call(self, stage: str, prompt: str, context: dict[str, Any]) -> str:
        self.counter += 1
        stem = f"{self.counter:04d}-{_safe_name(stage)}"
        save_text(self.prompt_dir / f"{stem}.txt", prompt)
        started = time.perf_counter()
        try:
            result = self.generator.generate(stage, SYSTEM_PROMPT, prompt, context)
        except Exception as error:
            self.calls.append(
                {
                    "index": self.counter,
                    "stage": stage,
                    "context": context,
                    "elapsed_wall_seconds": round(time.perf_counter() - started, 4),
                    "error": str(error),
                }
            )
            raise
        save_text(self.response_dir / f"{stem}.txt", result.text)
        self.calls.append(
            {
                "index": self.counter,
                "stage": stage,
                "context": context,
                "elapsed_wall_seconds": round(time.perf_counter() - started, 4),
                **result.metadata,
            }
        )
        return result.text


def load_human_feedback(path: str | Path) -> dict[str, list[str]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, str) and item.strip() for item in value)
        for key, value in document.items()
    ):
        raise LLMAsBTError("Human feedback must map subgoal ids to ordered non-empty strings.")
    return {key: [item.strip() for item in value] for key, value in document.items()}


def run_llm_as_bt_planner(
    scenario: Scenario,
    generator: LLMAsBTGenerator,
    output_root: str | Path,
    *,
    scheme: str = "one-step",
    max_iterations: int = 5,
    human_feedback: dict[str, list[str]] | None = None,
    max_recursive_depth: int = 12,
    max_recursive_expansions: int = 80,
    max_ticks: int = 160,
    invocation: list[str] | None = None,
) -> LLMAsBTRun:
    """Run one native ICL scheme and archive native and common evidence."""
    if scheme not in SCHEMES:
        raise LLMAsBTError(f"Unknown scheme '{scheme}'. Choose from {', '.join(SCHEMES)}.")
    if max_iterations <= 0 or max_recursive_depth <= 0 or max_recursive_expansions <= 0:
        raise LLMAsBTError("Iteration, depth, and expansion limits must be positive.")
    if human_feedback and scheme != "human":
        raise LLMAsBTError("Human feedback may be supplied only for the human scheme.")
    started = time.perf_counter()
    base = _new_run_directory(Path(output_root), scenario.task_id, scheme)
    prompt_dir = base / "prompts"
    response_dir = base / "native" / "responses"
    tree_dir = base / "native" / "trees"
    for directory in (prompt_dir, response_dir, tree_dir):
        directory.mkdir(parents=True, exist_ok=False)
    context = _RunContext(generator, prompt_dir, response_dir)
    generation_errors: list[dict[str, Any]] = []
    native_records: list[dict[str, Any]] = []
    forest: list[tuple[str, str, KiosNode]] = []
    world_state = list(scenario.initial_state)

    decomposition_prompt = build_decomposition_prompt(scenario)
    subgoals: list[Subgoal] = []
    try:
        response = context.call("decompose", decomposition_prompt, {})
        subgoals = _parse_decomposition(response, scenario)
    except Exception as error:
        generation_errors.append(
            {"type": "decomposition_error", "stage": "decompose", "message": str(error)}
        )
    save_json(base / "native" / "subgoals.json", {"subgoals": [item.to_dict() for item in subgoals]})

    feedback = human_feedback or {}
    unknown_feedback = sorted(set(feedback) - {item.id for item in subgoals})
    if unknown_feedback and not generation_errors:
        generation_errors.append(
            {
                "type": "unknown_human_feedback_subgoal",
                "stage": "human",
                "message": "Feedback references unknown subgoal(s): " + ", ".join(unknown_feedback),
            }
        )
    for index, subgoal in enumerate(subgoals, start=1):
        if generation_errors:
            break
        record: dict[str, Any] = {
            "index": index,
            "subgoal": subgoal.to_dict(),
            "start_world_state": list(world_state),
        }
        try:
            if scheme == "one-step":
                tree, execution, details = _one_step(context, scenario, subgoal, world_state)
            elif scheme == "iterative":
                tree, execution, details = _iterative(
                    context, scenario, subgoal, world_state, max_iterations=max_iterations
                )
            elif scheme == "human":
                tree, execution, details = _human(
                    context, scenario, subgoal, world_state, feedback.get(subgoal.id, [])
                )
            else:
                tree, execution, details = _recursive(
                    context,
                    scenario,
                    subgoal,
                    world_state,
                    max_depth=max_recursive_depth,
                    max_expansions=max_recursive_expansions,
                )
            record.update(details)
            record["execution"] = execution.to_dict()
            tree_path = tree_dir / f"{index:02d}-{_safe_name(subgoal.id)}.json"
            save_json(tree_path, tree.to_dict())
            record["tree_file"] = tree_path.relative_to(base).as_posix()
            forest.append((subgoal.id, subgoal.robot, tree))
            if execution.success:
                world_state = list(execution.world_state)
        except Exception as error:
            record["error"] = str(error)
            generation_errors.append(
                {
                    "type": "native_generation_error",
                    "stage": scheme,
                    "subgoal": subgoal.id,
                    "message": str(error),
                }
            )
        native_records.append(record)

    if isinstance(generator, ReplayGenerator) and generator.remaining and not generation_errors:
        generation_errors.append(
            {
                "type": "unused_replay_responses",
                "stage": "replay",
                "message": f"Replay contains {generator.remaining} unused ordered response(s).",
            }
        )

    plan_document: dict[str, Any] = {
        "schema_version": "2.0",
        "mission_id": scenario.task_id,
        "behavior_trees": {},
    }
    if forest:
        try:
            plan_document = native_forest_to_plan(
                forest, scenario, wait_timeout_ticks=max(20, max_ticks)
            )
        except Exception as error:
            generation_errors.append(
                {"type": "canonical_observer_error", "stage": "observe", "message": str(error)}
            )
    save_json(base / "native" / "generation_record.json", {"subgoals": native_records})

    plan = parse_plan(plan_document)
    validation = validate_plan(plan, scenario, suggest_producers=False)
    simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
    plan_generation_success = not generation_errors and len(forest) == len(subgoals) and bool(subgoals)
    wall_seconds = time.perf_counter() - started

    canonical_path = base / "canonical_plan.json"
    accepted_path = (
        base / "accepted_plan.json"
        if plan_generation_success and validation.valid and simulation.success
        else None
    )
    validation_path = base / "validation_report.json"
    simulation_path = base / "simulation_trace.json"
    metrics_path = base / "metrics.json"
    manifest_path = base / "manifest.json"
    scenario_path = base / "scenario.json"
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
        generator,
        context.calls,
        generation_errors,
        plan_generation_success,
        wall_seconds,
    )
    save_json(metrics_path, metrics)
    save_json(base / "human_feedback.json", feedback)

    artifact_files = [path for path in base.rglob("*") if path.is_file() and path != manifest_path]
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "id": LLM_AS_BT_METHOD_ID,
            "name": "LLM-as-BT-Planner",
            "paper": LLM_AS_BT_PAPER_URL,
            "publication": "IEEE International Conference on Robotics and Automation (ICRA), 2025",
            "official_repository": KIOS_REPOSITORY_URL,
            "official_repository_commit": KIOS_REPOSITORY_COMMIT,
            "official_executable_code_found": True,
            "implementation": "clean-room common-domain compatibility runner",
            "scheme": scheme,
        },
        "generator": {
            "provider": generator.provider,
            "model": generator.model,
            "real_model_inference": generator.real_model_inference,
            "calls": context.calls,
        },
        "invocation": invocation or [],
        "scenario_sha256": _sha256_json(scenario_to_dict(scenario)),
        "canonical_plan_sha256": _sha256_json(plan_document),
        "fidelity": {
            "paper_architecture": [
                "LLM assembly planner decomposes the mission into sequential subgoals",
                "one of four reported ICL schemes generates native KIOS JSON",
                "native dummy simulation supplies feedback only in the iterative scheme",
                "low-level actions are restricted to the predefined scenario skill library",
            ],
            "validator_feedback_to_model": False,
            "native_output": "KIOS summary/name/children JSON behavior trees",
            "input_adaptations": [
                "the Siemens gearset and single Panda skill library are replaced by the common multi-robot scenario",
                "the assigned robot id is the first grounded action argument",
                "human feedback is supplied as an archived file for repeatable non-interactive experiments",
            ],
            "output_adaptations": [
                "selector maps to canonical Fallback and memoryless sequence maps to ReactiveSequence",
                "cross-robot KIOS preconditions map to bounded WaitFor leaves to retain sequential subgoal dependencies",
                "declared resource acquisition/release wraps the same generated low-level action",
                "globally unique trace and task ids are assigned during observation",
            ],
            "semantic_rewrites": [],
            "known_reproduction_limits": [
                "the common symbolic/MuJoCo evaluators replace KIOS WebSocket skills, Neo4j, and Panda hardware",
                "canonical Fallback retains running-child memory while KIOS selectors are memoryless",
                "the paper reports GPT-4 while the pinned repository's current chains include GPT-4o",
            ],
        },
        "results": metrics,
        "files": {_relative(base, path): _sha256_file(path) for path in artifact_files},
    }
    save_json(manifest_path, manifest)
    return LLMAsBTRun(
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


def _one_step(
    context: _RunContext,
    scenario: Scenario,
    subgoal: Subgoal,
    world_state: list[str],
) -> tuple[KiosNode, NativeExecution, dict[str, Any]]:
    response = context.call(
        "one_step",
        build_one_step_prompt(scenario, subgoal.to_dict(), world_state),
        {"subgoal": subgoal.id},
    )
    tree, actions, thought = _parse_enveloped_tree(response, scenario, subgoal.robot)
    execution = simulate_kios_tree(tree, scenario, subgoal.robot, world_state)
    return tree, execution, {"thought": thought, "action_sequence": actions, "attempts": 1}


def _iterative(
    context: _RunContext,
    scenario: Scenario,
    subgoal: Subgoal,
    world_state: list[str],
    *,
    max_iterations: int,
) -> tuple[KiosNode, NativeExecution, dict[str, Any]]:
    previous_tree: dict[str, Any] | None = None
    previous_result: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    last_tree: KiosNode | None = None
    last_execution: NativeExecution | None = None
    for attempt in range(1, max_iterations + 1):
        response = context.call(
            "iterative",
            build_iterative_prompt(
                scenario, subgoal.to_dict(), world_state, previous_tree, previous_result
            ),
            {"subgoal": subgoal.id, "attempt": attempt},
        )
        tree, actions, thought = _parse_enveloped_tree(response, scenario, subgoal.robot)
        execution = simulate_kios_tree(tree, scenario, subgoal.robot, world_state)
        attempts.append(
            {
                "attempt": attempt,
                "thought": thought,
                "action_sequence": actions,
                "behavior_tree": tree.to_dict(),
                "execution": execution.to_dict(),
            }
        )
        last_tree, last_execution = tree, execution
        if execution.success:
            break
        previous_tree = tree.to_dict()
        previous_result = execution.to_dict()
    assert last_tree is not None and last_execution is not None
    return last_tree, last_execution, {"attempts": attempts, "attempt_count": len(attempts)}


def _human(
    context: _RunContext,
    scenario: Scenario,
    subgoal: Subgoal,
    world_state: list[str],
    feedback: list[str],
) -> tuple[KiosNode, NativeExecution, dict[str, Any]]:
    plan_response = context.call(
        "sequential_plan",
        build_sequential_plan_prompt(scenario, subgoal.to_dict(), world_state),
        {"subgoal": subgoal.id},
    )
    plan_document = _json_object(plan_response, "sequential plan")
    task_plan = _string_list(plan_document.get("task_plan"), "task_plan")
    for action in task_plan:
        _validate_action(action, scenario, subgoal.robot)
    explanation = plan_document.get("explanation", "")
    if not isinstance(explanation, str):
        raise LLMAsBTError("Sequential-plan explanation must be a string.")

    tree: KiosNode | None = None
    rounds: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for round_index in range(len(feedback) + 1):
        suggestion = feedback[round_index - 1] if round_index else None
        stage = "human_tree" if round_index == 0 else "human_refine"
        response = context.call(
            stage,
            build_human_tree_prompt(
                scenario,
                subgoal.to_dict(),
                world_state,
                task_plan,
                previous_tree=previous,
                human_feedback=suggestion,
            ),
            {"subgoal": subgoal.id, "attempt": round_index + 1},
        )
        tree = parse_kios_tree(_json_object(response, stage))
        validate_unit_subtrees(tree)
        observed = action_sequence(tree)
        if observed != task_plan:
            raise LLMAsBTError(
                f"Human-scheme tree actions {observed!r} do not match task_plan {task_plan!r}."
            )
        rounds.append(
            {
                "round": round_index + 1,
                "feedback": suggestion,
                "behavior_tree": tree.to_dict(),
            }
        )
        previous = tree.to_dict()
    assert tree is not None
    execution = simulate_kios_tree(tree, scenario, subgoal.robot, world_state)
    return tree, execution, {
        "explanation": explanation,
        "task_plan": task_plan,
        "human_rounds": rounds,
        "feedback_count": len(feedback),
    }


def _recursive(
    context: _RunContext,
    scenario: Scenario,
    subgoal: Subgoal,
    world_state: list[str],
    *,
    max_depth: int,
    max_expansions: int,
) -> tuple[KiosNode, NativeExecution, dict[str, Any]]:
    expansion_count = 0
    records: list[dict[str, Any]] = []

    def expand(node: KiosNode, state: list[str], depth: int) -> tuple[KiosNode, list[str]]:
        nonlocal expansion_count
        if depth > max_depth:
            raise LLMAsBTError(f"Recursive generation exceeded depth limit {max_depth}.")
        if node.kind == "action":
            return node, state
        if node.kind in {"target", "precondition"}:
            expansion_count += 1
            if expansion_count > max_expansions:
                raise LLMAsBTError(
                    f"Recursive generation exceeded expansion limit {max_expansions}."
                )
            plan_response = context.call(
                "make_plan",
                build_make_plan_prompt(scenario, subgoal.robot, node.body, state),
                {"subgoal": subgoal.id, "depth": depth},
            )
            plan_document = _json_object(plan_response, "MakePlan")
            task_plan = _string_list(plan_document.get("task_plan"), "task_plan")
            for action in task_plan:
                _validate_action(action, scenario, subgoal.robot)
            record: dict[str, Any] = {
                "depth": depth,
                "goal": node.body,
                "start_world_state": list(state),
                "task_plan": task_plan,
            }
            records.append(record)
            if not task_plan:
                record["skipped"] = True
                return node, state
            last_action = task_plan[-1]
            tree_response = context.call(
                "make_tree",
                build_make_tree_prompt(scenario, subgoal.robot, last_action),
                {"subgoal": subgoal.id, "depth": depth},
            )
            unit = parse_kios_tree(_json_object(tree_response, "MakeTree"))
            validate_unit_subtrees(unit)
            unit_actions = action_sequence(unit)
            if unit_actions != [last_action]:
                raise LLMAsBTError(
                    f"MakeTree must encode only final action {last_action!r}, got {unit_actions!r}."
                )
            record["unit_subtree"] = unit.to_dict()
            sequence = unit.children[1]
            current_state = list(state)
            for index, child in enumerate(sequence.children):
                sequence.children[index], _ignored = expand(child, current_state, depth + 1)
            prediction_response = context.call(
                "predict_state",
                build_predict_state_prompt(scenario, subgoal.robot, current_state, task_plan),
                {"subgoal": subgoal.id, "depth": depth},
            )
            prediction = _json_object(prediction_response, "PredictState")
            estimated = _world_state(prediction.get("estimated_world_state"), scenario)
            record["estimated_world_state"] = estimated
            return unit, estimated
        current_state = list(state)
        for index, child in enumerate(node.children):
            node.children[index], current_state = expand(child, current_state, depth + 1)
        return node, current_state

    root = KiosNode(
        kind="target",
        body=subgoal.target,
        summary=f"achieve {subgoal.target}",
    )
    tree, estimated_state = expand(root, list(world_state), 0)
    execution = simulate_kios_tree(tree, scenario, subgoal.robot, world_state)
    return tree, execution, {
        "recursive_expansion_count": expansion_count,
        "recursive_records": records,
        "estimated_final_world_state": estimated_state,
    }


def _parse_decomposition(text: str, scenario: Scenario) -> list[Subgoal]:
    document = _json_object(text, "decomposition")
    raw = document.get("subgoals")
    if not isinstance(raw, list) or not raw:
        raise LLMAsBTError("Decomposition requires a non-empty subgoals array.")
    subgoals: list[Subgoal] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"id", "robot", "target", "instruction"}:
            raise LLMAsBTError(
                f"Subgoal {index} must contain exactly id, robot, target, and instruction."
            )
        if not all(isinstance(item[key], str) and item[key].strip() for key in item):
            raise LLMAsBTError(f"Subgoal {index} fields must be non-empty strings.")
        identifier = item["id"].strip()
        robot = item["robot"].strip()
        target = canonical_predicate(item["target"])
        if identifier in seen:
            raise LLMAsBTError(f"Duplicate subgoal id '{identifier}'.")
        if robot not in scenario.robot_ids:
            raise LLMAsBTError(f"Subgoal '{identifier}' uses unknown robot '{robot}'.")
        _validate_predicate(target, scenario)
        seen.add(identifier)
        subgoals.append(Subgoal(identifier, robot, target, item["instruction"].strip()))
    return subgoals


def _parse_enveloped_tree(
    text: str, scenario: Scenario, robot_id: str
) -> tuple[KiosNode, list[str], str]:
    document = _json_object(text, "KIOS generation")
    required = {"thought", "action_sequence", "behavior_tree"}
    if set(document) != required:
        raise LLMAsBTError(
            "KIOS generation must contain exactly thought, action_sequence, and behavior_tree."
        )
    thought = document["thought"]
    if not isinstance(thought, str):
        raise LLMAsBTError("KIOS thought must be a string.")
    actions = _string_list(document["action_sequence"], "action_sequence")
    for action in actions:
        _validate_action(action, scenario, robot_id)
    tree = parse_kios_tree(document["behavior_tree"])
    validate_unit_subtrees(tree)
    observed = action_sequence(tree)
    if observed != actions:
        raise LLMAsBTError(
            f"KIOS action_sequence {actions!r} does not match tree action leaves {observed!r}."
        )
    return tree, actions, thought


def _validate_action(expression: str, scenario: Scenario, robot_id: str) -> None:
    name, arguments = parse_predicate(expression)
    if not name or not arguments or arguments[0] != robot_id:
        raise LLMAsBTError(
            f"Action '{expression}' must start with assigned robot '{robot_id}'."
        )
    capability = scenario.capability(robot_id, name)
    if capability is None:
        raise LLMAsBTError(f"Robot '{robot_id}' has no capability '{name}'.")
    if len(arguments[1:]) != len(capability.parameters):
        raise LLMAsBTError(
            f"Action '{expression}' expects {len(capability.parameters)} domain arguments."
        )
    unknown = [value for value in arguments[1:] if value not in scenario.constants]
    if unknown:
        raise LLMAsBTError(f"Action '{expression}' uses unknown constants: {', '.join(unknown)}.")
    for value, expected_type in zip(arguments[1:], capability.parameter_types):
        actual_type = scenario.constant_type(value)
        if expected_type and actual_type is not None and actual_type != expected_type:
            raise LLMAsBTError(
                f"Action '{expression}' uses {value!r} of type {actual_type}, expected {expected_type}."
            )


def _validate_predicate(predicate: str, scenario: Scenario) -> None:
    name, arguments = parse_predicate(predicate)
    signatures: dict[str, int] = {}
    literals = [*scenario.initial_state, *scenario.goal_state]
    for robot in scenario.robots:
        for capability in robot.capabilities:
            literals.extend(capability.preconditions)
            literals.extend(capability.effects.add)
    for literal in literals:
        known_name, known_arguments = parse_predicate(literal)
        signatures[known_name] = len(known_arguments)
    if name not in signatures or len(arguments) != signatures[name]:
        raise LLMAsBTError(f"Predicate '{predicate}' has no supported scenario signature.")
    unknown = [value for value in arguments if value not in scenario.constants]
    if unknown:
        raise LLMAsBTError(f"Predicate '{predicate}' uses unknown constants: {', '.join(unknown)}.")


def _world_state(value: Any, scenario: Scenario) -> list[str]:
    facts = _string_list(value, "estimated_world_state")
    normalized = [canonical_predicate(fact) for fact in facts]
    for fact in normalized:
        _validate_predicate(fact, scenario)
    if len(normalized) != len(set(normalized)):
        raise LLMAsBTError("estimated_world_state contains duplicate facts.")
    return normalized


def _json_object(text: str, label: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise LLMAsBTError(f"{label} response is empty.")
    matches = _FENCE.findall(text)
    if len(matches) > 1:
        raise LLMAsBTError(f"{label} response contains multiple JSON fences.")
    payload = matches[0] if matches else text
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as error:
        raise LLMAsBTError(f"{label} is not valid JSON: {error.msg}.") from error
    if not isinstance(document, dict):
        raise LLMAsBTError(f"{label} JSON root must be an object.")
    return document


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise LLMAsBTError(f"{label} must be an array of non-empty strings.")
    return [item.strip() for item in value]


def _metrics_payload(
    plan: Any,
    validation: ValidationReport,
    simulation: SimulationReport,
    generator: LLMAsBTGenerator,
    calls: list[dict[str, Any]],
    generation_errors: list[dict[str, Any]],
    plan_generation_success: bool,
    wall_seconds: float,
) -> dict[str, Any]:
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    return {
        "track": "nominal",
        "plan_generation_success": plan_generation_success,
        "static_validity": validation.valid,
        "symbolic_goal_success": simulation.goal_success,
        "nominal_execution_success": None,
        "wall_seconds": round(wall_seconds, 4),
        "model_calls": len(calls) if generator.real_model_inference else 0,
        "archived_response_count": len(calls) if not generator.real_model_inference else 0,
        "input_tokens": _sum_metadata(calls, "input_tokens") if generator.real_model_inference else None,
        "output_tokens": _sum_metadata(calls, "output_tokens") if generator.real_model_inference else None,
        "monetary_cost": None,
        "bt_node_count": len(nodes),
        "action_count": sum(node.type == "Action" for node in nodes),
        "maximum_tree_depth": max((_tree_depth(tree) for tree in plan.behavior_trees.values()), default=0),
        "validation_error_count": len(validation.errors),
        "simulation_error_count": len(simulation.errors),
        "native_generation_errors": generation_errors,
    }


def _sum_metadata(calls: list[dict[str, Any]], key: str) -> int | None:
    values = [value for call in calls if isinstance((value := call.get(key)), int)]
    return sum(values) if len(values) == len(calls) and calls else None


def _tree_depth(node: BTNode) -> int:
    return 1 + max((_tree_depth(child) for child in node.children), default=0)


def _new_run_directory(output_root: Path, task_id: str, scheme: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / f"{_safe_name(task_id)}-{_safe_name(scheme)}-{stamp}"
    counter = 1
    while base.exists():
        base = output_root / f"{_safe_name(task_id)}-{_safe_name(scheme)}-{stamp}-{counter}"
        counter += 1
    base.mkdir(parents=True, exist_ok=False)
    return base


def _safe_name(value: str) -> str:
    return _SAFE_ID.sub("-", value).strip("-") or "item"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(base: Path, path: Path) -> str:
    return path.relative_to(base).as_posix()
