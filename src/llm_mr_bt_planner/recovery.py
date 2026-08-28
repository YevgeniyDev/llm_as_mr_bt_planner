"""Failure-aware Behavior Tree adaptation with auditable LLM provenance."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

from .artifacts import load_plan_file
from .bt import iter_nodes
from .domain import Capability, Effects, Entity, Scenario
from .llm.base import LLMError
from .llm.openai_client import _send
from .plan import Plan, parse_plan
from .predicates import canonical_predicate
from .prompts import build_prompt
from .simulation import SimulationReport, simulate
from .validation import ValidationReport, validate_plan

DEFAULT_RECOVERY_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "high"
RecoveryProgress = Callable[[str, float], None]

RECOVERY_SYSTEM_PROMPT = (
    "You are the online recovery planner for a heterogeneous robot team. A nominal Behavior "
    "Tree has failed in a live physics state. Return a complete continuation Behavior Tree for "
    "every robot. Preserve safety and resource ownership, use only declared capabilities, avoid "
    "inventing objects or observations, use only Sequence and Fallback composite nodes, and "
    "reach every mission goal from the measured post-failure state. If the observation says the "
    "fallen object remains usable, recover that same object with the newly reported recovery "
    "capability. The returned JSON is executed without resetting the simulator."
)


class RecoveryClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return a plan document and non-secret request/response provenance."""


@dataclass(frozen=True)
class RecoveryPlanningResult:
    plan: Plan
    runtime_scenario: Scenario
    validation: ValidationReport
    simulation: SimulationReport
    attempts: tuple[dict[str, Any], ...]
    provider: str
    model: str
    reasoning_effort: str | None


class OpenAIResponsesRecoveryClient:
    """Responses API client dedicated to strict recovery-plan JSON."""

    provider = "openai"

    def __init__(
        self,
        *,
        model: str = DEFAULT_RECOVERY_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self._timeout = (
            timeout
            if timeout is not None
            else float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "180"))
        )

    def complete(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self._api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Set it before running the real LLM recovery experiment."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "reasoning": {"effort": self.reasoning_effort},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "multi_robot_recovery_bt",
                    "strict": True,
                    "schema": recovery_plan_json_schema(),
                }
            },
            "max_output_tokens": 12000,
            "store": False,
        }
        request = urllib.request.Request(
            self._responses_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        response = json.loads(_send(request, self._timeout))
        elapsed = time.perf_counter() - started
        output_text = _responses_output_text(response)
        document = json.loads(output_text)
        if not isinstance(document, dict):
            raise LLMError("OpenAI recovery response did not contain an object-valued plan.")
        provenance = {
            "provider": self.provider,
            "model_requested": self.model,
            "model_returned": response.get("model"),
            "response_id": response.get("id"),
            "status": response.get("status"),
            "reasoning_effort": self.reasoning_effort,
            "elapsed_wall_seconds": round(elapsed, 4),
            "usage": response.get("usage"),
            "request": {
                "endpoint": self._responses_url(),
                "model": self.model,
                "reasoning": {"effort": self.reasoning_effort},
                "structured_output": "multi_robot_recovery_bt",
                "max_output_tokens": payload["max_output_tokens"],
                "store": False,
            },
            "output_text": output_text,
        }
        return document, provenance

    def _responses_url(self) -> str:
        explicit = os.environ.get("OPENAI_RESPONSES_URL")
        if explicit:
            return explicit
        if self._base_url.endswith("/responses"):
            return self._base_url
        return f"{self._base_url}/responses"


class OracleRecoveryClient:
    """Explicitly non-LLM fixture for offline tests and deterministic dry runs."""

    provider = "deterministic_oracle"
    model = "none"
    reasoning_effort = None

    def __init__(self, plan_path: str | Path) -> None:
        self.plan_path = Path(plan_path).resolve()

    def complete(self, system: str, user: str) -> tuple[dict[str, Any], dict[str, Any]]:
        plan = load_plan_file(self.plan_path)
        return plan.to_dict(), {
            "provider": self.provider,
            "model_requested": self.model,
            "reasoning_effort": None,
            "fixture": str(self.plan_path),
            "warning": "Offline deterministic test oracle; this is not evidence of LLM adaptation.",
            "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
        }


def plan_recovery(
    client: RecoveryClient,
    scenario: Scenario,
    *,
    measured_initial_state: tuple[str, ...],
    failure_observation: dict[str, Any],
    nominal_plan: Plan,
    max_corrections: int = 2,
    max_ticks: int = 160,
    progress: RecoveryProgress | None = None,
) -> RecoveryPlanningResult:
    """Generate, validate, and contract-simulate a same-state continuation BT."""
    if max_corrections < 0:
        raise ValueError("max_corrections cannot be negative.")
    runtime_scenario = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=measured_initial_state,
        failure_observation=failure_observation,
    )
    _recovery_progress(progress, "Prepared the measured post-failure planning state", 0.03)
    attempts: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] | None = None
    for round_index in range(max_corrections + 1):
        round_label = (
            "Initial recovery candidate"
            if round_index == 0
            else f"Recovery correction {round_index}/{max_corrections}"
        )
        round_start = round_index / (max_corrections + 1)
        round_span = 1.0 / (max_corrections + 1)
        prompt = build_recovery_prompt(
            runtime_scenario,
            failure_observation=failure_observation,
            nominal_plan=nominal_plan,
            previous_diagnostics=diagnostics,
        )
        _recovery_progress(
            progress,
            f"{round_label}: sending failure snapshot and failed BT to {client.provider}/{client.model}",
            0.05 + 0.75 * round_start,
        )
        document, provenance = client.complete(RECOVERY_SYSTEM_PROMPT, prompt)
        _recovery_progress(
            progress,
            f"{round_label}: provider response received; parsing the complete replacement BT",
            0.05 + 0.75 * (round_start + round_span * 0.42),
        )
        plan = parse_plan(document)
        _recovery_progress(
            progress,
            f"{round_label}: running independent static validation",
            0.05 + 0.75 * (round_start + round_span * 0.60),
        )
        validation = validate_plan(plan, runtime_scenario, suggest_producers=True)
        _recovery_progress(
            progress,
            f"{round_label}: running deterministic continuation simulation",
            0.05 + 0.75 * (round_start + round_span * 0.76),
        )
        simulation = (
            simulate(plan, runtime_scenario, max_ticks=max_ticks)
            if validation.valid
            else SimulationReport(
                success=False,
                goal_success=False,
                final_state=list(runtime_scenario.initial_state),
                trace=[],
                errors=[{"type": "static_validation_failed"}],
            )
        )
        semantic_errors = _recovery_semantic_errors(plan)
        accepted = validation.valid and simulation.success and not semantic_errors
        attempts.append(
            {
                "round": round_index,
                "accepted": accepted,
                "prompt": {
                    "system": RECOVERY_SYSTEM_PROMPT,
                    "user": prompt,
                },
                "provenance": provenance,
                "candidate": document,
                "validation": {
                    "valid": validation.valid,
                    "errors": validation.to_dicts(),
                },
                "contract_simulation": simulation.to_dict(),
                "recovery_semantic_errors": semantic_errors,
            }
        )
        if accepted:
            _recovery_progress(
                progress,
                f"{round_label}: accepted; validation, simulation, and fallen-object recovery checks passed",
                0.92,
            )
            _recovery_progress(progress, "Validated continuation BT is ready for MuJoCo", 1.0)
            return RecoveryPlanningResult(
                plan=plan,
                runtime_scenario=runtime_scenario,
                validation=validation,
                simulation=simulation,
                attempts=tuple(attempts),
                provider=client.provider,
                model=client.model,
                reasoning_effort=getattr(client, "reasoning_effort", None),
            )
        reasons = [
            *(error.type for error in validation.errors),
            *(str(error.get("type", "simulation_error")) for error in simulation.errors),
            *semantic_errors,
        ]
        _recovery_progress(
            progress,
            f"{round_label}: rejected ({', '.join(sorted(set(reasons))) or 'goals not reached'})",
            0.05 + 0.75 * (round_start + round_span),
        )
        diagnostics = {
            "validation_errors": validation.to_dicts(),
            "simulation_errors": simulation.errors,
            "simulation_final_state": simulation.final_state,
            "recovery_semantic_errors": semantic_errors,
            "rejected_candidate": document,
        }
    raise RuntimeError(
        f"Recovery planner failed validation/simulation after {max_corrections + 1} attempt(s)."
    )


def _recovery_progress(
    callback: RecoveryProgress | None,
    message: str,
    fraction: float,
) -> None:
    if callback is not None:
        callback(message, max(0.0, min(1.0, fraction)))


def build_recovery_prompt(
    runtime_scenario: Scenario,
    *,
    failure_observation: dict[str, Any],
    nominal_plan: Plan,
    previous_diagnostics: dict[str, Any] | None = None,
) -> str:
    sections = [
        "Adapt the failed mission by returning a complete continuation BT for all three robots.",
        "Execution will resume in the exact current MuJoCo model/data state; no reset is allowed.",
        (
            "The only task object is primary_part. It fell to source_floor, was measured there, "
            "and remains usable. Have unitree_go2_z1 execute "
            "recover_fallen_part(primary_part,source_floor), then continue transport and installation. "
            "Do not invent a spare or replacement object."
        ),
        f"Failure observation:\n{json.dumps(failure_observation, indent=2)}",
        f"Failed nominal BT:\n{json.dumps(nominal_plan.to_dict(), indent=2)}",
        build_prompt(runtime_scenario),
    ]
    if previous_diagnostics is not None:
        sections.append(
            "The previous recovery candidate was rejected. Return a complete corrected replacement.\n"
            f"Diagnostics:\n{json.dumps(previous_diagnostics, indent=2)}"
        )
    return "\n\n".join(sections)


def plan_diff(nominal: Plan, adapted: Plan) -> str:
    before = json.dumps(nominal.to_dict(), indent=2, sort_keys=True).splitlines()
    after = json.dumps(adapted.to_dict(), indent=2, sort_keys=True).splitlines()
    return "\n".join(
        difflib.unified_diff(
            before,
            after,
            fromfile="nominal_behavior_tree.json",
            tofile="adapted_behavior_tree.json",
            lineterm="",
        )
    ) + "\n"


def recovery_plan_json_schema() -> dict[str, Any]:
    """Strict structured-output schema for the three declared recovery robots."""
    source = {"type": "string", "const": "llm"}
    common = {
        "id": {"type": "string", "minLength": 1},
        "source": source,
    }

    def leaf(node_type: str, *, task_id: bool = False, timeout: bool = False) -> dict[str, Any]:
        properties: dict[str, Any] = {
            **common,
            "type": {"type": "string", "const": node_type},
            "name": {"type": "string", "minLength": 1},
            "parameters": {"type": "array", "items": {"type": "string"}},
        }
        required = ["id", "type", "source", "name", "parameters"]
        if task_id:
            properties["task_id"] = {"type": "string", "minLength": 1}
            required.append("task_id")
        if timeout:
            properties["timeout_ticks"] = {"type": "integer", "minimum": 1}
            required.append("timeout_ticks")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    sequential = {
        "type": "object",
        "properties": {
            **common,
            "type": {
                "type": "string",
                "enum": ["Sequence", "ReactiveSequence", "Fallback"],
            },
            "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "minItems": 1},
        },
        "required": ["id", "type", "source", "children"],
        "additionalProperties": False,
    }
    parallel = {
        "type": "object",
        "properties": {
            **common,
            "type": {"type": "string", "enum": ["Parallel", "ParallelAll"]},
            "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "minItems": 1},
            "success_threshold": {"type": "integer", "minimum": 1},
        },
        "required": ["id", "type", "source", "children", "success_threshold"],
        "additionalProperties": False,
    }
    node = {
        "anyOf": [
            sequential,
            parallel,
            leaf("Action", task_id=True),
            leaf("Condition"),
            leaf("WaitFor", timeout=True),
            leaf("AcquireResource", timeout=True),
            leaf("ReleaseResource"),
        ]
    }
    robots = ("franka_a", "unitree_go2_z1", "franka_b")
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "const": "2.0"},
            "mission_id": {
                "type": "string",
                "const": "three_robot_component_installation",
            },
            "behavior_trees": {
                "type": "object",
                "properties": {robot: {"$ref": "#/$defs/node"} for robot in robots},
                "required": list(robots),
                "additionalProperties": False,
            },
        },
        "required": ["schema_version", "mission_id", "behavior_trees"],
        "additionalProperties": False,
        "$defs": {"node": node},
    }


def _responses_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    for item in response.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    raise LLMError("OpenAI Responses API result contained no output_text content.")


def _recovery_semantic_errors(plan: Plan) -> list[str]:
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    action_parameters = [node.parameters for node in nodes if node.type == "Action"]
    recovery_actions = [
        (robot, node)
        for robot, tree in plan.behavior_trees.items()
        for node in iter_nodes(tree)
        if node.type == "Action" and node.name == "recover_fallen_part"
    ]
    errors: list[str] = []
    if not any("primary_part" in parameters for parameters in action_parameters):
        errors.append("Recovery BT does not operate on the fallen primary_part.")
    if any("spare_part" in parameters for parameters in action_parameters):
        errors.append("Recovery BT invents an undeclared spare_part.")
    if not any(
        robot == "unitree_go2_z1"
        and node.parameters == ("primary_part", "source_floor")
        for robot, node in recovery_actions
    ):
        errors.append(
            "Recovery BT must have unitree_go2_z1 execute "
            "recover_fallen_part(primary_part,source_floor)."
        )
    unsupported_composites = sorted(
        {node.type for node in nodes if node.children and node.type not in {"Sequence", "Fallback"}}
    )
    if unsupported_composites:
        errors.append(
            "Recovery BT uses composites outside the MuJoCo adapter scope: "
            + ", ".join(unsupported_composites)
            + ". Use only Sequence and Fallback."
        )
    return errors


def build_runtime_recovery_scenario(
    scenario: Scenario,
    *,
    measured_initial_state: tuple[str, ...],
    failure_observation: dict[str, Any],
) -> Scenario:
    """Reveal the measured floor-recovery affordance only after the runtime failure."""
    expected = {
        "classification": "dropped_to_floor",
        "object": "primary_part",
        "object_usable": True,
        "recovery_location": "source_floor",
    }
    mismatches = [
        key for key, value in expected.items() if failure_observation.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Same-object recovery requires a measured, usable primary_part at source_floor; "
            f"observation fields did not match: {', '.join(mismatches)}."
        )

    go2 = scenario.robot("unitree_go2_z1")
    if go2 is None:
        raise ValueError("Recovery scenario does not declare unitree_go2_z1.")
    if scenario.capability("unitree_go2_z1", "recover_fallen_part") is not None:
        raise ValueError(
            "The nominal scenario already exposes recover_fallen_part; fault-blind planning "
            "requires adding it only after the measured failure."
        )

    recover_fallen_part = Capability(
        name="recover_fallen_part",
        parameters=("part", "location"),
        parameter_types=("part", "location"),
        resources=("source_zone",),
        action_type="manipulation",
        duration_ticks=3,
        timeout_ticks=60,
        preconditions=tuple(
            canonical_predicate(literal)
            for literal in (
                "system_ready()",
                "robot_ready(unitree_go2_z1)",
                "base_stationary(unitree_go2_z1)",
                "arm_stowed(unitree_go2_z1)",
                "docked(unitree_go2_z1,source_dock)",
                "usable(part)",
                "at(part,location)",
                "gripper_empty(unitree_go2_z1)",
            )
        ),
        effects=Effects(
            add=(canonical_predicate("holding(unitree_go2_z1,part)"),),
            delete=tuple(
                canonical_predicate(literal)
                for literal in (
                    "at(part)",
                    "gripper_empty(unitree_go2_z1)",
                    "arm_stowed(unitree_go2_z1)",
                )
            ),
        ),
    )
    runtime_robots = tuple(
        replace(robot, capabilities=(*robot.capabilities, recover_fallen_part))
        if robot.id == go2.id
        else robot
        for robot in scenario.robots
    )
    return replace(
        scenario,
        instruction=(
            f"{scenario.instruction} Continue from the measured failure snapshot without a reset. "
            "The same primary_part is intact and measured at source_floor. Recover it with the "
            "newly available Go2/Z1 floor-retrieval capability, then finish the mission."
        ),
        initial_state=tuple(canonical_predicate(fact) for fact in measured_initial_state),
        entities=(*scenario.entities, Entity(id="source_floor", type="location")),
        robots=runtime_robots,
    )
