"""Official-source-aligned common-domain runner for non-LLM MRBTP."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..artifacts import canonical_json
from ..bt import iter_nodes
from ..config import save_json
from ..domain import Scenario, scenario_to_dict
from ..plan import Plan, parse_plan
from ..simulation import SimulationReport, simulate, skipped_simulation
from ..validation import ValidationReport, validate_plan
from .llm_bt_native import reachable_action_templates
from .mrbtp_native import (
    MRBTPConstruction,
    intention_sharing_document,
    native_forest_document,
    plan_mrbtp,
    planning_graph_document,
    validate_native_construction,
)
from .mrbtp_source import (
    MRBTP_AAAI_URL,
    MRBTP_ARCHIVE_SHA256,
    MRBTP_COMMIT,
    MRBTP_PAPER_URL,
    MRBTP_REPOSITORY_URL,
)

MRBTP_METHOD_ID = "mrbtp"
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class MRBTPRun:
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


def run_mrbtp(
    scenario: Scenario,
    output_root: str | Path,
    *,
    max_expansions: int = 10_000,
    max_ticks: int = 300,
    invocation: list[str] | None = None,
    verified_source_manifest: str | Path | None = None,
) -> MRBTPRun:
    """Plan the common problem through FIFO MRBTP without the optional LLM plugin."""
    started = time.perf_counter()
    base = _new_run_directory(Path(output_root), scenario.task_id)
    native_dir = base / "native"
    native_dir.mkdir(parents=True, exist_ok=False)
    generation_errors: list[dict[str, Any]] = []
    construction: MRBTPConstruction | None = None
    try:
        construction = plan_mrbtp(scenario, max_expansions=max_expansions)
        generation_errors.extend(validate_native_construction(scenario, construction))
    except Exception as error:
        generation_errors.append({"type": "mrbtp_generation_error", "message": str(error)})

    artifact_files = _write_native_artifacts(
        scenario,
        native_dir,
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
    plan = parse_plan(plan_document)
    validation = validate_plan(
        plan,
        scenario,
        suggest_producers=False,
        allowed_sources=frozenset({"planner"}),
        validation_profile="reactive_policy",
    )
    plan_generation_success = (
        construction is not None
        and construction.solved
        and not generation_errors
        and set(plan.behavior_trees) == scenario.robot_ids
    )
    static_validity = plan_generation_success and validation.valid
    simulation = (
        simulate(plan, scenario, max_ticks=max_ticks)
        if static_validity
        else skipped_simulation()
    )
    wall_seconds = time.perf_counter() - started

    canonical_path = base / "canonical_plan.json"
    accepted_path = (
        base / "accepted_plan.json"
        if static_validity and simulation.success
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
            "valid": static_validity,
            "common_reactive_policy_errors": validation.to_dicts(),
            "native_mrbtp_errors": generation_errors,
            "common_validation_profile": "reactive_policy",
        },
    )
    save_json(simulation_path, simulation.to_dict())
    save_json(scenario_path, scenario_to_dict(scenario))
    metrics = _metrics_payload(
        plan,
        validation,
        simulation,
        construction,
        generation_errors,
        plan_generation_success,
        wall_seconds,
    )
    save_json(metrics_path, metrics)
    artifact_files.extend(
        [canonical_path, validation_path, simulation_path, metrics_path, scenario_path]
    )
    if accepted_path is not None:
        artifact_files.append(accepted_path)

    source_manifest_path = (
        Path(verified_source_manifest).resolve()
        if verified_source_manifest is not None
        else None
    )
    if source_manifest_path is not None and not source_manifest_path.is_file():
        raise ValueError("The verified MRBTP source manifest path does not exist.")
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "id": MRBTP_METHOD_ID,
            "name": "MRBTP",
            "paper": MRBTP_PAPER_URL,
            "aaai_doi": MRBTP_AAAI_URL,
            "official_repository": MRBTP_REPOSITORY_URL,
            "official_commit": MRBTP_COMMIT,
            "software_license": "MIT",
            "implementation": "official-source-aligned common-domain port",
            "variant": "FIFO MRBTP/MABTP without composite actions",
            "llm_subtree_plugin_enabled": False,
            "llm_calls": 0,
            "track": "nominal",
        },
        "invocation": invocation or [],
        "scenario_sha256": _sha256_json(scenario_to_dict(scenario)),
        "canonical_plan_sha256": _sha256_json(plan_document),
        "prepared_source": {
            "verified_before_run": source_manifest_path is not None,
            "manifest": str(source_manifest_path) if source_manifest_path is not None else None,
            "manifest_sha256": (
                _sha256_file(source_manifest_path) if source_manifest_path is not None else None
            ),
            "archive_sha256": MRBTP_ARCHIVE_SHA256,
        },
        "fidelity": {
            "native_architecture": [
                "shared FIFO queue initialized with the complete team goal",
                "per-robot action spaces and one native fallback policy per robot",
                "one-step in-tree or cross-tree expansion for every explored condition",
                "premise formula pre(a) union c minus add(a)",
                "per-agent subset pruning and common-domain conflict rejection",
                "homogeneous backup branches retained",
                "intention-sharing priority and physical-precondition blocking",
            ],
            "input_adaptations": [
                "official PlanningAction pre/add/del sets are instantiated from grounded common capability contracts",
                "the complete common goal state is supplied before proven invariant initial facts are removed from regression conditions",
                "common part-location, holding, gripper, and docking invariants provide conflict checks",
                "relaxed reachability removes grounded actions whose preconditions cannot be supported in this scenario",
                "a deterministic feasible landmark order selects one unresolved literal per FIFO expansion while retaining every reachable producer for that literal",
            ],
            "output_adaptations": [
                "the complete official-style backup policy remains in native_forest.json",
                "the solved native witness is projected to per-robot common BT JSON for deterministic execution",
                "capability-required exclusive resources are exposed as AcquireResource/ReleaseResource leaves",
                "cross-robot witness dependencies become bounded WaitFor leaves",
            ],
            "semantic_task_action_rewrites": [],
            "validator_feedback_to_planner": False,
            "common_validation_profile": (
                "reactive_policy structure/contracts plus independent native MRBTP invariant validation; "
                "direct flattened-order checks are inapplicable to fallback backup policies"
            ),
            "known_limits": [
                "the optional LLM-generated composite-action plugin is disabled for the non-LLM reference",
                "MiniGrid and VirtualHome are replaced by the shared symbolic and MuJoCo evaluators",
                "the common executor uses deterministic round-robin ticks and does not emulate speculative belief success",
                "native communication-loss and homogeneous-action failure experiments are outside this nominal track",
            ],
        },
        "results": metrics,
        "files": {
            _relative(base, path): _sha256_file(path)
            for path in artifact_files
            if path.is_file()
        },
    }
    save_json(manifest_path, manifest)
    return MRBTPRun(
        directory=base,
        canonical_plan=canonical_path,
        accepted_plan=accepted_path,
        validation_report=validation_path,
        simulation_trace=simulation_path,
        metrics=metrics_path,
        manifest=manifest_path,
        plan_generation_success=plan_generation_success,
        static_validity=static_validity,
        symbolic_goal_success=simulation.goal_success,
    )


def _write_native_artifacts(
    scenario: Scenario,
    native_dir: Path,
    construction: MRBTPConstruction | None,
    errors: list[dict[str, Any]],
) -> list[Path]:
    paths = {
        "problem": native_dir / "problem.json",
        "actions": native_dir / "action_spaces.json",
        "graph": native_dir / "planning_graph.json",
        "trace": native_dir / "expansion_trace.json",
        "forest": native_dir / "native_forest.json",
        "intentions": native_dir / "intention_sharing.json",
        "canonical": native_dir / "canonical_observation.json",
    }
    save_json(
        paths["problem"],
        {
            "initial_state": list(scenario.initial_state),
            "goal_condition": list(scenario.goal_state),
            "robot_order": [robot.id for robot in scenario.robots],
            "algorithm": "FIFO MRBTP without optional LLM composite actions",
        },
    )
    grounded = reachable_action_templates(scenario)
    save_json(
        paths["actions"],
        {
            robot.id: [action.to_dict() for action in grounded if action.robot == robot.id]
            for robot in scenario.robots
        },
    )
    if construction is None:
        save_json(paths["graph"], {"solved": False, "errors": errors})
        save_json(paths["trace"], {"events": [], "errors": errors})
        save_json(paths["forest"], {})
        save_json(paths["canonical"], {})
    else:
        save_json(paths["graph"], planning_graph_document(construction))
        save_json(paths["trace"], {"events": construction.trace, "errors": errors})
        save_json(paths["forest"], native_forest_document(construction))
        save_json(
            paths["canonical"],
            {robot: tree.to_dict() for robot, tree in construction.trees.items()},
        )
    save_json(paths["intentions"], intention_sharing_document(scenario))
    return list(paths.values())


def _metrics_payload(
    plan: Plan,
    validation: ValidationReport,
    simulation: SimulationReport,
    construction: MRBTPConstruction | None,
    native_errors: list[dict[str, Any]],
    plan_generation_success: bool,
    wall_seconds: float,
) -> dict[str, Any]:
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    edges = construction.expanded_edges if construction is not None else []
    return {
        "track": "nominal",
        "wall_seconds": round(wall_seconds, 4),
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "monetary_cost": 0.0,
        "plan_generation_success": plan_generation_success,
        "static_validity": plan_generation_success and validation.valid,
        "symbolic_goal_success": simulation.goal_success,
        "nominal_execution_success": None,
        "bt_node_count": len(nodes),
        "action_branch_count": sum(node.type == "Action" for node in nodes),
        "maximum_tree_depth": max(
            (_tree_depth(tree) for tree in plan.behavior_trees.values()),
            default=0,
        ),
        "expanded_condition_count": (
            len(construction.explored_conditions) if construction is not None else 0
        ),
        "expanded_edge_count": len(edges),
        "cross_tree_edge_count": sum(edge.operation == "cross_tree_expand" for edge in edges),
        "in_tree_edge_count": sum(edge.operation == "in_tree_expand" for edge in edges),
        "solution_witness_action_count": (
            len(construction.witness) if construction is not None else 0
        ),
        "common_validation_error_count": len(validation.errors),
        "native_validation_errors": native_errors,
    }


def _tree_depth(node) -> int:
    return 1 + max((_tree_depth(child) for child in node.children), default=0)


def _new_run_directory(output_root: Path, task_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / f"{_safe_name(task_id)}-nominal-{stamp}"
    counter = 1
    while base.exists():
        base = output_root / f"{_safe_name(task_id)}-nominal-{stamp}-{counter}"
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
