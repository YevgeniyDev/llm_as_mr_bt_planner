"""MRBTP baseline (Cai et al. 2025) - run the authors' released code, ingest results.

MRBTP (https://github.com/DIDS-EI/MRBTP) is a symbolic multi-robot BT planner whose
scenarios are MAGrid environment subclasses, not external JSON. So it is run *outside*
this package (see ``scripts/run_mrbtp.py``), and this module ingests the results it
produced into the same :class:`PlannerResult`/metrics shape as every other method.

MRBTP is reported under a transparent native protocol because its standard
Condition nodes return FAILURE while this project's synchronization guards return
RUNNING. Native results therefore share table columns but are explicitly labelled
``metric_scope=native_mrbtp`` and are never presented as same-simulator evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import resolve_project_path
from ..domain import Scenario
from ..llm.base import LLMClient
from ..plan import parse_plan
from ..planner import PlannerResult
from .base import score_plan

DEFAULT_RESULTS_PATH = "outputs/mrbtp_results.json"


def run_mrbtp(
    scenario: Scenario,
    client: LLMClient | None = None,  # noqa: ARG001 - MRBTP is not LLM-driven here
    *,
    results_path: str = DEFAULT_RESULTS_PATH,
    max_ticks: int = 80,
    **kwargs: Any,
) -> PlannerResult:
    """Return MRBTP's result for ``scenario`` from a precomputed results file."""
    record = _load_record(results_path, scenario.task_id)
    provider, model = "mrbtp", record.get("variant", "MRBTP")

    if record.get("plan") is not None and record.get("comparison_protocol") == "shared_adapter_v1":
        plan = parse_plan({**record["plan"], "task_id": scenario.task_id})
        validation, simulation = score_plan(plan, scenario, max_ticks=max_ticks)
        return PlannerResult(
            task_id=scenario.task_id,
            provider=provider,
            model=model,
            valid=validation.valid,
            success=simulation.success,
            goal_success=simulation.goal_success,
            correction_rounds=int(record.get("feedback_rounds", 0)),
            plan=plan.to_dict(),
            validation_errors=validation.to_dicts(),
            simulation={
                "final_state": simulation.final_state,
                "trace": simulation.trace,
                "errors": simulation.errors,
            },
            wall_seconds=float(record.get("planning_time", 0.0)),
        )

    protocol = record.get("comparison_protocol", "legacy_unverified")
    success = bool(record.get("native_success", record.get("success", False)))
    native_valid = bool(record.get("valid", False)) and protocol == "mrbtp_native_v1"
    errors = []
    if not success:
        errors.append({"type": f"mrbtp_{record.get('outcome', 'unverified')}",
                       "message": record.get("error") or "MRBTP did not return a grounded native plan."})
    return PlannerResult(
        task_id=scenario.task_id,
        provider=provider,
        model=model,
        valid=native_valid,
        success=success,
        goal_success=bool(record.get("goal_success", success)),
        correction_rounds=int(record.get("feedback_rounds", 0)),
        plan=record.get("plan_summary", {}),
        validation_errors=errors,
        simulation={"final_state": [], "trace": [], "errors": errors},
        wall_seconds=float(record.get("planning_time", 0.0)),
        metric_scope="native_mrbtp",
        native_metrics={
            "comparison_protocol": protocol,
            "outcome": record.get("outcome"),
            "timed_out": bool(record.get("timed_out", False)),
            "grounded_condition_found": bool(record.get("grounded_condition_found", False)),
            "tree_count": int(record.get("tree_count", 0)),
            "expanded_count": record.get("expanded_count"),
        },
    )


def _load_record(results_path: str, task_id: str) -> dict[str, Any]:
    path = resolve_project_path(results_path)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"MRBTP results file not found: {path}. Run scripts/run_mrbtp.py first "
            f"(it drives the third_party/MRBTP code and writes this file)."
        )
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("scenarios", data) if isinstance(data, dict) else {}
    if task_id not in records:
        raise KeyError(f"No MRBTP result for scenario '{task_id}' in {path}.")
    return records[task_id]


# MRBTP AnyTreeNode.node_type -> our composite type.
_COMPOSITE = {"sequence": "Sequence", "selector": "Fallback"}


def _convert_node(node: dict[str, Any]) -> dict[str, Any]:
    """One serialized MRBTP ``AnyTreeNode`` -> one of our BT nodes.

    Serialized node shape (produced by ``scripts/run_mrbtp.py`` from AnyTreeNode):
    ``{"node_type", "cls_name", "args": [...], "children": [...]}``. Action names are
    bare (no robot token) - the porting names PlanningActions ``action(p1,p2)`` so
    ``cls_name`` is our action name and ``args`` our parameters.
    """
    node_type = node.get("node_type")
    cls_name = node.get("cls_name")
    args = [str(a) for a in node.get("args", [])]
    children = node.get("children", [])

    if node_type in _COMPOSITE:
        return {"type": _COMPOSITE[node_type], "children": [_convert_node(c) for c in children]}
    if node_type == "action":
        return {"type": "Action", "name": cls_name, "parameters": args}
    if node_type in ("condition", "composite_condition"):
        if children:  # a composite condition wraps several sub-conditions
            return {"type": "Sequence", "children": [_convert_node(c) for c in children]}
        return {"type": "Condition", "name": cls_name, "parameters": args}
    # Unknown node type: pass children through inside a Sequence so nothing is dropped.
    return {"type": "Sequence", "children": [_convert_node(c) for c in children]}


def mrbtp_bt_to_plan(mrbtp_trees: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    """Convert MRBTP per-robot trees (serialized AnyTreeNodes) into our Plan JSON.

    ``mrbtp_trees`` maps ``robot_id -> serialized AnyTreeNode root``. We rebuild the
    per-robot behavior trees, then derive ``task_graph`` and ``assignments`` from the
    Action leaves for inspection or an explicitly labelled adapter study. MRBTP shares
    conditions across trees for coordination rather than declaring explicit edges, so
    ``synchronization`` is left empty (our V2 checks global producer existence, which
    a sound MRBTP plan satisfies).
    """
    behavior_trees: dict[str, Any] = {}
    task_graph: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for robot_id, root in mrbtp_trees.items():
        tree = _convert_node(root)
        behavior_trees[robot_id] = tree
        for index, leaf in enumerate(_iter_action_leaves(tree)):
            task_id = f"{robot_id}__{index}"
            task_graph.append(
                {"id": task_id, "action": leaf["name"], "parameters": leaf["parameters"], "depends_on": []}
            )
            assignments.append({"task_id": task_id, "robot": robot_id})
    return {
        "task_id": scenario.task_id,
        "task_graph": task_graph,
        "assignments": assignments,
        "synchronization": [],
        "behavior_trees": behavior_trees,
    }


def _iter_action_leaves(node: dict[str, Any]):
    if node.get("type") == "Action":
        yield node
    for child in node.get("children", []):
        yield from _iter_action_leaves(child)
