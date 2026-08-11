"""Canonical, auditable output artifacts for one planner run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import save_json, save_text
from .domain import Scenario, scenario_to_dict
from .plan import Plan, parse_plan
from .simulation import SimulationReport
from .validation import ValidationReport
from .viz import plan_to_html
from .xml_export import export_behaviortree_cpp_xml

ARTIFACT_FORMAT = "lmrbtp.multi_robot_bt/2.0"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def build_bt_artifact(
    plan: Plan,
    scenario: Scenario,
    *,
    provider: str,
    model: str,
    correction_rounds: int,
    validation: ValidationReport,
    simulation: SimulationReport,
) -> dict[str, Any]:
    plan_document = plan.to_dict()
    scenario_document = scenario_to_dict(scenario)
    identity = {
        "artifact_format": ARTIFACT_FORMAT,
        "scenario_sha256": sha256_json(scenario_document),
        "plan": plan_document,
    }
    return {
        **identity,
        "artifact_sha256": sha256_json(identity),
        "provenance": {
            "provider": provider,
            "model": model,
            "correction_rounds": correction_rounds,
            "generation_method": "direct_llm_behavior_tree",
            "semantic_rewrites": [],
            "node_source": "llm",
        },
        "verification": {
            "static_validation_passed": validation.valid,
            "contract_simulation_passed": simulation.success,
            "goal_reached": simulation.goal_success,
        },
        "execution_scope": {
            "validated": "symbolic capability and synchronization contract",
            "not_validated": ["robot dynamics", "collision avoidance", "perception", "ROS 2", "real hardware"],
        },
    }


def extract_plan_document(document: dict[str, Any]) -> dict[str, Any]:
    """Accept a plain plan or the exported artifact wrapper."""
    if document.get("artifact_format") == ARTIFACT_FORMAT:
        plan = document.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("BT artifact has no object-valued 'plan'.")
        return plan
    return document


def verify_artifact(document: dict[str, Any]) -> bool:
    if document.get("artifact_format") != ARTIFACT_FORMAT:
        return False
    identity = {
        "artifact_format": document.get("artifact_format"),
        "scenario_sha256": document.get("scenario_sha256"),
        "plan": document.get("plan"),
    }
    return document.get("artifact_sha256") == sha256_json(identity)


@dataclass(frozen=True)
class ArtifactBundle:
    directory: Path
    behavior_tree_json: Path | None
    behavior_tree_xml: Path | None
    validation_report: Path
    simulation_trace: Path
    html_report: Path
    manifest: Path
    scenario: Path
    result: Path
    pipeline_log: Path

    def download_paths(self) -> list[str]:
        paths = [
            str(self.validation_report),
            str(self.simulation_trace),
            str(self.html_report),
            str(self.manifest),
            str(self.pipeline_log),
        ]
        if self.behavior_tree_json is not None:
            paths.insert(0, str(self.behavior_tree_json))
        if self.behavior_tree_xml is not None:
            paths.insert(1, str(self.behavior_tree_xml))
        return paths


def write_artifact_bundle(
    output_root: str | Path,
    plan: Plan,
    scenario: Scenario,
    *,
    provider: str,
    model: str,
    correction_rounds: int,
    validation: ValidationReport,
    simulation: SimulationReport,
    result_payload: dict[str, Any],
    pipeline_log: list[str],
    publish_final: bool | None = None,
) -> ArtifactBundle:
    output_root = Path(output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / f"{_safe_name(scenario.task_id)}-{stamp}"
    counter = 1
    while base.exists():
        base = output_root / f"{_safe_name(scenario.task_id)}-{stamp}-{counter}"
        counter += 1
    base.mkdir(parents=True, exist_ok=False)

    publish_final = validation.valid and simulation.success if publish_final is None else publish_final
    bundle = ArtifactBundle(
        directory=base,
        behavior_tree_json=base / "behavior_tree.json" if publish_final else None,
        behavior_tree_xml=base / "behavior_tree.xml" if publish_final else None,
        validation_report=base / "validation_report.json",
        simulation_trace=base / "simulation_trace.json",
        html_report=base / "report.html",
        manifest=base / "manifest.json",
        scenario=base / "scenario.json",
        result=base / "result.json",
        pipeline_log=base / "pipeline.log",
    )
    artifact = build_bt_artifact(
        plan,
        scenario,
        provider=provider,
        model=model,
        correction_rounds=correction_rounds,
        validation=validation,
        simulation=simulation,
    )
    if bundle.behavior_tree_json is not None and bundle.behavior_tree_xml is not None:
        save_json(bundle.behavior_tree_json, artifact)
        save_text(bundle.behavior_tree_xml, export_behaviortree_cpp_xml(plan))
    save_json(
        bundle.validation_report,
        {
            "valid": validation.valid,
            "errors": validation.to_dicts(),
            "semantics": {
                "Condition": "SUCCESS when true, otherwise FAILURE",
                "WaitFor": "SUCCESS when true, RUNNING until timeout, then FAILURE",
                "Action": "SUCCESS after applying declared effects; unmet preconditions are FAILURE",
            },
        },
    )
    save_json(bundle.simulation_trace, simulation.to_dict())
    save_json(bundle.scenario, scenario_to_dict(scenario))
    save_json(bundle.result, result_payload)
    save_text(bundle.pipeline_log, "\n".join(pipeline_log) + "\n")
    save_text(
        bundle.html_report,
        plan_to_html(
            plan,
            title=f"Multi-robot BT: {scenario.task_id}",
            meta={
                "provider/model": f"{provider}/{model}",
                "valid": validation.valid,
                "contract simulation": simulation.success,
                "artifact SHA-256": artifact["artifact_sha256"],
            },
            trace=simulation.trace,
        ),
    )

    files = [
        bundle.validation_report,
        bundle.simulation_trace,
        bundle.html_report,
        bundle.scenario,
        bundle.result,
        bundle.pipeline_log,
    ]
    if bundle.behavior_tree_json is not None:
        files.append(bundle.behavior_tree_json)
    if bundle.behavior_tree_xml is not None:
        files.append(bundle.behavior_tree_xml)
    manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": scenario.task_id,
        "artifact_sha256": artifact["artifact_sha256"] if publish_final else None,
        "pipeline_passed": publish_final,
        "final_bt_published": publish_final,
        "files": {path.name: _sha256_file(path) for path in files},
        "scope": artifact["execution_scope"],
    }
    save_json(bundle.manifest, manifest)
    return bundle


def load_plan_file(path: str | Path) -> Plan:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_plan(extract_plan_document(document))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    return safe or "mission"
