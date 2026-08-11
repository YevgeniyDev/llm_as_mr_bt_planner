"""Application service shared by the CLI and Gradio UI."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import ArtifactBundle, extract_plan_document, write_artifact_bundle
from .domain import Scenario, parse_scenario
from .llm import get_client
from .plan import Plan, parse_plan
from .planner import CancellationCheck, PlannerResult, PlanningCancelled, run_planner
from .simulation import SimulationReport, simulate, skipped_simulation
from .validation import ValidationReport, validate_plan

ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class PipelineOutcome:
    scenario: Scenario
    plan: Plan
    planner_result: PlannerResult
    validation: ValidationReport
    simulation: SimulationReport
    artifacts: ArtifactBundle
    log_entries: tuple[str, ...]


class PlannerService:
    def __init__(self, output_root: str | Path = "outputs/runs", client_factory=get_client) -> None:
        self.output_root = Path(output_root)
        self.client_factory = client_factory

    def parse_scenario_document(self, document: dict[str, Any]) -> Scenario:
        return parse_scenario(document, strict=True)

    def generate(
        self,
        scenario_document: dict[str, Any],
        *,
        provider: str,
        api_key: str,
        model: str | None = None,
        max_corrections: int = 4,
        max_ticks: int = 100,
        progress: ProgressCallback | None = None,
        cancelled: CancellationCheck | None = None,
    ) -> PipelineOutcome:
        started = time.monotonic()
        log_entries: list[str] = []

        def emit(message: str, fraction: float) -> None:
            elapsed = time.monotonic() - started
            timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            log_entries.append(f"{timestamp} | {elapsed:8.3f}s | {fraction:6.1%} | {message}")
            _progress(progress, message, fraction)

        _check_cancelled(cancelled)
        emit("Validating uploaded scenario", 0.02)
        scenario = self.parse_scenario_document(scenario_document)
        emit(
            f"Scenario validation passed: {scenario.task_id} ({len(scenario.robots)} robots, "
            f"{len(scenario.goal_state)} goals)",
            0.07,
        )
        if not api_key.strip():
            raise ValueError(f"An explicit {provider} API key is required for generation.")
        _check_cancelled(cancelled)
        emit(f"Creating the explicit {provider} provider client", 0.10)
        client = self.client_factory(provider, model=model or None, api_key=api_key.strip())

        def planner_progress(message: str, fraction: float) -> None:
            emit(message, 0.12 + fraction * 0.63)

        result = run_planner(
            scenario,
            client,
            max_corrections=max_corrections,
            max_ticks=max_ticks,
            progress=planner_progress,
            cancelled=cancelled,
        )
        _check_cancelled(cancelled)
        emit("Independently reloading the exact LLM-generated behavior trees", 0.77)
        plan = parse_plan(result.plan)
        emit("Re-running static validation without rewriting the LLM tree", 0.80)
        validation = validate_plan(plan, scenario, suggest_producers=True)
        if validation.valid:
            emit("Independent static validation passed", 0.83)
        else:
            kinds = ", ".join(sorted({error.type for error in validation.errors}))
            emit(f"Independent static validation failed: {kinds}", 0.83)
        _check_cancelled(cancelled)
        emit("Running the exact tree in the deterministic contract simulator", 0.86)
        simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
        if simulation.success:
            last_tick = max((int(event.get("tick", 0)) for event in simulation.trace), default=0)
            emit(f"Contract simulation passed in {last_tick} tick(s)", 0.90)
        else:
            kinds = ", ".join(sorted({str(error.get("type", "simulation_error")) for error in simulation.errors}))
            emit(f"Contract simulation failed: {kinds or 'static validation did not pass'}", 0.90)
        _check_cancelled(cancelled)
        publish_final = validation.valid and simulation.success
        emit(
            "Writing the final BT and audit bundle" if publish_final else "Writing diagnostics without a final BT",
            0.94,
        )
        publication_message = (
            "All checks passed; final BT publication is authorized"
            if publish_final
            else "Checks failed; diagnostic-only publication is authorized and no final BT will be written"
        )
        emit(publication_message, 0.97)
        artifacts = write_artifact_bundle(
            self.output_root,
            plan,
            scenario,
            provider=result.provider,
            model=result.model,
            correction_rounds=result.correction_rounds,
            validation=validation,
            simulation=simulation,
            result_payload=result.to_dict(),
            pipeline_log=log_entries,
            publish_final=publish_final,
        )
        return PipelineOutcome(
            scenario,
            plan,
            result,
            validation,
            simulation,
            artifacts,
            tuple(log_entries),
        )

    def validate_and_simulate(
        self,
        scenario_document: dict[str, Any],
        plan_document: dict[str, Any],
        *,
        max_ticks: int = 100,
    ) -> tuple[Scenario, Plan, ValidationReport, SimulationReport]:
        scenario = self.parse_scenario_document(scenario_document)
        plan = parse_plan(extract_plan_document(plan_document))
        validation = validate_plan(plan, scenario, suggest_producers=True)
        simulation = simulate(plan, scenario, max_ticks=max_ticks) if validation.valid else skipped_simulation()
        return scenario, plan, validation, simulation

    @staticmethod
    def load_json(path: str | Path) -> dict[str, Any]:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"JSON root in '{path}' must be an object.")
        return document


def _progress(callback: ProgressCallback | None, message: str, fraction: float) -> None:
    if callback is not None:
        callback(message, fraction)


def _check_cancelled(cancelled: CancellationCheck | None) -> None:
    if cancelled is not None and cancelled():
        raise PlanningCancelled("Pipeline cancelled by the user; no final BT was published.")
