from __future__ import annotations

from pathlib import Path

from llm_planner import LLMPlanner
from metrics import compute_trial_metrics
from repair import SimpleRepairLoop
from schemas import load_json, save_json
from simulator import SymbolicSimulator
from validator import BTValidator


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "outputs" / "llm_experiment"

    capabilities = load_json(str(data_dir / "capabilities.json"))
    task = load_json(str(data_dir / "packaging_task.json"))

    planner = LLMPlanner(task=task, capabilities=capabilities, mode="auto")
    plan = planner.generate_plan()
    save_json(str(output_dir / "llm_plan.json"), plan)

    before = _evaluate_plan(plan, task, capabilities)
    save_json(str(output_dir / "validation_result.json"), before["validation"])
    save_json(str(output_dir / "simulation_result.json"), before["simulation"])
    save_json(str(output_dir / "metrics.json"), before["metrics"])

    if before["validation"].get("valid"):
        repaired_plan = plan
        after = before
    else:
        repaired_plan = SimpleRepairLoop().repair(plan, before["validation"])
        after = _evaluate_plan(repaired_plan, task, capabilities)

    save_json(str(output_dir / "repaired_plan.json"), repaired_plan)
    save_json(str(output_dir / "repaired_validation_result.json"), after["validation"])
    save_json(str(output_dir / "repaired_simulation_result.json"), after["simulation"])
    save_json(str(output_dir / "repaired_metrics.json"), after["metrics"])

    _print_summary(planner, before, after)


def _evaluate_plan(plan: dict, task: dict, capabilities: dict) -> dict:
    validation_result = BTValidator(plan=plan, capabilities=capabilities).validate()
    if validation_result.get("valid"):
        simulation_result = SymbolicSimulator(
            initial_state=task["initial_state"],
            behavior_trees=plan["behavior_trees"],
            capabilities=capabilities,
        ).run()
    else:
        simulation_result = {
            "success": False,
            "skipped": True,
            "reason": "validation_failed",
            "final_state": [],
            "trace": [],
            "errors": [],
        }

    return {
        "validation": validation_result,
        "simulation": simulation_result,
        "metrics": compute_trial_metrics(validation_result, simulation_result, task.get("goal_state", [])),
    }


def _count_errors(result: dict) -> int:
    validation_errors = len(result["validation"].get("errors", []))
    if validation_errors:
        return validation_errors
    return len(result["simulation"].get("errors", []))


def _print_summary(planner: LLMPlanner, before: dict, after: dict) -> None:
    print("LLM planning experiment")
    print("=" * 32)
    print(f"LLM mode used: {planner.mode_used}")
    if planner.last_error:
        print(f"LLM fallback note: {planner.last_error}")
    print(f"Valid before repair: {_yes_no(before['validation'].get('valid'))}")
    print(f"Success before repair: {_yes_no(before['simulation'].get('success'))}")
    print(f"Errors before repair: {_count_errors(before)}")
    print(f"Valid after repair: {_yes_no(after['validation'].get('valid'))}")
    print(f"Success after repair: {_yes_no(after['simulation'].get('success'))}")
    print(f"Errors after repair: {_count_errors(after)}")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
