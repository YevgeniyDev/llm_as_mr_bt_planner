from __future__ import annotations

import argparse
from pathlib import Path

from experiments import build_experiment_plans
from llm_planner import LLMPlanner
from metrics import compute_trial_metrics
from planner import RuleBasedPlanner
from repair import SimpleRepairLoop
from schemas import load_json, save_json
from simulator import SymbolicSimulator
from validator import BTValidator


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    outputs_dir = _experiments_output_dir(project_root, args.task)

    capabilities = load_json(str(data_dir / "capabilities.json"))
    task = load_json(str(data_dir / _task_filename(args.task)))
    base_plan = RuleBasedPlanner(task=task, capabilities=capabilities).generate_plan()
    experiment_plans = build_experiment_plans(base_plan)
    if args.task == "packaging_delivery":
        experiment_plans["mock_llm_bad"] = LLMPlanner(task=task, capabilities=capabilities, mode="mock_bad").generate_plan()
    repair_loop = SimpleRepairLoop()

    summary = []
    for experiment_name, plan in experiment_plans.items():
        experiment_dir = outputs_dir / experiment_name
        before = _evaluate_plan(plan, task, capabilities)
        save_json(str(experiment_dir / "plan.json"), plan)
        save_json(str(experiment_dir / "validation_result.json"), before["validation"])
        save_json(str(experiment_dir / "simulation_result.json"), before["simulation"])
        save_json(str(experiment_dir / "metrics.json"), before["metrics"])

        if before["validation"].get("valid"):
            repaired_plan = plan
            after = before
        else:
            repaired_plan = repair_loop.repair(plan, before["validation"])
            after = _evaluate_plan(repaired_plan, task, capabilities)

        save_json(str(experiment_dir / "repaired_plan.json"), repaired_plan)
        save_json(str(experiment_dir / "repaired_validation_result.json"), after["validation"])
        save_json(str(experiment_dir / "repaired_simulation_result.json"), after["simulation"])
        save_json(str(experiment_dir / "repaired_metrics.json"), after["metrics"])

        summary.append(
            {
                "experiment": experiment_name,
                "valid_before_repair": bool(before["validation"].get("valid")),
                "success_before_repair": bool(before["simulation"].get("success")),
                "num_errors_before_repair": _count_reportable_errors(before),
                "valid_after_repair": bool(after["validation"].get("valid")),
                "success_after_repair": bool(after["simulation"].get("success")),
                "num_errors_after_repair": _count_reportable_errors(after),
            }
        )

    save_json(str(outputs_dir / "summary.json"), summary)
    _print_summary_table(summary)


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

    metrics = compute_trial_metrics(validation_result, simulation_result, task.get("goal_state", []))
    return {
        "validation": validation_result,
        "simulation": simulation_result,
        "metrics": metrics,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run symbolic ablation experiments.")
    parser.add_argument(
        "--task",
        choices=["packaging_delivery", "gear_assembly"],
        default="packaging_delivery",
        help="Task to run ablations for.",
    )
    return parser.parse_args()


def _task_filename(task_id: str) -> str:
    if task_id == "gear_assembly":
        return "gear_assembly_task.json"
    return "packaging_task.json"


def _experiments_output_dir(project_root: Path, task_id: str) -> Path:
    if task_id == "gear_assembly":
        return project_root / "outputs" / "experiments_gear_assembly"
    return project_root / "outputs" / "experiments"


def _count_reportable_errors(result: dict) -> int:
    validation_errors = len(result["validation"].get("errors", []))
    if validation_errors:
        return validation_errors
    return len(result["simulation"].get("errors", []))


def _print_summary_table(summary: list[dict]) -> None:
    headers = ["Experiment", "Valid", "Success", "Repaired Valid", "Repaired Success", "Errors", "Repaired Errors"]
    rows = [
        [
            item["experiment"],
            _yes_no(item["valid_before_repair"]),
            _yes_no(item["success_before_repair"]),
            _yes_no(item["valid_after_repair"]),
            _yes_no(item["success_after_repair"]),
            str(item["num_errors_before_repair"]),
            str(item["num_errors_after_repair"]),
        ]
        for item in summary
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def _format_row(row: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    main()
