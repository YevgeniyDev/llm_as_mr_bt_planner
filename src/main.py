from __future__ import annotations

import argparse
from pathlib import Path

from metrics import compute_trial_metrics
from planner import RuleBasedPlanner
from schemas import load_json, save_json
from simulator import SymbolicSimulator
from validator import BTValidator


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    capabilities = load_json(str(data_dir / "capabilities.json"))
    task = load_json(str(data_dir / task_filename(args.task)))

    planner = RuleBasedPlanner(task=task, capabilities=capabilities)
    plan = planner.generate_plan()
    output_prefix = output_prefix_for_task(task["task_id"])
    save_json(str(outputs_dir / f"{output_prefix}_plan.json"), plan)

    validator = BTValidator(plan=plan, capabilities=capabilities)
    validation_result = validator.validate()
    save_json(str(outputs_dir / f"{output_prefix}_validation_result.json"), validation_result)
    if task["task_id"] == "packaging_delivery":
        save_json(str(outputs_dir / "validation_result.json"), validation_result)

    simulator = SymbolicSimulator(
        initial_state=task["initial_state"],
        behavior_trees=plan["behavior_trees"],
        capabilities=capabilities,
    )
    simulation_result = simulator.run()
    save_json(str(outputs_dir / f"{output_prefix}_simulation_result.json"), simulation_result)
    if task["task_id"] == "packaging_delivery":
        save_json(str(outputs_dir / "simulation_result.json"), simulation_result)

    metrics = compute_trial_metrics(validation_result, simulation_result, task.get("goal_state", []))
    save_json(str(outputs_dir / f"{output_prefix}_metrics.json"), metrics)
    if task["task_id"] == "packaging_delivery":
        save_json(str(outputs_dir / "metrics.json"), metrics)

    print_summary(task["task_id"], validation_result, simulation_result, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the symbolic multi-robot BT planner.")
    parser.add_argument(
        "--task",
        choices=["packaging_delivery", "gear_assembly"],
        default="packaging_delivery",
        help="Task to plan and simulate.",
    )
    return parser.parse_args()


def task_filename(task_id: str) -> str:
    if task_id == "gear_assembly":
        return "gear_assembly_task.json"
    return "packaging_task.json"


def output_prefix_for_task(task_id: str) -> str:
    if task_id == "gear_assembly":
        return "gear_assembly"
    return "packaging"


def print_summary(task_id: str, validation_result: dict, simulation_result: dict, metrics: dict) -> None:
    print("LLM-as-Multi-Robot-BT-Planner symbolic prototype")
    print("=" * 58)
    print(f"Task: {task_id}")
    print()

    print(f"Validation: {'valid' if validation_result.get('valid') else 'invalid'}")
    if validation_result.get("errors"):
        print("Validation errors:")
        for error in validation_result["errors"]:
            print(f"- [{error.get('type')}] {error.get('message')}")
    else:
        print("Validation errors: none")

    print()
    print(f"Simulation: {'success' if simulation_result.get('success') else 'failure'}")
    print(f"Goal success: {metrics.get('goal_success')}")
    if simulation_result.get("errors"):
        print("Simulation errors:")
        for error in simulation_result["errors"]:
            print(f"- [{error.get('type')}] {error.get('message')}")
    else:
        print("Simulation errors: none")

    print()
    print("Final state:")
    for predicate in simulation_result.get("final_state", []):
        print(f"- {predicate}")

    print()
    print("Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")

    print()
    print("For ablation experiments, run: python src/run_experiments.py")


if __name__ == "__main__":
    main()
