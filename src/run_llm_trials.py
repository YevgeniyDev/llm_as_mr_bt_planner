from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from llm_planner import LLMPlanner
from metrics import compute_trial_metrics
from repair import SimpleRepairLoop
from schemas import load_json, save_json
from simulator import SymbolicSimulator
from validator import BTValidator


VALID_MODES = {"auto", "mock_bad", "mock_good", "openai"}


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / args.output_dir

    capabilities = load_json(str(data_dir / "capabilities.json"))
    task = load_json(str(data_dir / _task_filename(args.task)))
    _prepare_output_dir(output_dir)

    trial_summaries = []
    for trial_index in range(1, args.trials + 1):
        trial_dir = output_dir / f"trial_{trial_index:02d}"
        trial_summary = _run_trial(
            trial=trial_index,
            mode=args.mode,
            task=task,
            capabilities=capabilities,
            trial_dir=trial_dir,
        )
        save_json(str(trial_dir / "trial_summary.json"), trial_summary)
        trial_summaries.append(trial_summary)

    aggregate_summary = _aggregate_summaries(trial_summaries, args.mode, args.task)
    save_json(str(output_dir / "summary.json"), aggregate_summary)
    _write_summary_csv(output_dir / "summary.csv", trial_summaries)

    _print_trial_table(trial_summaries)
    _print_aggregate_summary(aggregate_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated LLM planning trials.")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run.")
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="auto", help="LLMPlanner mode.")
    parser.add_argument("--output-dir", default="outputs/llm_trials", help="Output directory.")
    parser.add_argument(
        "--task",
        choices=["packaging_delivery", "gear_assembly"],
        default="packaging_delivery",
        help="Task to use for LLM trials.",
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    return args


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("trial_"):
            shutil.rmtree(path)
        elif path.name in {"summary.json", "summary.csv"}:
            path.unlink()


def _run_trial(trial: int, mode: str, task: dict, capabilities: dict, trial_dir: Path) -> dict:
    trial_dir.mkdir(parents=True, exist_ok=True)
    planner = LLMPlanner(task=task, capabilities=capabilities, mode=mode)

    try:
        plan = planner.generate_plan()
    except Exception as error:
        return _record_failed_trial(
            trial=trial,
            trial_dir=trial_dir,
            task_id=task.get("task_id", "unknown"),
            mode_used="api_error",
            error_message=_safe_error_message(error),
        )

    plan.setdefault("task_id", task.get("task_id"))
    save_json(str(trial_dir / "llm_plan.json"), plan)

    if mode == "openai" and planner.mode_used == "mock_bad" and planner.last_error:
        return _record_failed_trial(
            trial=trial,
            trial_dir=trial_dir,
            task_id=task.get("task_id", "unknown"),
            mode_used="openai_unavailable",
            error_message=planner.last_error,
        )

    if planner.mode_used == "mock_bad_fallback":
        return _record_failed_trial(
            trial=trial,
            trial_dir=trial_dir,
            task_id=task.get("task_id", "unknown"),
            mode_used=planner.mode_used,
            error_message=planner.last_error or "OpenAI-compatible API call failed.",
        )

    before = _evaluate_plan(plan, task, capabilities)
    save_json(str(trial_dir / "validation_result.json"), before["validation"])
    save_json(str(trial_dir / "simulation_result.json"), before["simulation"])
    save_json(str(trial_dir / "metrics.json"), before["metrics"])

    repaired_plan = SimpleRepairLoop().repair(plan, before["validation"])
    save_json(str(trial_dir / "repaired_plan.json"), repaired_plan)

    after = _evaluate_plan(repaired_plan, task, capabilities)
    save_json(str(trial_dir / "repaired_validation_result.json"), after["validation"])
    save_json(str(trial_dir / "repaired_simulation_result.json"), after["simulation"])
    save_json(str(trial_dir / "repaired_metrics.json"), after["metrics"])

    return _build_trial_summary(
        trial=trial,
        task_id=task.get("task_id", "unknown"),
        mode_used=planner.mode_used,
        before=before,
        after=after,
    )


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


def _record_failed_trial(trial: int, trial_dir: Path, task_id: str, mode_used: str, error_message: str) -> dict:
    trial_dir.mkdir(parents=True, exist_ok=True)
    safe_message = _safe_error_message(error_message)
    (trial_dir / "api_error.txt").write_text(safe_message + "\n", encoding="utf-8")
    placeholder_plan = {"task_id": task_id, "_llm_metadata": {"mode_used": mode_used, "error": "api_error"}}
    if not (trial_dir / "llm_plan.json").exists():
        save_json(str(trial_dir / "llm_plan.json"), placeholder_plan)
    save_json(str(trial_dir / "repaired_plan.json"), placeholder_plan)

    validation_result = {
        "valid": False,
        "errors": [
            {
                "type": "api_error",
                "message": "LLM API call failed; see api_error.txt for sanitized details.",
            }
        ],
    }
    simulation_result = {
        "success": False,
        "skipped": True,
        "reason": "api_error",
        "final_state": [],
        "trace": [],
        "errors": [],
    }
    metrics = compute_trial_metrics(validation_result, simulation_result, [])

    save_json(str(trial_dir / "validation_result.json"), validation_result)
    save_json(str(trial_dir / "simulation_result.json"), simulation_result)
    save_json(str(trial_dir / "metrics.json"), metrics)
    save_json(str(trial_dir / "repaired_validation_result.json"), validation_result)
    save_json(str(trial_dir / "repaired_simulation_result.json"), simulation_result)
    save_json(str(trial_dir / "repaired_metrics.json"), metrics)

    return {
        "trial": trial,
        "task_id": task_id,
        "mode_used": mode_used,
        "valid_before_repair": False,
        "success_before_repair": False,
        "errors_before_repair": 1,
        "valid_after_repair": False,
        "success_after_repair": False,
        "errors_after_repair": 1,
        "trace_length_after_repair": 0,
    }


def _build_trial_summary(trial: int, task_id: str, mode_used: str, before: dict, after: dict) -> dict:
    return {
        "trial": trial,
        "task_id": task_id,
        "mode_used": mode_used,
        "valid_before_repair": bool(before["validation"].get("valid")),
        "success_before_repair": bool(before["simulation"].get("success")),
        "errors_before_repair": _count_errors(before),
        "valid_after_repair": bool(after["validation"].get("valid")),
        "success_after_repair": bool(after["simulation"].get("success")),
        "errors_after_repair": _count_errors(after),
        "trace_length_after_repair": len(after["simulation"].get("trace", [])),
    }


def _count_errors(result: dict) -> int:
    validation_errors = len(result["validation"].get("errors", []))
    if validation_errors:
        return validation_errors
    return len(result["simulation"].get("errors", []))


def _aggregate_summaries(trial_summaries: list[dict], mode_requested: str, task_id: str) -> dict:
    num_trials = len(trial_summaries)
    raw_valid_count = sum(1 for item in trial_summaries if item["valid_before_repair"])
    raw_success_count = sum(1 for item in trial_summaries if item["success_before_repair"])
    repaired_valid_count = sum(1 for item in trial_summaries if item["valid_after_repair"])
    repaired_success_count = sum(1 for item in trial_summaries if item["success_after_repair"])

    return {
        "num_trials": num_trials,
        "mode_requested": mode_requested,
        "task_id": task_id,
        "raw_valid_count": raw_valid_count,
        "raw_success_count": raw_success_count,
        "repaired_valid_count": repaired_valid_count,
        "repaired_success_count": repaired_success_count,
        "raw_valid_rate": _rate(raw_valid_count, num_trials),
        "raw_success_rate": _rate(raw_success_count, num_trials),
        "repaired_valid_rate": _rate(repaired_valid_count, num_trials),
        "repaired_success_rate": _rate(repaired_success_count, num_trials),
        "average_errors_before_repair": _average(item["errors_before_repair"] for item in trial_summaries),
        "average_errors_after_repair": _average(item["errors_after_repair"] for item in trial_summaries),
        "average_trace_length_after_repair": _average(item["trace_length_after_repair"] for item in trial_summaries),
    }


def _write_summary_csv(path: Path, trial_summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "task_id",
        "mode_used",
        "valid_before_repair",
        "success_before_repair",
        "errors_before_repair",
        "valid_after_repair",
        "success_after_repair",
        "errors_after_repair",
        "trace_length_after_repair",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_summaries)


def _task_filename(task_id: str) -> str:
    if task_id == "gear_assembly":
        return "gear_assembly_task.json"
    return "packaging_task.json"


def _print_trial_table(trial_summaries: list[dict]) -> None:
    headers = [
        "Trial",
        "Task",
        "Mode",
        "Raw Valid",
        "Raw Success",
        "Raw Errors",
        "Repaired Valid",
        "Repaired Success",
        "Repaired Errors",
    ]
    rows = [
        [
            str(item["trial"]),
            item.get("task_id", ""),
            item["mode_used"],
            _yes_no(item["valid_before_repair"]),
            _yes_no(item["success_before_repair"]),
            str(item["errors_before_repair"]),
            _yes_no(item["valid_after_repair"]),
            _yes_no(item["success_after_repair"]),
            str(item["errors_after_repair"]),
        ]
        for item in trial_summaries
    ]
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def _print_aggregate_summary(summary: dict) -> None:
    print()
    print("Aggregate summary")
    print("=" * 18)
    print(f"Raw valid rate: {summary['raw_valid_rate']:.2f}")
    print(f"Raw success rate: {summary['raw_success_rate']:.2f}")
    print(f"Repaired valid rate: {summary['repaired_valid_rate']:.2f}")
    print(f"Repaired success rate: {summary['repaired_success_rate']:.2f}")
    print(f"Average errors before repair: {summary['average_errors_before_repair']:.2f}")
    print(f"Average errors after repair: {summary['average_errors_after_repair']:.2f}")


def _format_row(row: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def _average(values) -> float:
    value_list = list(values)
    if not value_list:
        return 0.0
    return sum(value_list) / len(value_list)


def _safe_error_message(error: object) -> str:
    message = str(error)
    message = message.replace("\r", " ").replace("\n", " ")
    message = re.sub(r"sk-proj-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    return message


if __name__ == "__main__":
    main()
