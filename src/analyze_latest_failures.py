from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    trial_root = project_root / args.output_dir

    if not trial_root.exists():
        print(f"No trial directory found at {trial_root}.")
        print("Run python src/run_llm_trials.py --trials 5 --mode openai --task gear_assembly first.")
        return

    trial_dirs = sorted(
        path
        for path in trial_root.iterdir()
        if path.is_dir() and path.name.startswith("trial_")
    )
    if not trial_dirs:
        print(f"No trial_XX folders found in {trial_root}.")
        return

    raw_counts, raw_examples = _collect_errors(trial_dirs, "validation_result.json")
    repaired_counts, repaired_examples = _collect_errors(trial_dirs, "repaired_validation_result.json")

    print(f"Failure analysis for task: {args.task}")
    print(f"Trial directory: {trial_root}")
    print()
    _print_error_section("Common raw error types", raw_counts, raw_examples)
    print()
    _print_error_section("Common repaired error types", repaired_counts, repaired_examples)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze latest LLM trial validation failures.")
    parser.add_argument(
        "--task",
        choices=["packaging_delivery", "gear_assembly"],
        default="gear_assembly",
        help="Task label to print in the report.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/llm_trials",
        help="Trial output directory to analyze.",
    )
    return parser.parse_args()


def _collect_errors(trial_dirs: list[Path], filename: str) -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for trial_dir in trial_dirs:
        path = trial_dir / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            result = json.load(file)

        for error in result.get("errors", []):
            error_type = error.get("type", "unknown_error")
            counts[error_type] += 1
            message = error.get("message", "")
            if message and len(examples[error_type]) < 3:
                examples[error_type].append(message)

    return counts, examples


def _print_error_section(title: str, counts: Counter, examples: dict[str, list[str]]) -> None:
    print(title)
    print("-" * len(title))
    if not counts:
        print("No errors found.")
        return

    for error_type, count in counts.most_common():
        print(f"{error_type}: {count}")
        for message in examples.get(error_type, []):
            print(f"  - {message}")


if __name__ == "__main__":
    main()
