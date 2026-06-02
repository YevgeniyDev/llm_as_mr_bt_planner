from __future__ import annotations

import json
from pathlib import Path


def aggregate_error_types(trial_dirs: list[Path]) -> dict:
    """Count validator error types across LLM trial directories."""
    counts: dict[str, int] = {}
    num_trials = len(trial_dirs)

    for trial_dir in trial_dirs:
        validation_path = trial_dir / "validation_result.json"
        if not validation_path.exists():
            continue

        with validation_path.open("r", encoding="utf-8") as file:
            validation_result = json.load(file)

        for error in validation_result.get("errors", []):
            error_type = error.get("type", "unknown_error")
            counts[error_type] = counts.get(error_type, 0) + 1

    return {
        error_type: {
            "count": count,
            "average_per_trial": count / num_trials if num_trials else 0.0,
        }
        for error_type, count in sorted(counts.items())
    }
