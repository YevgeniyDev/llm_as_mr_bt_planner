from __future__ import annotations


def compute_trial_metrics(validation_result: dict, simulation_result: dict, goal_state: list[str] | None = None) -> dict:
    if goal_state is None:
        goal_success = bool(simulation_result.get("success"))
    else:
        final_state = set(simulation_result.get("final_state", []))
        goal_success = all(predicate in final_state for predicate in goal_state)

    return {
        "valid_bt": bool(validation_result.get("valid")),
        "goal_success": goal_success,
        "num_validation_errors": len(validation_result.get("errors", [])),
        "num_simulation_errors": len(simulation_result.get("errors", [])),
        "trace_length": len(simulation_result.get("trace", [])),
    }
