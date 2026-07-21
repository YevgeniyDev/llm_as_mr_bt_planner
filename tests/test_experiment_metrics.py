from llm_mr_bt_planner.experiments.report import aggregate, to_latex_tables
from llm_mr_bt_planner.experiments.runner import _record_from_result
from llm_mr_bt_planner.planner import PlannerResult


def _result():
    return PlannerResult(
        task_id="toy",
        provider="stub",
        model="stub-1",
        valid=False,
        success=False,
        goal_success=False,
        correction_rounds=2,
        plan={
            "behavior_trees": {
                "A": {"type": "Sequence", "children": [
                    {"type": "Condition", "name": "ready", "parameters": []},
                    {"type": "Action", "name": "act", "parameters": []},
                ]}
            },
            "synchronization": [{"condition": "ready()", "producer": "B", "consumer": "A"}],
        },
        validation_errors=[
            {"type": "missing_sync_producer", "message": "missing"},
            {"type": "invalid_capability", "message": "bad"},
        ],
        simulation={
            "trace": [{"tick": 3, "event": "action", "name": "act"}],
            "errors": [{"type": "deadlock"}],
            "final_state": [],
        },
        wall_seconds=1.25,
    )


def test_trial_classifies_sync_and_complexity_metrics():
    record = _record_from_result(
        _result(), scenario="toy", trial=1, method="proposed",
        condition="pure", seed=7, temperature=0.7,
    )
    assert record.synchronization_errors == 1
    assert record.capability_errors == 1
    assert record.causal_errors == 1
    assert record.deadlock
    assert record.plan_nodes == 3
    assert record.action_nodes == 1
    assert record.condition_nodes == 1
    assert record.synchronization_edges == 1
    assert record.makespan_ticks == 3


def test_aggregate_contains_rates_intervals_and_complexity():
    record = _record_from_result(
        _result(), scenario="toy", trial=1, method="proposed",
        condition="pure", seed=7, temperature=0.7,
    )
    row = aggregate([record])[0]
    assert row["synchronization_error_rate"] == 1.0
    assert row["capability_error_rate"] == 1.0
    assert row["deadlock_rate"] == 1.0
    assert row["validity_ci95"].startswith("[")
    assert "Main comparison" in to_latex_tables([record])


def test_native_timeout_uses_native_metrics_without_shared_error_classification():
    result = _result()
    result.metric_scope = "native_mrbtp"
    result.native_metrics = {"timed_out": True}
    result.validation_errors = [{"type": "mrbtp_timeout", "message": "timeout"}]
    record = _record_from_result(
        result, scenario="toy", trial=1, method="mrbtp",
        condition="mrbtp_native", seed=None, temperature=None,
    )
    assert record.timeout
    assert record.structural_errors == 0
    assert record.synchronization_errors == 0
