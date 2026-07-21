from pathlib import Path

from scripts.run_experiment_matrix import validate_matrix


def test_fixed_protocol_is_complete_and_unique():
    summary = validate_matrix(Path("experiments/protocol_v1.json"))
    assert summary["protocol_version"] == "llm-mr-bt-eval-v1"
    assert summary["conditions"] == 8
    assert summary["scenarios"] == 2
    assert summary["planned_llm_trials"] == 420
    assert summary["planned_native_trials"] == 2
