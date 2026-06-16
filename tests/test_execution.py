from __future__ import annotations

import pytest

from mrbtp.execution import SymbolicExecutionBackend, get_backend


def test_symbolic_backend_runs_a_correct_plan(toy_scenario, toy_plan):
    result = SymbolicExecutionBackend().execute(toy_plan, toy_scenario)
    assert result.backend == "symbolic"
    assert result.success and result.goal_success


def test_get_backend_unknown():
    with pytest.raises(ValueError):
        get_backend("warp_drive")


def test_ros_backend_is_an_actionable_scaffold(toy_plan, toy_scenario):
    backend = get_backend("ros")
    with pytest.raises(NotImplementedError) as info:
        backend.execute(toy_plan, toy_scenario)
    assert "export_behaviortree_cpp_xml" in str(info.value)
