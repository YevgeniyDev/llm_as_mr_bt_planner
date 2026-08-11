from __future__ import annotations

import pytest

from llm_mr_bt_planner.domain import (
    Effects,
    ScenarioError,
    apply_grounded,
    ground_effects,
    normalize_effects,
    parse_scenario,
)


def test_explicit_effects_parsed():
    effects = normalize_effects({"add": ["p(x)"], "delete": ["q(x)"]}, "r", "cap")
    assert effects == Effects(add=("p(x)",), delete=("q(x)",))


def test_legacy_effect_list_is_rejected():
    with pytest.raises(ScenarioError, match="add/delete object"):
        normalize_effects(["holding(r, x)"], "r", "cap")


def test_apply_grounded_functional_fluent_replaces_location():
    state = {"robot_at(r, a)"}
    adds, deletes = ground_effects(Effects(add=("robot_at(r, b)",), delete=("robot_at(r)",)), {})
    apply_grounded(state, adds, deletes)
    assert state == {"robot_at(r, b)"}


def test_apply_grounded_deletes_run_before_adds():
    state = {"p(x, old)"}
    apply_grounded(state, ["p(x, new)"], ["p(x)"])
    assert state == {"p(x, new)"}


def test_parse_scenario_requires_fields():
    with pytest.raises(ScenarioError):
        parse_scenario({"task_id": "t"})


def test_scenario_helpers(courier_scenario):
    assert courier_scenario.robot("unitree_go2_z1") is not None
    assert courier_scenario.robot_ids == {"franka_a", "unitree_go2_z1", "franka_b"}
    assert "source_dock" in courier_scenario.constants
    capability = courier_scenario.capability("unitree_go2_z1", "navigate_destination")
    assert capability is not None
    assert capability.effects.add == ("docked(unitree_go2_z1, destination_dock)",)
