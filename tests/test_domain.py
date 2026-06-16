from __future__ import annotations

import warnings

import pytest

from mrbtp.domain import (
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


def test_legacy_effects_convert_with_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        effects = normalize_effects(["holding(r, x)", "not_at(x)"], "r", "cap")
    assert effects.add == ("holding(r, x)",)
    assert effects.delete == ("at(x)",)


def test_legacy_and_explicit_effects_are_equivalent():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = normalize_effects(["tool_at(t, loc)", "not_holding(r, t)"], "r", "place")
    # The explicit form must add a functional prefix-delete to match legacy '_at' semantics.
    explicit = Effects(add=("tool_at(t, loc)",), delete=("tool_at(t)", "holding(r, t)"))

    state_legacy = {"tool_at(t, old)", "holding(r, t)"}
    state_explicit = set(state_legacy)
    apply_grounded(state_legacy, *ground_effects(Effects(add=legacy.add, delete=("tool_at(t)", *legacy.delete)), {}))
    apply_grounded(state_explicit, *ground_effects(explicit, {}))
    assert state_legacy == state_explicit == {"tool_at(t, loc)"}


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


def test_scenario_helpers(gear_scenario):
    assert gear_scenario.robot("go2_z1") is not None
    assert "go2_z1" in gear_scenario.robot_ids
    assert "parts_drawer" in gear_scenario.constants
    cap = gear_scenario.capability("go2_z1", "open_drawer")
    assert cap is not None and cap.effects.add == ("drawer_open(drawer)",)
