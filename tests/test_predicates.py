from __future__ import annotations

from mrbtp.predicates import (
    format_predicate,
    matches_pattern,
    parse_predicate,
    substitute,
    unify_effect_args,
)


def test_parse_and_format_roundtrip():
    assert parse_predicate("at(robot, zone)") == ("at", ["robot", "zone"])
    assert format_predicate("at", ["robot", "zone"]) == "at(robot, zone)"


def test_parse_bare_token_and_empty_args():
    assert parse_predicate("done") == ("done", [])
    assert parse_predicate("p()") == ("p", [])
    assert parse_predicate(None) == ("", [])


def test_substitute_respects_word_boundaries():
    # 'tool' must not be replaced inside 'tool_zone'
    assert substitute("at(tool, tool_zone)", {"tool": "screwdriver"}) == "at(screwdriver, tool_zone)"


def test_matches_pattern_exact_and_prefix_and_wildcard():
    assert matches_pattern("tray_at(t, parts_zone)", "tray_at(t, parts_zone)")
    assert matches_pattern("tray_at(t, parts_zone)", "tray_at(t)")  # prefix delete
    assert matches_pattern("tray_at(t, parts_zone)", "tray_at(_, parts_zone)")  # wildcard
    assert not matches_pattern("tray_at(t, parts_zone)", "tray_at(t, tool_zone)")
    assert not matches_pattern("tray_at(t, parts_zone)", "other(t)")
    # pattern cannot have more args than the fact
    assert not matches_pattern("tray_at(t)", "tray_at(t, parts_zone)")


def test_unify_effect_args():
    constants = {"tray", "parts_zone", "tool_zone"}
    # variable 'x' binds to constant target
    assert unify_effect_args(["x", "parts_zone"], ["tray", "parts_zone"], ["x"], constants) == {"x": "tray"}
    # constant mismatch fails
    assert unify_effect_args(["x", "parts_zone"], ["tray", "tool_zone"], ["x"], constants) is None
    # arity mismatch fails
    assert unify_effect_args(["x"], ["tray", "parts_zone"], ["x"], constants) is None
