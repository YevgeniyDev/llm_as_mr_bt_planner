"""Symbolic predicate utilities.

A *predicate* (or *fact*) is written ``name(arg1, arg2, ...)``. Arguments are
either scenario constants (objects, locations, robot ids) or, inside capability
templates, variables drawn from the capability's ``parameters`` list.

This module is the declarative core of the planner. It deliberately contains
*no* naming-convention magic (the old prototype special-cased ``_open`` /
``_closed`` / ``_at`` suffixes and a hard-coded ``robot_near`` derivation).
World-state semantics are expressed explicitly by the domain instead - see
:mod:`llm_mr_bt_planner.domain`.
"""

from __future__ import annotations

import re
from typing import Iterable

WILDCARD = "_"


def parse_predicate(predicate: str | None) -> tuple[str, list[str]]:
    """Split ``name(a, b)`` into ``("name", ["a", "b"])``.

    A bare token with no parentheses parses to ``(token, [])``. Empty argument
    slots are dropped so ``name()`` yields no arguments.
    """
    if not predicate or "(" not in predicate or not predicate.endswith(")"):
        return str(predicate or ""), []
    name, raw_args = predicate.split("(", 1)
    args = [arg.strip() for arg in raw_args[:-1].split(",") if arg.strip()]
    return name.strip(), args


def format_predicate(name: str, parameters: Iterable[str]) -> str:
    """Inverse of :func:`parse_predicate`."""
    return f"{name}({', '.join(parameters)})"


def substitute(template: str, bindings: dict[str, str]) -> str:
    """Replace whole-word variable occurrences in ``template`` using ``bindings``.

    Word boundaries prevent ``tool`` from matching inside ``tool_zone``.
    """
    result = template
    for variable, value in bindings.items():
        result = re.sub(rf"\b{re.escape(variable)}\b", value, result)
    return result


def matches_pattern(fact: str, pattern: str) -> bool:
    """Return ``True`` when ``fact`` matches a (possibly partial) ``pattern``.

    Matching rules, applied after both sides are parsed:

    * predicate names must be equal;
    * the pattern may supply *fewer* arguments than the fact - the missing
      trailing positions act as wildcards (prefix match);
    * a pattern argument equal to :data:`WILDCARD` (``"_"``) matches anything;
    * every other pattern argument must equal the fact argument at that index.

    This single rule subsumes both exact deletes (``holding(go2, tray)``) and
    "remove whatever value this fluent currently has" deletes (``tray_at(tray)``
    or ``tray_at(tray, _)``).
    """
    fact_name, fact_args = parse_predicate(fact)
    pattern_name, pattern_args = parse_predicate(pattern)
    if fact_name != pattern_name:
        return False
    if len(pattern_args) > len(fact_args):
        return False
    for pattern_arg, fact_arg in zip(pattern_args, fact_args):
        if pattern_arg != WILDCARD and pattern_arg != fact_arg:
            return False
    return True


def unify_effect_args(
    effect_args: list[str],
    target_args: list[str],
    action_parameters: Iterable[str],
    constants: set[str],
) -> dict[str, str] | None:
    """Try to bind a capability effect's arguments to a concrete target.

    Returns the variable bindings that make ``effect_args`` equal to
    ``target_args``, or ``None`` if no consistent binding exists. Used when
    searching for capabilities that can *produce* a wanted predicate.
    """
    if len(effect_args) != len(target_args):
        return None

    action_parameter_set = set(action_parameters)
    bindings: dict[str, str] = {}
    for effect_arg, target_arg in zip(effect_args, target_args):
        if effect_arg in constants:
            if effect_arg != target_arg:
                return None
        elif effect_arg in action_parameter_set:
            if target_arg not in constants and target_arg != effect_arg:
                return None
            existing = bindings.get(effect_arg)
            if existing is not None and existing != target_arg:
                return None
            bindings[effect_arg] = target_arg
        elif effect_arg != target_arg:
            return None
    return bindings
