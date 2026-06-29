"""Scenario domain model: typed dataclasses, loading, and world-state semantics.

The domain is fully declarative. A capability's effects are an explicit
``add`` / ``delete`` pair (PDDL-style), where delete literals may be partial
patterns (see :func:`llm_mr_bt_planner.predicates.matches_pattern`). This removes the hidden
naming conventions of the original prototype:

* ``open_drawer`` now explicitly ``delete``\\s ``drawer_closed(drawer)`` instead
  of relying on a ``_open`` / ``_closed`` suffix rule;
* single-valued ("functional") fluents such as ``tray_at(tray, location)`` carry
  an explicit prefix delete ``tray_at(tray)`` instead of relying on a ``_at``
  suffix rule.

Legacy scenarios that still use a flat list of string effects (with the
``not_`` negation convention) are converted on load by :func:`normalize_effects`,
which emits a :class:`DeprecationWarning` so the conversion is never silent.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .predicates import (
    format_predicate,
    matches_pattern,
    parse_predicate,
    substitute,
    unify_effect_args,
)


@dataclass(frozen=True)
class Effects:
    """The add/delete lists a capability applies, as predicate templates."""

    add: tuple[str, ...] = ()
    delete: tuple[str, ...] = ()


@dataclass(frozen=True)
class Capability:
    name: str
    parameters: tuple[str, ...]
    preconditions: tuple[str, ...]
    effects: Effects


@dataclass(frozen=True)
class Robot:
    id: str
    name: str
    type: str
    capabilities: tuple[Capability, ...]

    def capability(self, name: str) -> Capability | None:
        return next((cap for cap in self.capabilities if cap.name == name), None)

    @property
    def capability_names(self) -> set[str]:
        return {cap.name for cap in self.capabilities}


@dataclass(frozen=True)
class Scenario:
    task_id: str
    instruction: str
    initial_state: tuple[str, ...]
    goal_state: tuple[str, ...]
    objects: tuple[str, ...]
    locations: tuple[str, ...]
    robots: tuple[Robot, ...]

    def robot(self, robot_id: str) -> Robot | None:
        return next((robot for robot in self.robots if robot.id == robot_id), None)

    @property
    def robot_ids(self) -> set[str]:
        return {robot.id for robot in self.robots}

    @property
    def constants(self) -> set[str]:
        return set(self.objects) | set(self.locations) | self.robot_ids

    def capability(self, robot_id: str, action: str) -> Capability | None:
        robot = self.robot(robot_id)
        return robot.capability(action) if robot else None


# --------------------------------------------------------------------------- #
# World-state semantics
# --------------------------------------------------------------------------- #


def ground_effects(effects: Effects, bindings: dict[str, str]) -> tuple[list[str], list[str]]:
    """Substitute ``bindings`` into an effect template, returning (adds, deletes)."""
    adds = [substitute(literal, bindings) for literal in effects.add]
    deletes = [substitute(pattern, bindings) for pattern in effects.delete]
    return adds, deletes


def apply_grounded(state: set[str], adds: list[str], deletes: list[str]) -> None:
    """Mutate ``state`` in place: remove every fact matching a delete pattern,
    then add the positive facts. Deletes run first so an add is never clobbered.
    """
    for pattern in deletes:
        state.difference_update({fact for fact in state if matches_pattern(fact, pattern)})
    state.update(adds)


def positive_effects(effects: Effects, bindings: dict[str, str]) -> list[str]:
    """The grounded add-list only - used to compute what a plan can *produce*."""
    return [substitute(literal, bindings) for literal in effects.add]


# --------------------------------------------------------------------------- #
# Producer search (used by prompts and validation)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProducerSpec:
    robot: str
    action: str
    parameters: tuple[str, ...]

    def describe(self) -> str:
        return f"robot {self.robot} action {format_predicate(self.action, self.parameters)}"


def candidate_producers(predicate: str, scenario: Scenario) -> list[ProducerSpec]:
    """Find capabilities whose add-effects can be instantiated to produce ``predicate``.

    A capability is excluded when producing the predicate would itself require a
    goal predicate as a precondition (avoids suggesting circular producers),
    unless the predicate is itself a goal.
    """
    target_name, target_args = parse_predicate(predicate)
    goals = set(scenario.goal_state)
    constants = scenario.constants
    specs: list[ProducerSpec] = []
    for robot in scenario.robots:
        for capability in robot.capabilities:
            for effect in capability.effects.add:
                effect_name, effect_args = parse_predicate(effect)
                if effect_name != target_name:
                    continue
                bindings = unify_effect_args(effect_args, target_args, capability.parameters, constants)
                if bindings is None:
                    continue
                if predicate not in goals and _requires_goal(capability, bindings, goals):
                    continue
                parameters = tuple(bindings.get(param, param) for param in capability.parameters)
                specs.append(ProducerSpec(robot.id, capability.name, parameters))
    return specs[:5]


def _requires_goal(capability: Capability, bindings: dict[str, str], goals: set[str]) -> bool:
    return any(substitute(pre, bindings) in goals for pre in capability.preconditions)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class ScenarioError(ValueError):
    """Raised when a scenario file is structurally invalid."""


def load_scenario(path: str | Path) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_scenario(data)


def parse_scenario(data: dict[str, Any]) -> Scenario:
    _require(data, ["task_id", "instruction", "initial_state", "goal_state", "robots"])
    robots = tuple(_parse_robot(robot) for robot in data.get("robots", []))
    if not robots:
        raise ScenarioError("Scenario must define at least one robot.")
    return Scenario(
        task_id=str(data["task_id"]),
        instruction=str(data["instruction"]),
        initial_state=tuple(data.get("initial_state", [])),
        goal_state=tuple(data.get("goal_state", [])),
        objects=tuple(data.get("objects", [])),
        locations=tuple(data.get("locations", [])),
        robots=robots,
    )


def _parse_robot(data: dict[str, Any]) -> Robot:
    if "id" not in data:
        raise ScenarioError("Each robot needs an 'id'.")
    capabilities = tuple(_parse_capability(cap, data["id"]) for cap in data.get("capabilities", []))
    return Robot(
        id=str(data["id"]),
        name=str(data.get("name", data["id"])),
        type=str(data.get("type", "robot")),
        capabilities=capabilities,
    )


def _parse_capability(data: dict[str, Any], robot_id: str) -> Capability:
    if "name" not in data:
        raise ScenarioError(f"Robot '{robot_id}' has a capability without a 'name'.")
    return Capability(
        name=str(data["name"]),
        parameters=tuple(data.get("parameters", [])),
        preconditions=tuple(data.get("preconditions", [])),
        effects=normalize_effects(data.get("effects", {}), robot_id, data["name"]),
    )


def normalize_effects(raw: Any, robot_id: str, capability: str) -> Effects:
    """Accept either the explicit ``{"add": [...], "delete": [...]}`` form or the
    legacy flat list (with the ``not_`` negation convention) and return
    :class:`Effects`.
    """
    if isinstance(raw, dict):
        return Effects(add=tuple(raw.get("add", [])), delete=tuple(raw.get("delete", [])))
    if isinstance(raw, list):
        return _convert_legacy_effects(raw, robot_id, capability)
    raise ScenarioError(
        f"Capability '{robot_id}.{capability}' effects must be a list or an add/delete object."
    )


def _convert_legacy_effects(effects: list[str], robot_id: str, capability: str) -> Effects:
    warnings.warn(
        f"Capability '{robot_id}.{capability}' uses legacy list-style effects with the "
        "'not_' negation convention. Convert it to the explicit {'add': [...], 'delete': [...]} "
        "form; the implicit conversion will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )
    adds: list[str] = []
    deletes: list[str] = []
    for effect in effects:
        name, args = parse_predicate(effect)
        if name.startswith("not_"):
            deletes.append(format_predicate(name.removeprefix("not_"), args))
        else:
            adds.append(effect)
    return Effects(add=tuple(adds), delete=tuple(deletes))


def _require(data: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ScenarioError(f"Scenario is missing required field(s): {', '.join(missing)}.")
