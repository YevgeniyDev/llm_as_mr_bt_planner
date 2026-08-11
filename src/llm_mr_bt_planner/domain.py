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

Only the explicit effect form is accepted by the active schema and parser.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .predicates import (
    canonical_predicate,
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
    parameter_types: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    action_type: str = "task"
    duration_ticks: int = 1
    timeout_ticks: int = 20


@dataclass(frozen=True)
class Entity:
    id: str
    type: str


@dataclass(frozen=True)
class Resource:
    id: str
    capacity: int = 1


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
    schema_version: str = "1.0"
    entities: tuple[Entity, ...] = ()
    resources: tuple[Resource, ...] = ()

    def robot(self, robot_id: str) -> Robot | None:
        return next((robot for robot in self.robots if robot.id == robot_id), None)

    @property
    def robot_ids(self) -> set[str]:
        return {robot.id for robot in self.robots}

    @property
    def constants(self) -> set[str]:
        return set(self.objects) | set(self.locations) | {entity.id for entity in self.entities} | self.robot_ids

    @property
    def resource_ids(self) -> set[str]:
        return {resource.id for resource in self.resources}

    def constant_type(self, constant: str) -> str | None:
        robot = self.robot(constant)
        if robot is not None:
            return robot.type
        entity = next((entity for entity in self.entities if entity.id == constant), None)
        return entity.type if entity else None

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


SCENARIO_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "scenario.schema.json"


def load_scenario(path: str | Path, *, strict: bool | None = None) -> Scenario:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    use_strict = "schema_version" in data if strict is None else strict
    return parse_scenario(data, strict=use_strict)


def parse_scenario(data: dict[str, Any], *, strict: bool = False) -> Scenario:
    if not isinstance(data, dict):
        raise ScenarioError("Scenario JSON root must be an object.")
    if strict:
        _validate_json_schema(data)
    _require(data, ["task_id", "instruction", "initial_state", "goal_state", "robots"])
    robots = tuple(_parse_robot(robot) for robot in data.get("robots", []))
    if not robots:
        raise ScenarioError("Scenario must define at least one robot.")
    entities = tuple(
        Entity(id=str(entity["id"]), type=str(entity["type"]))
        for entity in data.get("entities", [])
        if isinstance(entity, dict) and "id" in entity and "type" in entity
    )
    resources = tuple(
        Resource(id=str(resource["id"]), capacity=int(resource.get("capacity", 1)))
        for resource in data.get("resources", [])
        if isinstance(resource, dict) and "id" in resource
    )
    derived_objects = tuple(
        entity.id for entity in entities if entity.type in {"part", "carrier", "tool", "object"}
    )
    derived_locations = tuple(
        entity.id for entity in entities if entity.type in {"location", "dock", "zone", "route", "fixture"}
    )
    scenario = Scenario(
        task_id=str(data["task_id"]),
        instruction=str(data["instruction"]),
        initial_state=tuple(canonical_predicate(item) for item in data.get("initial_state", [])),
        goal_state=tuple(canonical_predicate(item) for item in data.get("goal_state", [])),
        objects=tuple(data.get("objects", derived_objects)),
        locations=tuple(data.get("locations", derived_locations)),
        robots=robots,
        schema_version=str(data.get("schema_version", "legacy")),
        entities=entities,
        resources=resources,
    )
    _validate_scenario_semantics(scenario)
    return scenario


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Return the canonical, upload-compatible scenario document."""
    return {
        "schema_version": scenario.schema_version if scenario.schema_version != "legacy" else "1.0",
        "task_id": scenario.task_id,
        "instruction": scenario.instruction,
        "initial_state": list(scenario.initial_state),
        "goal_state": list(scenario.goal_state),
        "entities": [{"id": entity.id, "type": entity.type} for entity in scenario.entities],
        "resources": [{"id": resource.id, "capacity": resource.capacity} for resource in scenario.resources],
        "robots": [
            {
                "id": robot.id,
                "name": robot.name,
                "type": robot.type,
                "capabilities": [
                    {
                        "name": capability.name,
                        "parameters": list(capability.parameters),
                        "parameter_types": list(capability.parameter_types),
                        "resources": list(capability.resources),
                        "action_type": capability.action_type,
                        "duration_ticks": capability.duration_ticks,
                        "timeout_ticks": capability.timeout_ticks,
                        "preconditions": list(capability.preconditions),
                        "effects": {
                            "add": list(capability.effects.add),
                            "delete": list(capability.effects.delete),
                        },
                    }
                    for capability in robot.capabilities
                ],
            }
            for robot in scenario.robots
        ],
    }


def _validate_json_schema(data: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:  # pragma: no cover - declared runtime dependency
        raise ScenarioError("Strict scenario validation requires the 'jsonschema' package.") from error
    if not SCENARIO_SCHEMA_PATH.exists():
        raise ScenarioError(f"Bundled scenario schema is missing: {SCENARIO_SCHEMA_PATH}")
    schema = json.loads(SCENARIO_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda item: list(item.path))
    if errors:
        rendered = []
        for schema_error in errors[:12]:
            path = ".".join(str(part) for part in schema_error.absolute_path) or "$"
            rendered.append(f"{path}: {schema_error.message}")
        suffix = f" (+{len(errors) - 12} more)" if len(errors) > 12 else ""
        raise ScenarioError("Scenario JSON Schema validation failed: " + "; ".join(rendered) + suffix)


def _validate_scenario_semantics(scenario: Scenario) -> None:
    robot_ids = [robot.id for robot in scenario.robots]
    if len(robot_ids) != len(set(robot_ids)):
        raise ScenarioError("Robot ids must be unique.")
    constants = scenario.constants
    entity_ids = [entity.id for entity in scenario.entities]
    if len(entity_ids) != len(set(entity_ids)):
        raise ScenarioError("Entity ids must be unique.")
    overlap = set(entity_ids) & scenario.robot_ids
    if overlap:
        raise ScenarioError(f"Entity and robot ids overlap: {', '.join(sorted(overlap))}.")
    resource_ids = [resource.id for resource in scenario.resources]
    if len(resource_ids) != len(set(resource_ids)):
        raise ScenarioError("Resource ids must be unique.")
    if set(resource_ids) - constants:
        raise ScenarioError("Every resource id must name a declared entity.")
    arities: dict[str, int] = {}
    full_literals: list[tuple[str, str]] = [
        *((literal, "state") for literal in (*scenario.initial_state, *scenario.goal_state)),
        *(
            (literal, f"{robot.id}.{capability.name}")
            for robot in scenario.robots
            for capability in robot.capabilities
            for literal in (*capability.preconditions, *capability.effects.add)
        ),
    ]
    for literal, where in full_literals:
        name, args = parse_predicate(literal)
        previous = arities.setdefault(name, len(args))
        if previous != len(args):
            raise ScenarioError(
                f"Predicate '{name}' has inconsistent arity: expected {previous}, got {len(args)} in {where}."
            )

    def check_literal(
        literal: str,
        allowed_variables: set[str],
        where: str,
        *,
        allow_partial: bool = False,
    ) -> None:
        try:
            name, args = parse_predicate(literal)
        except (TypeError, ValueError) as error:
            raise ScenarioError(f"Invalid predicate '{literal}' in {where}: {error}") from error
        previous = arities.setdefault(name, len(args))
        if previous != len(args) and not (allow_partial and len(args) <= previous):
            raise ScenarioError(
                f"Predicate '{name}' has inconsistent arity: expected {previous}, got {len(args)} in {where}."
            )
        unknown = [arg for arg in args if arg not in constants and arg not in allowed_variables]
        if unknown:
            raise ScenarioError(
                f"Predicate '{literal}' in {where} uses unknown constant/parameter(s): {', '.join(unknown)}."
            )

    for index, literal in enumerate((*scenario.initial_state, *scenario.goal_state)):
        check_literal(literal, set(), f"state[{index}]")
    for robot in scenario.robots:
        names = [capability.name for capability in robot.capabilities]
        if len(names) != len(set(names)):
            raise ScenarioError(f"Robot '{robot.id}' capability names must be unique.")
        for capability in robot.capabilities:
            variables = set(capability.parameters)
            if len(variables) != len(capability.parameters):
                raise ScenarioError(f"Capability '{robot.id}.{capability.name}' parameters must be unique.")
            if capability.parameter_types and len(capability.parameter_types) != len(capability.parameters):
                raise ScenarioError(
                    f"Capability '{robot.id}.{capability.name}' parameter_types must match parameters."
                )
            unknown_resources = set(capability.resources) - scenario.resource_ids
            if unknown_resources:
                raise ScenarioError(
                    f"Capability '{robot.id}.{capability.name}' uses undeclared resources: "
                    f"{', '.join(sorted(unknown_resources))}."
                )
            if capability.duration_ticks <= 0 or capability.timeout_ticks <= 0:
                raise ScenarioError(f"Capability '{robot.id}.{capability.name}' durations/timeouts must be positive.")
            if capability.duration_ticks > capability.timeout_ticks:
                raise ScenarioError(
                    f"Capability '{robot.id}.{capability.name}' duration_ticks exceeds timeout_ticks."
                )
            for literal in capability.preconditions:
                check_literal(literal, variables, f"{robot.id}.{capability.name}.preconditions")
            for literal in capability.effects.add:
                check_literal(literal, variables, f"{robot.id}.{capability.name}.effects")
            for literal in capability.effects.delete:
                check_literal(
                    literal,
                    variables,
                    f"{robot.id}.{capability.name}.effects.delete",
                    allow_partial=True,
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
    effects = normalize_effects(data.get("effects", {}), robot_id, data["name"])
    return Capability(
        name=str(data["name"]),
        parameters=tuple(data.get("parameters", [])),
        preconditions=tuple(canonical_predicate(item) for item in data.get("preconditions", [])),
        effects=Effects(
            add=tuple(canonical_predicate(item) for item in effects.add),
            delete=tuple(canonical_predicate(item) for item in effects.delete),
        ),
        parameter_types=tuple(data.get("parameter_types", [])),
        resources=tuple(data.get("resources", [])),
        action_type=str(data.get("action_type", "task")),
        duration_ticks=int(data.get("duration_ticks", 1)),
        timeout_ticks=int(data.get("timeout_ticks", 20)),
    )


def normalize_effects(raw: Any, robot_id: str, capability: str) -> Effects:
    """Parse the explicit ``{"add": [...], "delete": [...]}`` effect contract."""
    if isinstance(raw, dict):
        return Effects(add=tuple(raw.get("add", [])), delete=tuple(raw.get("delete", [])))
    raise ScenarioError(
        f"Capability '{robot_id}.{capability}' effects must be an add/delete object."
    )


def _require(data: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ScenarioError(f"Scenario is missing required field(s): {', '.join(missing)}.")
