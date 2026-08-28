"""Native goal grammar and reactive-planner adapter for BETR-XP-LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..bt import BTNode, iter_nodes
from ..domain import Scenario
from ..predicates import canonical_predicate, parse_predicate
from .llm_bt_native import (
    ExpansionResult,
    GroundAction,
    ParsedGoal,
    expand_initial_trees,
    ground_action_templates,
)

_TOKEN = re.compile(r"\s*([A-Za-z][A-Za-z0-9_]*|[&|~()])")


class BetrXPNativeError(ValueError):
    """Raised when a native formal goal or parameter update is not admissible."""


@dataclass(frozen=True)
class EntityAlias:
    native: str
    common: str
    entity_type: str

    def to_dict(self) -> dict[str, str]:
        return {"native": self.native, "common": self.common, "type": self.entity_type}


@dataclass(frozen=True)
class ConditionSchema:
    native: str
    common: str
    argument_types: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "native": self.native,
            "common": self.common,
            "argument_types": list(self.argument_types),
            "description": self.description,
        }


@dataclass(frozen=True)
class FormalLiteral:
    native: str
    predicate: str
    negated: bool

    def to_dict(self) -> dict[str, Any]:
        return {"native": self.native, "predicate": self.predicate, "negated": self.negated}


@dataclass(frozen=True)
class ParsedFormula:
    raw_formula: str
    alternatives: tuple[tuple[FormalLiteral, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_formula": self.raw_formula,
            "alternatives": [
                [literal.to_dict() for literal in alternative]
                for alternative in self.alternatives
            ],
        }


@dataclass(frozen=True)
class PlannedAlternative:
    index: int
    literals: tuple[FormalLiteral, ...]
    parsed_goals: tuple[ParsedGoal, ...]
    expansion: ExpansionResult
    estimated_cost: int


def build_entity_aliases(scenario: Scenario) -> list[EntityAlias]:
    aliases: list[EntityAlias] = []
    used: set[str] = set()
    entries = [
        *((entity.id, entity.type) for entity in scenario.entities),
        *((robot.id, "robot") for robot in scenario.robots),
    ]
    for common, entity_type in entries:
        base = _pascal(common)
        native = base
        suffix = 2
        while native.lower() in used:
            native = f"{base}{suffix}"
            suffix += 1
        used.add(native.lower())
        aliases.append(EntityAlias(native, common, entity_type))
    return aliases


def build_condition_schemas(scenario: Scenario) -> list[ConditionSchema]:
    signatures: dict[str, tuple[str, ...]] = {}
    for predicate in (*scenario.initial_state, *scenario.goal_state):
        name, arguments = parse_predicate(predicate)
        inferred = tuple(_constant_type(scenario, argument) for argument in arguments)
        signatures[name] = _merge_types(signatures.get(name), inferred)
    for robot in scenario.robots:
        for capability in robot.capabilities:
            parameter_types = dict(zip(capability.parameters, capability.parameter_types))
            for template in (
                *capability.preconditions,
                *capability.effects.add,
            ):
                name, arguments = parse_predicate(template)
                inferred = tuple(
                    parameter_types.get(argument)
                    or _constant_type(scenario, argument)
                    for argument in arguments
                )
                signatures[name] = _merge_types(signatures.get(name), inferred)
    return [
        ConditionSchema(
            native=_pascal(name),
            common=name,
            argument_types=types,
            description=_condition_description(name, types),
        )
        for name, types in sorted(signatures.items())
    ]


def encode_predicate(
    predicate: str,
    schemas: list[ConditionSchema],
    entities: list[EntityAlias],
) -> str:
    """Encode one common predicate in the paper's underscore-separated goal grammar."""
    name, arguments = parse_predicate(predicate)
    schema = next((item for item in schemas if item.common == name), None)
    if schema is None:
        raise BetrXPNativeError(f"No formal condition alias exists for '{name}'.")
    entity_lookup = {entity.common: entity.native for entity in entities}
    try:
        native_arguments = [entity_lookup[argument] for argument in arguments]
    except KeyError as error:
        raise BetrXPNativeError(
            f"No formal object alias exists for '{error.args[0]}'."
        ) from error
    return "_".join([schema.native, *native_arguments])


def parse_goal_response(
    response: str,
    schemas: list[ConditionSchema],
    entities: list[EntityAlias],
) -> ParsedFormula:
    """Parse the released ``Goal:`` first-order formula into DNF alternatives."""
    if not isinstance(response, str) or not response.strip():
        raise BetrXPNativeError("BETR-XP-LLM goal response is empty.")
    match = re.search(r"(?:^|\n)\s*Goal\s*:\s*(.+)\s*$", response, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise BetrXPNativeError("BETR-XP-LLM goal response must contain a 'Goal:' formula.")
    formula = match.group(1).strip()
    tokens = _tokenize(formula)
    expression, index = _parse_or(tokens, 0)
    if index != len(tokens):
        raise BetrXPNativeError(f"Unexpected goal token '{tokens[index]}'.")
    native_dnf = _to_dnf(expression)
    schema_lookup = {schema.native.lower(): schema for schema in schemas}
    entity_lookup = {entity.native.lower(): entity for entity in entities}
    alternatives: list[tuple[FormalLiteral, ...]] = []
    for native_alternative in native_dnf:
        literals: list[FormalLiteral] = []
        seen: set[tuple[str, bool]] = set()
        for atom, negated in native_alternative:
            pieces = atom.split("_")
            schema = schema_lookup.get(pieces[0].lower())
            if schema is None:
                raise BetrXPNativeError(f"Unknown formal condition '{pieces[0]}'.")
            arguments = pieces[1:]
            if len(arguments) != len(schema.argument_types):
                raise BetrXPNativeError(
                    f"Condition '{schema.native}' expects {len(schema.argument_types)} argument(s), "
                    f"got {len(arguments)}."
                )
            common_arguments: list[str] = []
            for argument, expected_type in zip(arguments, schema.argument_types):
                entity = entity_lookup.get(argument.lower())
                if entity is None:
                    raise BetrXPNativeError(f"Unknown formal object '{argument}'.")
                if expected_type not in {"constant", entity.entity_type}:
                    raise BetrXPNativeError(
                        f"Object '{argument}' has type '{entity.entity_type}', expected '{expected_type}'."
                    )
                common_arguments.append(entity.common)
            predicate = canonical_predicate(
                f"{schema.common}({','.join(common_arguments)})"
            )
            key = (predicate, negated)
            if key not in seen:
                literals.append(FormalLiteral(atom, predicate, negated))
                seen.add(key)
        alternatives.append(tuple(literals))
    if not alternatives or any(not alternative for alternative in alternatives):
        raise BetrXPNativeError("BETR-XP-LLM goal formula contains an empty alternative.")
    return ParsedFormula(formula, tuple(alternatives))


def plan_formula(scenario: Scenario, formula: ParsedFormula) -> list[PlannedAlternative]:
    """Run the reactive backchaining planner for each native DNF alternative."""
    planned: list[PlannedAlternative] = []
    for index, alternative in enumerate(formula.alternatives):
        if any(literal.negated for literal in alternative):
            raise BetrXPNativeError(
                "The common benchmark has no closed-world negative-goal representation; "
                "a negated BETR-XP-LLM goal cannot be observed without a semantic rewrite."
            )
        planned.append(
            plan_predicates(
                scenario,
                [literal.predicate for literal in alternative],
                alternative_index=index,
                literals=alternative,
            )
        )
    return planned


def plan_predicates(
    scenario: Scenario,
    predicates: list[str],
    *,
    alternative_index: int = 0,
    literals: tuple[FormalLiteral, ...] | None = None,
    grounded_actions: list[GroundAction] | None = None,
) -> PlannedAlternative:
    """Plan a known canonical goal conjunction through the same reactive planner."""
    actions = grounded_actions if grounded_actions is not None else ground_action_templates(scenario)
    goals = _parsed_goals(scenario, predicates, actions)
    ordered = _priority_order(scenario, goals, actions)
    expansion = expand_initial_trees(
        scenario,
        ordered,
        node_namespace="betrxp",
        grounded_actions=actions,
    )
    action_count = sum(
        node.type == "Action"
        for tree in expansion.trees.values()
        for node in iter_nodes(tree)
    )
    resolved_literals = literals or tuple(
        FormalLiteral(predicate, predicate, False) for predicate in predicates
    )
    return PlannedAlternative(
        index=alternative_index,
        literals=resolved_literals,
        parsed_goals=tuple(ordered),
        expansion=expansion,
        estimated_cost=action_count,
    )


def select_lowest_cost(planned: list[PlannedAlternative]) -> PlannedAlternative:
    feasible = [item for item in planned if not item.expansion.unresolved]
    candidates = feasible or planned
    if not candidates:
        raise BetrXPNativeError("BETR-XP-LLM planner produced no goal alternative.")
    return min(candidates, key=lambda item: (item.estimated_cost, item.index))


def native_flat_policy(trees: dict[str, BTNode]) -> dict[str, list[str]]:
    return {robot: _flatten(tree) for robot, tree in trees.items()}


def grounded_skill_library(scenario: Scenario) -> list[dict[str, Any]]:
    return [
        {
            **action.to_dict(),
            "native_skill": _native_skill(action),
        }
        for action in ground_action_templates(scenario)
    ]


def pickup_binding(
    scenario: Scenario,
    *,
    part: str,
    location: str,
) -> GroundAction:
    """Bind the native generic Pick(part, location) skill to a common capability."""
    candidates = [
        action
        for action in ground_action_templates(scenario)
        if action.robot == "unitree_go2_z1"
        and part in action.parameters
        and (
            (location == "source_cradle" and action.name == "pick_source_cradle")
            or (
                location != "source_cradle"
                and action.name == "recover_fallen_part"
                and location in action.parameters
            )
        )
    ]
    if not candidates:
        raise BetrXPNativeError(
            f"No declared common skill binds native Pick({part}, {location})."
        )
    return candidates[0]


def parse_parameter_response(response: str, allowed_values: Iterable[str]) -> tuple[str, str]:
    """Parse the released reasoning/parameter response and enforce the scene vocabulary."""
    if not isinstance(response, str) or not response.strip():
        raise BetrXPNativeError("BETR-XP-LLM parameter-resolution response is empty.")
    reasoning_match = re.search(
        r"Reasoning\s*:\s*(.*?)\s*Parameter value\s*:",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    value_match = re.search(
        r"Parameter value\s*:\s*([^\r\n]+)",
        response,
        flags=re.IGNORECASE,
    )
    if reasoning_match is None or value_match is None:
        raise BetrXPNativeError(
            "BETR-XP-LLM recovery response requires 'Reasoning:' and 'Parameter value:'."
        )
    reasoning = reasoning_match.group(1).strip()
    value = value_match.group(1).strip().strip('"\'` .')
    lookup = {item.lower(): item for item in allowed_values}
    selected = lookup.get(value.lower())
    if selected is None:
        raise BetrXPNativeError(
            f"Resolved parameter '{value}' is not one of the observed scene values."
        )
    return reasoning, selected


def _parsed_goals(
    scenario: Scenario,
    predicates: list[str],
    actions: list[GroundAction],
) -> list[ParsedGoal]:
    result: list[ParsedGoal] = []
    for index, predicate in enumerate(predicates, start=1):
        name, arguments = parse_predicate(predicate)
        producers = [action for action in actions if predicate in action.add_effects]
        argument_owner = next((argument for argument in arguments if argument in scenario.robot_ids), None)
        if argument_owner is not None:
            owner = argument_owner
        elif producers:
            owner = min(
                producers,
                key=lambda action: (action.timeout_ticks, action.robot, action.name, action.parameters),
            ).robot
        elif predicate in scenario.initial_state:
            owner = scenario.robots[0].id
        else:
            raise BetrXPNativeError(
                f"Formal goal '{predicate}' has no initial support or declared skill postcondition."
            )
        result.append(
            ParsedGoal(
                order=index,
                robot=owner,
                predicate=predicate,
                target=name,
                destination="formal_goal",
            )
        )
    return result


def _priority_order(
    scenario: Scenario,
    goals: list[ParsedGoal],
    actions: list[GroundAction],
) -> list[ParsedGoal]:
    """Reproduce the planner's conflict-priority pass for common goal sequences."""
    delete_closures = {
        goal.predicate: _causal_delete_closure(goal.predicate, scenario, actions, ())
        for goal in goals
    }
    by_robot: dict[str, list[ParsedGoal]] = {robot.id: [] for robot in scenario.robots}
    for goal in goals:
        by_robot.setdefault(goal.robot, []).append(goal)
    ordered: list[ParsedGoal] = []
    for robot in scenario.robots:
        local = by_robot[robot.id]
        edges: dict[int, set[int]] = {index: set() for index in range(len(local))}
        indegree = {index: 0 for index in range(len(local))}
        for producer_index, producer_goal in enumerate(local):
            deleted = delete_closures[producer_goal.predicate]
            for restored_index, restored_goal in enumerate(local):
                if producer_index == restored_index or restored_goal.predicate not in deleted:
                    continue
                if restored_index not in edges[producer_index]:
                    edges[producer_index].add(restored_index)
                    indegree[restored_index] += 1
        ready = [index for index in range(len(local)) if indegree[index] == 0]
        local_order: list[int] = []
        while ready:
            current = ready.pop(0)
            local_order.append(current)
            for following in sorted(edges[current]):
                indegree[following] -= 1
                if indegree[following] == 0:
                    ready.append(following)
        if len(local_order) != len(local):
            local_order = list(range(len(local)))
        ordered.extend(local[index] for index in local_order)
    return [
        ParsedGoal(index, goal.robot, goal.predicate, goal.target, goal.destination)
        for index, goal in enumerate(ordered, start=1)
    ]


def _causal_delete_closure(
    predicate: str,
    scenario: Scenario,
    actions: list[GroundAction],
    stack: tuple[str, ...],
) -> set[str]:
    if predicate in scenario.initial_state or predicate in stack:
        return set()
    producers = [action for action in actions if predicate in action.add_effects]
    if not producers:
        return set()
    selected = min(
        producers,
        key=lambda action: (action.timeout_ticks, action.robot, action.name, action.parameters),
    )
    deleted = set(selected.delete_effects)
    for precondition in selected.preconditions:
        deleted.update(
            _causal_delete_closure(precondition, scenario, actions, (*stack, predicate))
        )
    return deleted


def _flatten(node: BTNode) -> list[str]:
    if node.type in {"Sequence", "ReactiveSequence", "Fallback"}:
        token = "f(" if node.type == "Fallback" else "s("
        result = [token]
        for child in node.children:
            result.extend(_flatten(child))
        result.append(")")
        return result
    if node.type == "Condition":
        return [f"{node.label()}?"]
    if node.type == "WaitFor":
        return [f"wait for {node.label()}!"]
    if node.type == "AcquireResource":
        return [f"acquire {node.name}!"]
    if node.type == "ReleaseResource":
        return [f"release {node.name}!"]
    if node.type == "Action":
        parameters = ", ".join(node.parameters)
        return [f"{node.name}({parameters})!"]
    raise BetrXPNativeError(f"Cannot serialize unsupported native node '{node.type}'.")


def _native_skill(action: GroundAction) -> dict[str, Any]:
    if action.name == "pick_source_cradle":
        return {
            "name": "Pick",
            "parameters": {"part": action.parameters[0], "location": "source_cradle"},
        }
    if action.name == "recover_fallen_part":
        return {
            "name": "Pick",
            "parameters": {"part": action.parameters[0], "location": action.parameters[1]},
        }
    return {
        "name": _pascal(action.name),
        "parameters": list(action.parameters),
    }


def _tokenize(formula: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(formula):
        match = _TOKEN.match(formula, position)
        if match is None:
            raise BetrXPNativeError(
                f"Invalid goal formula near '{formula[position:position + 20]}'."
            )
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise BetrXPNativeError("BETR-XP-LLM goal formula is empty.")
    return tokens


def _parse_or(tokens: list[str], index: int):
    node, index = _parse_and(tokens, index)
    while index < len(tokens) and tokens[index] == "|":
        right, index = _parse_and(tokens, index + 1)
        node = ("or", node, right)
    return node, index


def _parse_and(tokens: list[str], index: int):
    node, index = _parse_unary(tokens, index)
    while index < len(tokens) and tokens[index] == "&":
        right, index = _parse_unary(tokens, index + 1)
        node = ("and", node, right)
    return node, index


def _parse_unary(tokens: list[str], index: int):
    if index >= len(tokens):
        raise BetrXPNativeError("Goal formula ends before an operand.")
    token = tokens[index]
    if token == "~":
        child, following = _parse_unary(tokens, index + 1)
        return ("not", child), following
    if token == "(":
        child, following = _parse_or(tokens, index + 1)
        if following >= len(tokens) or tokens[following] != ")":
            raise BetrXPNativeError("Goal formula has an unmatched '('.")
        return child, following + 1
    if token in {"&", "|", ")"}:
        raise BetrXPNativeError(f"Unexpected goal token '{token}'.")
    return ("atom", token), index + 1


def _to_dnf(expression, negated: bool = False) -> list[list[tuple[str, bool]]]:
    kind = expression[0]
    if kind == "atom":
        return [[(expression[1], negated)]]
    if kind == "not":
        return _to_dnf(expression[1], not negated)
    left = _to_dnf(expression[1], negated)
    right = _to_dnf(expression[2], negated)
    effective_kind = kind if not negated else ("or" if kind == "and" else "and")
    if effective_kind == "or":
        return [*left, *right]
    return [[*left_item, *right_item] for left_item in left for right_item in right]


def _pascal(value: str) -> str:
    pieces = re.findall(r"[A-Za-z0-9]+", value)
    return "".join(piece[:1].upper() + piece[1:] for piece in pieces) or "Value"


def _merge_types(
    existing: tuple[str, ...] | None,
    candidate: tuple[str, ...],
) -> tuple[str, ...]:
    if existing is None:
        return candidate
    if len(existing) != len(candidate):
        raise BetrXPNativeError("One predicate name is used with inconsistent arities.")
    return tuple(
        left if left == right else right if left == "constant" else left
        for left, right in zip(existing, candidate)
    )


def _condition_description(name: str, types: tuple[str, ...]) -> str:
    arguments = ", ".join(types) if types else "no arguments"
    return f"{name} is true for ({arguments}). Negating requires it to be false."


def _constant_type(scenario: Scenario, value: str) -> str:
    if value in scenario.robot_ids:
        return "robot"
    return scenario.constant_type(value) or "constant"
