"""Static validation of a generated plan against a scenario.

The validator turns "is this plan correct?" into concrete, machine-checkable
errors with candidate-producer suggestions - the structured feedback that makes
LLM self-correction tractable. Every check from the original prototype is
preserved; they now operate on the typed :class:`llm_mr_bt_planner.plan.Plan` and the
declarative effect model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bt import COMPOSITES, BTNode, iter_leaves, iter_nodes
from .domain import Scenario, apply_grounded, candidate_producers, ground_effects, positive_effects
from .plan import Plan
from .predicates import parse_predicate, substitute


@dataclass
class ValidationError:
    type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "message": self.message}


@dataclass
class ValidationReport:
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, error_type: str, message: str) -> None:
        self.errors.append(ValidationError(error_type, message))

    def to_dicts(self) -> list[dict[str, str]]:
        return [error.to_dict() for error in self.errors]


def validate_plan(
    plan: Plan,
    scenario: Scenario,
    suggest_producers: bool = False,
    *,
    allowed_sources: frozenset[str] | None = None,
    validation_profile: str = "direct",
) -> ValidationReport:
    """Validate ``plan`` against ``scenario``.

    The checks are task-agnostic: structure, capability contracts, predicate
    support, causality, synchronization, resources, and liveness. When
    ``suggest_producers`` is enabled, errors may include producer candidates
    derived from declared effects so the correction loop gets actionable data.
    ``allowed_sources`` and ``reactive_policy`` are explicit comparison-adapter
    opt-ins; the default direct-generation contract remains LLM-only and applies
    flattened synchronization and resource checks.
    """
    report = ValidationReport()

    if validation_profile not in {"direct", "reactive_policy"}:
        report.add(
            "invalid_validation_profile",
            f"Unknown validation profile '{validation_profile}'.",
        )
        return report

    _validate_raw_schema(plan, report)
    for field_name in plan.missing_fields():
        report.add("missing_field", f"Plan is missing '{field_name}'.")
    for robot, reason in plan.unparsable_trees.items():
        report.add("invalid_bt", f"Behavior tree for '{robot}' could not be parsed: {reason}.")
    if report.errors:
        return report

    _validate_behavior_trees(plan, scenario, report)
    _validate_direct_bt_contract(
        plan,
        scenario,
        report,
        allowed_sources=allowed_sources or frozenset({"llm"}),
    )
    _validate_predicate_support(plan, scenario, report, suggest_producers)
    if validation_profile == "direct":
        _validate_explicit_waits(plan, scenario, report)
        _validate_resources(plan, scenario, report)
    return report


# --------------------------------------------------------------------------- #
# Exact LLM output schema
# --------------------------------------------------------------------------- #


_PLAN_FIELDS = {"schema_version", "mission_id", "behavior_trees"}
_COMMON_NODE_FIELDS = {"id", "type", "source"}
_COMPOSITE_NODE_FIELDS = _COMMON_NODE_FIELDS | {"children"}
_LEAF_NODE_FIELDS = _COMMON_NODE_FIELDS | {"name", "parameters"}


def _validate_raw_schema(plan: Plan, report: ValidationReport) -> None:
    """Reject anything the parser would otherwise normalize or discard.

    A valid schema-v2 plan round-trips exactly. This makes it auditable that the
    accepted BT is the model's tree rather than a silently rewritten derivative.
    """
    unknown_plan_fields = sorted(set(plan.raw) - _PLAN_FIELDS)
    if unknown_plan_fields:
        report.add("unknown_plan_field", f"Plan contains unsupported field(s): {', '.join(unknown_plan_fields)}.")
    trees = plan.raw.get("behavior_trees")
    if "behavior_trees" in plan.raw and not isinstance(trees, dict):
        report.add("invalid_plan_type", "Plan 'behavior_trees' must be an object keyed by robot id.")
        return
    if isinstance(trees, dict):
        for robot_id, node in trees.items():
            _validate_raw_node(node, f"behavior_trees.{robot_id}", report)
    if plan.raw and not report.errors and plan.to_dict() != plan.raw:
        report.add(
            "non_exact_roundtrip",
            "The BT would require parser normalization. Every accepted field must round-trip exactly as generated.",
        )


def _validate_raw_node(data: Any, path: str, report: ValidationReport) -> None:
    if not isinstance(data, dict):
        report.add("invalid_bt", f"{path} must be an object.")
        return
    node_type = data.get("type")
    if node_type in COMPOSITES:
        allowed = set(_COMPOSITE_NODE_FIELDS)
        if node_type in {"Parallel", "ParallelAll"}:
            allowed.add("success_threshold")
        required = set(_COMPOSITE_NODE_FIELDS)
    else:
        allowed = set(_LEAF_NODE_FIELDS)
        required = set(_LEAF_NODE_FIELDS)
        if node_type == "Action":
            allowed.add("task_id")
            required.add("task_id")
        if node_type in {"WaitFor", "AcquireResource"}:
            allowed.add("timeout_ticks")
            required.add("timeout_ticks")
    unknown = sorted(set(data) - allowed)
    if unknown:
        report.add("unknown_node_field", f"{path} contains unsupported field(s): {', '.join(unknown)}.")
    missing = sorted(required - set(data))
    if missing:
        report.add("missing_node_field", f"{path} is missing required field(s): {', '.join(missing)}.")
    for key in ("id", "type", "source"):
        if key in data and not isinstance(data[key], str):
            report.add("invalid_node_field", f"{path}.{key} must be a string.")
    if node_type in COMPOSITES:
        children = data.get("children")
        if isinstance(children, list):
            for index, child in enumerate(children):
                _validate_raw_node(child, f"{path}.children[{index}]", report)
        return
    if "name" in data and not isinstance(data["name"], str):
        report.add("invalid_node_field", f"{path}.name must be a string.")
    parameters = data.get("parameters")
    if "parameters" in data and (
        not isinstance(parameters, list) or any(not isinstance(item, str) for item in parameters)
    ):
        report.add("invalid_node_field", f"{path}.parameters must be an array of strings.")
    if "task_id" in data and not isinstance(data["task_id"], str):
        report.add("invalid_node_field", f"{path}.task_id must be a string.")


# --------------------------------------------------------------------------- #
# Behavior trees
# --------------------------------------------------------------------------- #


def _validate_behavior_trees(plan: Plan, scenario: Scenario, report: ValidationReport) -> None:
    for robot_id in sorted(scenario.robot_ids - set(plan.behavior_trees)):
        report.add("missing_robot_tree", f"Plan has no behavior tree for scenario robot '{robot_id}'.")
    for robot_id, tree in plan.behavior_trees.items():
        robot = scenario.robot(robot_id)
        if robot is None:
            report.add("unknown_robot", f"Behavior tree is defined for unknown robot '{robot_id}'.")
            continue
        _validate_bt_node(tree, robot.capability_names, robot_id, report, f"behavior_trees.{robot_id}")


def _validate_bt_node(
    node: BTNode,
    capability_names: set[str],
    robot_id: str,
    report: ValidationReport,
    path: str,
) -> None:
    if node.type in COMPOSITES:
        if not node.children:
            report.add("invalid_bt", f"{path} composite '{node.type}' must have at least one child.")
        if node.type in {"Parallel", "ParallelAll"}:
            threshold = node.success_threshold
            expected_all = (
                node.type == "ParallelAll"
                and isinstance(threshold, int)
                and not isinstance(threshold, bool)
                and threshold == len(node.children)
            )
            valid_parallel = (
                node.type == "Parallel"
                and isinstance(threshold, int)
                and not isinstance(threshold, bool)
                and 1 <= threshold <= len(node.children)
            )
            if not (expected_all or valid_parallel):
                report.add(
                    "invalid_parallel_threshold",
                    f"{path} {node.type} success_threshold is invalid for {len(node.children)} child(ren).",
                )
        for index, child in enumerate(node.children):
            _validate_bt_node(child, capability_names, robot_id, report, f"{path}.children[{index}]")
        return
    if node.type == "Action":
        if node.name not in capability_names:
            report.add("invalid_bt_action", f"Robot '{robot_id}' cannot execute BT action '{node.name}' at {path}.")
        return
    if node.type in {"Condition", "WaitFor"}:
        if not node.name:
            report.add("invalid_bt", f"{path} {node.type} is missing name.")
        if node.type == "WaitFor" and (
            not isinstance(node.timeout_ticks, int) or isinstance(node.timeout_ticks, bool) or node.timeout_ticks <= 0
        ):
            report.add("invalid_wait_timeout", f"{path} WaitFor needs a positive integer timeout_ticks.")
        return
    if node.type in {"AcquireResource", "ReleaseResource"}:
        if not node.name:
            report.add("invalid_bt", f"{path} {node.type} is missing its resource name.")
        if node.type == "AcquireResource" and (
            not isinstance(node.timeout_ticks, int) or isinstance(node.timeout_ticks, bool) or node.timeout_ticks <= 0
        ):
            report.add("invalid_resource_timeout", f"{path} AcquireResource needs a positive timeout_ticks.")
        return
    report.add("invalid_bt", f"{path} uses unsupported node type '{node.type}'.")


# --------------------------------------------------------------------------- #
# Predicate support
# --------------------------------------------------------------------------- #


def _validate_predicate_support(
    plan: Plan, scenario: Scenario, report: ValidationReport, suggest_producers: bool
) -> None:
    initial_state = set(scenario.initial_state)
    produced = produced_predicates(plan, scenario)

    for goal in scenario.goal_state:
        if goal not in initial_state and goal not in produced:
            report.add(
                "unsupported_goal",
                f"Goal '{goal}' is not initially true and no generated BT action creates it."
                f"{_candidate_text(candidate_producers(goal, scenario), suggest_producers)}",
            )

    for robot_id, tree in plan.behavior_trees.items():
        robot = scenario.robot(robot_id)
        if robot is None:
            continue
        leaves = list(iter_leaves(tree))
        for leaf in leaves:
            if leaf.type in {"Condition", "WaitFor"}:
                _check_condition(robot_id, leaf, scenario, initial_state, produced, report, suggest_producers)
            elif leaf.type == "Action":
                _check_action_preconditions(robot_id, leaf, scenario, initial_state, produced, report, suggest_producers)


def _check_condition(
    robot_id: str,
    leaf: BTNode,
    scenario: Scenario,
    initial_state: set[str],
    produced: set[str],
    report: ValidationReport,
    suggest_producers: bool,
) -> None:
    predicate = leaf.label()
    if predicate in initial_state or predicate in produced:
        return
    report.add(
        "unsupported_condition" if leaf.type == "Condition" else "unsupported_wait",
        f"{leaf.type} '{predicate}' in robot '{robot_id}' BT is not initially true and no generated action creates it."
        f"{_same_name_text(predicate, initial_state | produced, suggest_producers)}"
        f"{_candidate_text(candidate_producers(predicate, scenario), suggest_producers)}",
    )

def _check_action_preconditions(
    robot_id: str,
    leaf: BTNode,
    scenario: Scenario,
    initial_state: set[str],
    produced: set[str],
    report: ValidationReport,
    suggest_producers: bool,
) -> None:
    capability = scenario.capability(robot_id, leaf.name or "")
    if capability is None:
        return
    bindings = dict(zip(capability.parameters, leaf.parameters))
    for precondition in capability.preconditions:
        predicate = substitute(precondition, bindings)
        if predicate in initial_state or predicate in produced:
            continue
        report.add(
            "unsupported_precondition",
            f"Action {leaf.label()} on robot '{robot_id}' needs '{predicate}', "
            f"but no initial predicate or generated action creates it."
            f"{_candidate_text(candidate_producers(predicate, scenario), suggest_producers)}",
        )


def produced_predicates(plan: Plan, scenario: Scenario) -> set[str]:
    """Every positive predicate any generated BT action can create."""
    produced: set[str] = set()
    for robot_id, tree in plan.behavior_trees.items():
        for leaf in iter_leaves(tree):
            if leaf.type != "Action":
                continue
            capability = scenario.capability(robot_id, leaf.name or "")
            if capability is None:
                continue
            bindings = dict(zip(capability.parameters, leaf.parameters))
            produced.update(positive_effects(capability.effects, bindings))
    return produced


# --------------------------------------------------------------------------- #
# Direct LLM BT schema-v2 contract
# --------------------------------------------------------------------------- #


def _validate_direct_bt_contract(
    plan: Plan,
    scenario: Scenario,
    report: ValidationReport,
    *,
    allowed_sources: frozenset[str],
) -> None:
    if plan.schema_version != "2.0":
        report.add("unsupported_schema_version", f"Unsupported plan schema_version '{plan.schema_version}'.")
    if plan.mission_id != scenario.task_id:
        report.add(
            "mission_mismatch",
            f"Plan mission_id '{plan.mission_id}' does not match scenario task_id '{scenario.task_id}'.",
        )

    seen_nodes: set[str] = set()
    seen_tasks: set[str] = set()
    signatures = _predicate_signatures(scenario)
    for robot_id, tree in plan.behavior_trees.items():
        robot = scenario.robot(robot_id)
        if robot is None:
            continue
        for node in iter_nodes(tree):
            if not node.node_id or node.id_generated:
                report.add("missing_node_id", f"A node in robot '{robot_id}' has no id.")
            elif node.node_id in seen_nodes:
                report.add("duplicate_node_id", f"Node id '{node.node_id}' is duplicated.")
            else:
                seen_nodes.add(node.node_id)

            if node.type == "Action":
                capability = robot.capability(node.name or "")
                if not node.task_id:
                    report.add("missing_task_id", f"Action '{node.label()}' on '{robot_id}' has no task_id.")
                elif node.task_id in seen_tasks:
                    report.add("duplicate_task_id", f"Action task_id '{node.task_id}' is duplicated.")
                else:
                    seen_tasks.add(node.task_id)
                if capability is not None:
                    if len(node.parameters) != len(capability.parameters):
                        report.add(
                            "action_arity",
                            f"Action '{node.label()}' on '{robot_id}' expects {len(capability.parameters)} "
                            f"parameter(s), got {len(node.parameters)}.",
                        )
                    unknown = [parameter for parameter in node.parameters if parameter not in scenario.constants]
                    if unknown:
                        report.add(
                            "unknown_constant",
                            f"Action '{node.label()}' uses unknown constant(s): {', '.join(unknown)}.",
                        )
                    if capability.parameter_types and len(node.parameters) == len(capability.parameter_types):
                        for parameter, expected_type in zip(node.parameters, capability.parameter_types):
                            actual_type = scenario.constant_type(parameter)
                            if expected_type == "robot":
                                matches = parameter in scenario.robot_ids
                            else:
                                matches = actual_type == expected_type
                            if not matches:
                                report.add(
                                    "argument_type",
                                    f"Action '{node.label()}' parameter '{parameter}' must have type "
                                    f"'{expected_type}', got '{actual_type or 'unknown'}'.",
                                )
                    if node.source not in allowed_sources:
                        report.add(
                            "invalid_provenance",
                            f"Action '{node.label()}' must have source in {sorted(allowed_sources)}.",
                        )
            elif node.type in {"Condition", "WaitFor"} and node.name:
                expected = signatures.get(node.name)
                if expected is None:
                    report.add("unknown_predicate", f"{node.type} '{node.label()}' uses an unknown predicate.")
                elif len(node.parameters) != expected:
                    report.add(
                        "predicate_arity",
                        f"{node.type} '{node.label()}' expects {expected} parameter(s), got {len(node.parameters)}.",
                    )
                unknown = [parameter for parameter in node.parameters if parameter not in scenario.constants]
                if unknown:
                    report.add(
                        "unknown_constant",
                        f"{node.type} '{node.label()}' uses unknown constant(s): {', '.join(unknown)}.",
                    )
                if node.source not in allowed_sources:
                    report.add(
                        "invalid_provenance",
                        f"{node.type} '{node.label()}' must have source in {sorted(allowed_sources)}.",
                    )
            elif node.type in {"AcquireResource", "ReleaseResource"}:
                if node.name not in scenario.resource_ids:
                    report.add("unknown_resource", f"{node.type} uses undeclared resource '{node.name}'.")
                if node.source not in allowed_sources:
                    report.add(
                        "invalid_provenance",
                        f"{node.type} '{node.name}' must have source in {sorted(allowed_sources)}.",
                    )
            elif node.type in COMPOSITES and node.source not in allowed_sources:
                report.add(
                    "invalid_provenance",
                    f"Composite node '{node.node_id}' must have source in {sorted(allowed_sources)}.",
                )


def _predicate_signatures(scenario: Scenario) -> dict[str, int]:
    signatures: dict[str, int] = {}
    literals = [*scenario.initial_state, *scenario.goal_state]
    for robot in scenario.robots:
        for capability in robot.capabilities:
            literals.extend(capability.preconditions)
            literals.extend(capability.effects.add)
            literals.extend(capability.effects.delete)
    for literal in literals:
        name, args = parse_predicate(literal)
        signatures[name] = max(signatures.get(name, 0), len(args))
    return signatures


def _validate_explicit_waits(plan: Plan, scenario: Scenario, report: ValidationReport) -> None:
    """Validate cross-robot causality with explicit WaitFor leaves.

    The check is intentionally conservative: local facts must be established by
    an earlier action in the same flattened execution order; cross-robot facts
    must have an earlier WaitFor and an actual producer action in another tree.
    The simulator remains the final dynamic check for composite control flow.
    """
    initial = set(scenario.initial_state)
    produced_by: dict[str, set[str]] = {}
    deleted_by: dict[str, set[str]] = {}
    for producer, tree in plan.behavior_trees.items():
        for leaf in iter_leaves(tree):
            if leaf.type != "Action":
                continue
            capability = scenario.capability(producer, leaf.name or "")
            if capability is None or len(leaf.parameters) != len(capability.parameters):
                continue
            bindings = dict(zip(capability.parameters, leaf.parameters))
            for predicate in positive_effects(capability.effects, bindings):
                produced_by.setdefault(predicate, set()).add(producer)
            _, deletes = ground_effects(capability.effects, bindings)
            for predicate in deletes:
                deleted_by.setdefault(predicate, set()).add(producer)

    # Robot-level dependency graphs reject valid phased collaboration such as
    # mobile-base -> mounted-arm -> mobile-base.  Track the concrete flattened
    # leaf occurrences instead: a deadlock exists only when producer/wait
    # precedence and per-robot program order form a cycle.
    precedence_edges: dict[str, set[str]] = {}
    producer_events: dict[str, list[str]] = {}
    leaves_by_robot: dict[str, list[BTNode]] = {
        robot_id: list(iter_leaves(tree)) for robot_id, tree in plan.behavior_trees.items()
    }
    for producer, leaves in leaves_by_robot.items():
        previous: str | None = None
        for index, leaf in enumerate(leaves):
            event = f"{producer}@{index}"
            precedence_edges.setdefault(event, set())
            if previous is not None:
                precedence_edges[event].add(previous)
            previous = event
            if leaf.type != "Action":
                continue
            capability = scenario.capability(producer, leaf.name or "")
            if capability is None or len(leaf.parameters) != len(capability.parameters):
                continue
            bindings = dict(zip(capability.parameters, leaf.parameters))
            for predicate in positive_effects(capability.effects, bindings):
                producer_events.setdefault(predicate, []).append(event)

    for robot_id, tree in plan.behavior_trees.items():
        known = set(initial)
        waited: set[str] = set()
        leaves = list(iter_leaves(tree))
        for index, leaf in enumerate(leaves):
            predicate = leaf.label()
            if leaf.type == "Condition":
                # A Condition is a branch/guard and is allowed to fail. It does
                # not establish a predicate or replace cross-robot WaitFor.
                continue
            if leaf.type == "WaitFor":
                # A fact initially/local-sequentially known is not redundant
                # when another tree can invalidate it before this wait. This
                # is common for a mounted arm's stowed/deployed handshake.
                externally_mutable = bool(deleted_by.get(predicate, set()) - {robot_id})
                if predicate in known and not externally_mutable:
                    report.add(
                        "redundant_wait",
                        f"WaitFor '{predicate}' on '{robot_id}' is already guaranteed true at this point.",
                    )
                producers = produced_by.get(predicate, set()) - {robot_id}
                if not producers:
                    local = robot_id in produced_by.get(predicate, set())
                    report.add(
                        "same_robot_wait" if local else "missing_wait_producer",
                        f"WaitFor '{predicate}' on '{robot_id}' has no producer action in another robot's tree.",
                    )
                wait_event = f"{robot_id}@{index}"
                seen_producer_robots: set[str] = set()
                for producer_event in producer_events.get(predicate, []):
                    producer_robot = producer_event.rsplit("@", 1)[0]
                    if producer_robot == robot_id or producer_robot in seen_producer_robots:
                        continue
                    # WaitFor has OR semantics when several actions/trees can
                    # establish the same literal. Requiring every occurrence
                    # creates false cycles with repeated phased handshakes.
                    precedence_edges[wait_event].add(producer_event)
                    seen_producer_robots.add(producer_robot)
                waited.add(predicate)
                known.add(predicate)
                continue
            if leaf.type != "Action":
                continue
            capability = scenario.capability(robot_id, leaf.name or "")
            if capability is None or len(leaf.parameters) != len(capability.parameters):
                continue
            bindings = dict(zip(capability.parameters, leaf.parameters))
            for template in capability.preconditions:
                needed = substitute(template, bindings)
                if needed in known:
                    continue
                external = produced_by.get(needed, set()) - {robot_id}
                if external and needed not in waited:
                    report.add(
                        "missing_wait_for",
                        f"Action '{leaf.label()}' on '{robot_id}' consumes cross-robot predicate '{needed}' "
                        "without an earlier exact WaitFor.",
                    )
                elif not external:
                    report.add(
                        "precondition_not_ordered",
                        f"Action '{leaf.label()}' on '{robot_id}' needs '{needed}' before any local action "
                        "or explicit cross-robot wait establishes it.",
                    )
            adds, deletes = ground_effects(capability.effects, bindings)
            apply_grounded(known, adds, deletes)

    cycle = _wait_cycle(precedence_edges)
    if cycle:
        robots = [event.rsplit("@", 1)[0] for event in cycle]
        collapsed = [robot for index, robot in enumerate(robots) if index == 0 or robot != robots[index - 1]]
        report.add("wait_cycle", f"Cross-robot WaitFor dependency cycle: {' -> '.join(collapsed)}.")


def _validate_resources(plan: Plan, scenario: Scenario, report: ValidationReport) -> None:
    resource_edges: dict[str, set[str]] = {resource: set() for resource in scenario.resource_ids}
    for robot_id, tree in plan.behavior_trees.items():
        held: set[str] = set()
        for leaf in iter_leaves(tree):
            if leaf.type == "AcquireResource":
                resource = leaf.name or ""
                if resource in held:
                    report.add(
                        "double_acquire",
                        f"Robot '{robot_id}' acquires resource '{resource}' while already owning it.",
                    )
                for owner in held:
                    resource_edges.setdefault(owner, set()).add(resource)
                held.add(resource)
            elif leaf.type == "ReleaseResource":
                resource = leaf.name or ""
                if resource not in held:
                    report.add(
                        "release_without_acquire",
                        f"Robot '{robot_id}' releases resource '{resource}' without owning it.",
                    )
                held.discard(resource)
            elif leaf.type == "Action":
                capability = scenario.capability(robot_id, leaf.name or "")
                missing = set(capability.resources if capability else ()) - held
                if missing:
                    report.add(
                        "resource_not_dominating_action",
                        f"Action '{leaf.label()}' on '{robot_id}' is not dominated by acquisition of "
                        f"{', '.join(sorted(missing))}.",
                    )
        if held:
            report.add(
                "resource_not_released",
                f"Robot '{robot_id}' can finish while still owning: {', '.join(sorted(held))}.",
            )
    cycle = _wait_cycle(resource_edges)
    if cycle:
        report.add("resource_cycle", f"Static resource acquisition cycle: {' -> '.join(cycle)}.")


def _wait_cycle(edges: dict[str, set[str]]) -> list[str]:
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> list[str]:
        if node in visiting:
            start = visiting.index(node)
            return [*visiting[start:], node]
        if node in visited:
            return []
        visiting.append(node)
        for dependency in edges.get(node, set()):
            found = visit(dependency)
            if found:
                return found
        visiting.pop()
        visited.add(node)
        return []

    for node in edges:
        found = visit(node)
        if found:
            return found
    return []


# --------------------------------------------------------------------------- #
# Synchronization
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _candidate_text(candidates: list[Any], suggest_producers: bool) -> str:
    if not suggest_producers or not candidates:
        return ""
    return f" Candidate producer actions: {', '.join(c.describe() for c in candidates)}."


def _same_name_text(predicate: str, candidates: set[str], suggest_producers: bool) -> str:
    if not suggest_producers:
        return ""
    name, _ = parse_predicate(predicate)
    matches = sorted(c for c in candidates if parse_predicate(c)[0] == name and c != predicate)
    if not matches:
        return ""
    return f" Did you mean: {', '.join(matches[:5])}?"
