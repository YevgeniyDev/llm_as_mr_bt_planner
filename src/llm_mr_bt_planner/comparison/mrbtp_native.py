"""Source-aligned FIFO cross-tree expansion and common observation for MRBTP.

The planning state transitions follow Algorithm 2 and the official MIT source's
``MABTP``/``PlanningAgent`` path. The optional LLM/composite-action plugin is
deliberately disabled for the primary non-LLM comparison configuration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ..bt import BTNode
from ..domain import Scenario
from ..predicates import parse_predicate
from .llm_bt_native import GroundAction, ground_action_templates


class MRBTPNativeError(RuntimeError):
    """Raised when MRBTP exhausts its safety bound or cannot expose a solution."""


@dataclass
class PlanningCondition:
    condition_set: frozenset[str]
    action: GroundAction | None = None
    children: list[PlanningCondition] = field(default_factory=list)


@dataclass
class PlanningAgent:
    robot: str
    actions: tuple[GroundAction, ...]
    goal_condition: PlanningCondition
    expanded_conditions: dict[frozenset[str], PlanningCondition]


@dataclass(frozen=True)
class ExpansionEdge:
    index: int
    robot: str
    target: frozenset[str]
    premise: frozenset[str]
    action: GroundAction
    operation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "robot": self.robot,
            "target_condition": sorted(self.target),
            "premise_condition": sorted(self.premise),
            "action": self.action.to_dict(),
            "operation": self.operation,
        }


@dataclass
class MRBTPConstruction:
    agents: dict[str, PlanningAgent]
    solved: bool
    solution_premise: frozenset[str] | None
    explored_conditions: list[frozenset[str]]
    expanded_edges: list[ExpansionEdge]
    witness: list[ExpansionEdge]
    trace: list[dict[str, Any]]
    trees: dict[str, BTNode]


def plan_mrbtp(
    scenario: Scenario,
    *,
    max_expansions: int = 10_000,
) -> MRBTPConstruction:
    """Run the paper's FIFO multi-robot cross-tree expansion."""
    if max_expansions < 1:
        raise MRBTPNativeError("max_expansions must be positive.")
    goal = frozenset(scenario.goal_state)
    start = frozenset(scenario.initial_state)
    grounded = ground_action_templates(scenario)
    agents: dict[str, PlanningAgent] = {}
    for robot in scenario.robots:
        root = PlanningCondition(goal)
        agents[robot.id] = PlanningAgent(
            robot=robot.id,
            actions=tuple(action for action in grounded if action.robot == robot.id),
            goal_condition=root,
            expanded_conditions={goal: root},
        )
    if goal <= start:
        trees = _common_trees(scenario, agents)
        return MRBTPConstruction(
            agents=agents,
            solved=True,
            solution_premise=goal,
            explored_conditions=[],
            expanded_edges=[],
            witness=[],
            trace=[{"event": "goal_already_satisfied", "goal": sorted(goal)}],
            trees=trees,
        )

    queue: deque[frozenset[str]] = deque([goal])
    queued = {goal}
    explored: list[frozenset[str]] = []
    edges: list[ExpansionEdge] = []
    trace: list[dict[str, Any]] = []
    origin: dict[frozenset[str], ExpansionEdge] = {}
    solution_edge: ExpansionEdge | None = None

    while queue and solution_edge is None:
        condition = queue.popleft()
        queued.discard(condition)
        explored.append(condition)
        trace.append(
            {
                "event": "pop_unexpanded_condition",
                "condition": sorted(condition),
                "queue_size_after_pop": len(queue),
            }
        )
        if len(explored) > max_expansions:
            raise MRBTPNativeError(
                f"MRBTP exceeded the {max_expansions}-condition expansion safety bound."
            )

        round_solution: ExpansionEdge | None = None
        for robot in scenario.robots:
            agent = agents[robot.id]
            new_edges = _expand_one_robot(
                scenario,
                agent,
                condition,
                next_index=len(edges) + 1,
            )
            edges.extend(new_edges)
            trace.append(
                {
                    "event": "one_step_cross_tree_expansion",
                    "robot": robot.id,
                    "condition": sorted(condition),
                    "new_conditions": [sorted(edge.premise) for edge in new_edges],
                    "operations": [edge.operation for edge in new_edges],
                }
            )
            for edge in new_edges:
                origin.setdefault(edge.premise, edge)
                if edge.premise not in queued and edge.premise not in explored:
                    queue.append(edge.premise)
                    queued.add(edge.premise)
                if round_solution is None and edge.premise <= start:
                    round_solution = edge
        solution_edge = round_solution

    solved = solution_edge is not None
    witness = _solution_witness(solution_edge, goal, origin) if solution_edge else []
    trees = _common_trees(scenario, agents)
    trace.append(
        {
            "event": "planning_complete",
            "solved": solved,
            "expanded_condition_count": len(explored),
            "expanded_edge_count": len(edges),
            "solution_premise": sorted(solution_edge.premise) if solution_edge else None,
            "witness_actions": [
                f"{edge.action.name}({','.join(edge.action.parameters)})" for edge in witness
            ],
        }
    )
    return MRBTPConstruction(
        agents=agents,
        solved=solved,
        solution_premise=solution_edge.premise if solution_edge else None,
        explored_conditions=explored,
        expanded_edges=edges,
        witness=witness,
        trace=trace,
        trees=trees,
    )


def native_forest_document(construction: MRBTPConstruction) -> dict[str, Any]:
    """Serialize all native backup branches without common executor leaves."""
    return {
        robot: _native_root(agent.goal_condition)
        for robot, agent in construction.agents.items()
    }


def planning_graph_document(construction: MRBTPConstruction) -> dict[str, Any]:
    return {
        "solved": construction.solved,
        "solution_premise": (
            sorted(construction.solution_premise)
            if construction.solution_premise is not None
            else None
        ),
        "explored_conditions": [sorted(condition) for condition in construction.explored_conditions],
        "expanded_edges": [edge.to_dict() for edge in construction.expanded_edges],
        "solution_witness": [edge.to_dict() for edge in construction.witness],
        "per_robot_expanded_conditions": {
            robot: [sorted(condition) for condition in agent.expanded_conditions]
            for robot, agent in construction.agents.items()
        },
    }


def intention_sharing_document(scenario: Scenario) -> dict[str, Any]:
    return {
        "enabled": True,
        "priority_order": [robot.id for robot in scenario.robots],
        "native_protocol": {
            "intention_queue": "ordered current actions broadcast by robots",
            "belief_success": "union of add(a)-del(a) for higher-priority intentions",
            "belief_failure": "union of del(a)-add(a) for higher-priority intentions",
            "blocking": (
                "an action whose believed-success precondition is not yet physically true "
                "shares its intention and returns RUNNING"
            ),
        },
        "common_observation": (
            "native premise conditions remain reactive guards and bounded team-goal WaitFor "
            "leaves keep all robot roots active; speculative belief success is not emulated"
        ),
    }


def validate_native_construction(
    scenario: Scenario,
    construction: MRBTPConstruction,
) -> list[dict[str, Any]]:
    """Check the source-aligned planning invariants independently of common BT flow."""
    errors: list[dict[str, Any]] = []
    action_library = set(ground_action_templates(scenario))
    if not construction.solved:
        errors.append({"type": "unsolvable", "message": "MRBTP did not reach the initial state."})
    if set(construction.agents) != scenario.robot_ids:
        errors.append(
            {
                "type": "agent_set_mismatch",
                "message": "MRBTP did not construct exactly one native policy per robot.",
            }
        )
    for edge in construction.expanded_edges:
        action = edge.action
        pre = frozenset(action.preconditions)
        add = frozenset(action.add_effects)
        delete = frozenset(action.delete_effects)
        expected = (pre | edge.target) - add
        if action not in action_library or action.robot != edge.robot:
            errors.append(
                {
                    "type": "action_space_violation",
                    "message": f"Edge {edge.index} uses an action outside robot {edge.robot}'s library.",
                }
            )
        if not edge.target & ((pre | add) - delete) or edge.target - delete != edge.target:
            errors.append(
                {
                    "type": "invalid_premise_action",
                    "message": f"Edge {edge.index} violates MRBTP action-selection conditions.",
                }
            )
        if edge.premise != expected:
            errors.append(
                {
                    "type": "invalid_premise_formula",
                    "message": f"Edge {edge.index} does not use pre(a) union c minus add(a).",
                }
            )
        if _condition_conflict(scenario, edge.premise):
            errors.append(
                {
                    "type": "conflicting_premise",
                    "message": f"Edge {edge.index} contains a contradictory common-domain condition.",
                }
            )
    if construction.solved and not frozenset(scenario.goal_state) <= frozenset(
        scenario.initial_state
    ):
        if not construction.witness:
            errors.append(
                {"type": "missing_witness", "message": "Solved MRBTP graph has no expansion witness."}
            )
        else:
            if not construction.witness[0].premise <= frozenset(scenario.initial_state):
                errors.append(
                    {
                        "type": "witness_not_grounded",
                        "message": "MRBTP witness does not begin in the supplied initial state.",
                    }
                )
            if construction.witness[-1].target != frozenset(scenario.goal_state):
                errors.append(
                    {
                        "type": "witness_goal_mismatch",
                        "message": "MRBTP witness does not terminate at the complete team goal.",
                    }
                )
            for current, parent in zip(construction.witness, construction.witness[1:]):
                if current.target != parent.premise:
                    errors.append(
                        {
                            "type": "disconnected_witness",
                            "message": (
                                f"MRBTP witness edge {current.index} does not connect to "
                                f"edge {parent.index}."
                            ),
                        }
                    )
    return errors


def _expand_one_robot(
    scenario: Scenario,
    agent: PlanningAgent,
    condition: frozenset[str],
    *,
    next_index: int,
) -> list[ExpansionEdge]:
    inside = agent.expanded_conditions.get(condition)
    new_nodes: list[PlanningCondition] = []
    new_edges: list[ExpansionEdge] = []
    for action in agent.actions:
        pre = frozenset(action.preconditions)
        add = frozenset(action.add_effects)
        delete = frozenset(action.delete_effects)
        if not condition & ((pre | add) - delete):
            continue
        if condition - delete != condition:
            continue
        premise = (pre | condition) - add
        if any(expanded <= premise for expanded in agent.expanded_conditions):
            continue
        if _condition_conflict(scenario, premise):
            continue
        planning_condition = PlanningCondition(premise, action=action)
        agent.expanded_conditions[premise] = planning_condition
        new_nodes.append(planning_condition)
        operation = "in_tree_expand" if inside is not None else "cross_tree_expand"
        new_edges.append(
            ExpansionEdge(
                index=next_index + len(new_edges),
                robot=agent.robot,
                target=condition,
                premise=premise,
                action=action,
                operation=operation,
            )
        )
    if inside is not None:
        inside.children.extend(new_nodes)
    elif new_nodes:
        wrapper = PlanningCondition(condition, children=new_nodes)
        agent.goal_condition.children.append(wrapper)
    return new_edges


def _condition_conflict(scenario: Scenario, condition: frozenset[str]) -> bool:
    part_ids = {entity.id for entity in scenario.entities if entity.type == "part"}
    locations: dict[str, set[str]] = {part: set() for part in part_ids}
    holders: dict[str, set[str]] = {part: set() for part in part_ids}
    empty_grippers: set[str] = set()
    docks: dict[str, set[str]] = {robot.id: set() for robot in scenario.robots}
    for literal in condition:
        name, arguments = parse_predicate(literal)
        if name == "at" and len(arguments) == 2 and arguments[0] in part_ids:
            locations[arguments[0]].add(arguments[1])
        elif name == "holding" and len(arguments) == 2 and arguments[1] in part_ids:
            holders[arguments[1]].add(arguments[0])
        elif name == "gripper_empty" and len(arguments) == 1:
            empty_grippers.add(arguments[0])
        elif name == "docked" and len(arguments) == 2 and arguments[0] in docks:
            docks[arguments[0]].add(arguments[1])
    for part in part_ids:
        if len(locations[part]) > 1 or len(holders[part]) > 1:
            return True
        if locations[part] and holders[part]:
            return True
        if any(holder in empty_grippers for holder in holders[part]):
            return True
    return any(len(values) > 1 for values in docks.values())


def _solution_witness(
    solution: ExpansionEdge,
    goal: frozenset[str],
    origin: dict[frozenset[str], ExpansionEdge],
) -> list[ExpansionEdge]:
    witness = [solution]
    target = solution.target
    visited: set[frozenset[str]] = set()
    while target != goal:
        if target in visited or target not in origin:
            raise MRBTPNativeError("MRBTP could not reconstruct its expansion witness.")
        visited.add(target)
        parent = origin[target]
        witness.append(parent)
        target = parent.target
    return witness


def _native_root(goal: PlanningCondition) -> dict[str, Any]:
    return {
        "node": "Selector",
        "children": [
            _native_condition_set(goal.condition_set),
            *[_native_branch(child) for child in goal.children],
        ],
    }


def _native_branch(node: PlanningCondition) -> dict[str, Any]:
    if node.action is None:
        return {
            "node": "Selector",
            "children": [
                _native_condition_set(node.condition_set),
                *[_native_branch(child) for child in node.children],
            ],
        }
    condition: dict[str, Any] = _native_condition_set(node.condition_set)
    if node.children:
        condition = {
            "node": "Selector",
            "children": [condition, *[_native_branch(child) for child in node.children]],
        }
    return {
        "node": "Sequence",
        "children": [condition, _native_action(node.action)],
    }


def _native_condition_set(condition: frozenset[str]) -> dict[str, Any]:
    return {"node": "ConditionSet", "literals": sorted(condition)}


def _native_action(action: GroundAction) -> dict[str, Any]:
    return {
        "node": "Action",
        "robot": action.robot,
        "name": action.name,
        "parameters": list(action.parameters),
    }


def _common_trees(
    scenario: Scenario,
    agents: dict[str, PlanningAgent],
) -> dict[str, BTNode]:
    grounded = ground_action_templates(scenario)
    trees: dict[str, BTNode] = {}
    for robot in scenario.robots:
        counter = _NodeCounter(f"mrbtp.{robot.id}")
        policy = _common_root(agents[robot.id].goal_condition, scenario, counter)
        stable = _stable_initial_literal(scenario, grounded)
        drive = BTNode(
            type="Fallback",
            node_id=counter.next("drive_or_idle"),
            source="planner",
            children=[policy, _condition(stable, counter.next("idle_guard"))],
        )
        completion = [
            _wait(goal, counter.next("team_goal"), timeout_ticks=120)
            for goal in scenario.goal_state
        ]
        trees[robot.id] = BTNode(
            type="ReactiveSequence",
            node_id=f"mrbtp.{robot.id}.root",
            source="planner",
            children=[drive, *completion],
        )
    return trees


def _common_root(
    goal: PlanningCondition,
    scenario: Scenario,
    counter: _NodeCounter,
) -> BTNode:
    children = [
        _condition_set(goal.condition_set, scenario, counter),
        *[_common_branch(child, scenario, counter) for child in goal.children],
    ]
    return _fallback_or_only(children, counter.next("native_selector"))


def _common_branch(
    node: PlanningCondition,
    scenario: Scenario,
    counter: _NodeCounter,
) -> BTNode:
    if node.action is None:
        children = [
            _condition_set(node.condition_set, scenario, counter),
            *[_common_branch(child, scenario, counter) for child in node.children],
        ]
        return _fallback_or_only(children, counter.next("cross_tree_selector"))

    gate = _condition_set(node.condition_set, scenario, counter)
    if node.children:
        gate = _fallback_or_only(
            [
                gate,
                *[_common_branch(child, scenario, counter) for child in node.children],
            ],
            counter.next("in_tree_selector"),
        )
    steps = [gate]
    for resource in sorted(node.action.resources):
        steps.append(
            BTNode(
                type="AcquireResource",
                node_id=counter.next("acquire"),
                name=resource,
                timeout_ticks=max(1, node.action.timeout_ticks),
                source="planner",
            )
        )
    steps.append(
        BTNode(
            type="Action",
            node_id=counter.next(node.action.name),
            task_id=counter.next("task"),
            name=node.action.name,
            parameters=node.action.parameters,
            source="planner",
        )
    )
    for resource in reversed(sorted(node.action.resources)):
        steps.append(
            BTNode(
                type="ReleaseResource",
                node_id=counter.next("release"),
                name=resource,
                source="planner",
            )
        )
    return _sequence_or_only(steps, counter.next("action_sequence"))


def _condition_set(
    condition: frozenset[str],
    scenario: Scenario,
    counter: _NodeCounter,
) -> BTNode:
    nodes: list[BTNode] = []
    for literal in sorted(condition):
        nodes.append(_condition(literal, counter.next("condition")))
    if not nodes:
        return _condition(_stable_initial_literal(scenario, []), counter.next("empty_premise"))
    return _sequence_or_only(nodes, counter.next("condition_set"))


def _fallback_or_only(children: list[BTNode], node_id: str) -> BTNode:
    materialized = [child for child in children if child is not None]
    if len(materialized) == 1:
        return materialized[0]
    return BTNode(type="Fallback", node_id=node_id, children=materialized, source="planner")


def _sequence_or_only(children: list[BTNode], node_id: str) -> BTNode:
    if len(children) == 1:
        return children[0]
    return BTNode(type="Sequence", node_id=node_id, children=children, source="planner")


def _condition(predicate: str, node_id: str) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="Condition",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        source="planner",
    )


def _wait(predicate: str, node_id: str, *, timeout_ticks: int) -> BTNode:
    name, parameters = parse_predicate(predicate)
    return BTNode(
        type="WaitFor",
        node_id=node_id,
        name=name,
        parameters=tuple(parameters),
        timeout_ticks=timeout_ticks,
        source="planner",
    )


def _stable_initial_literal(
    scenario: Scenario,
    actions: list[GroundAction],
) -> str:
    deleted = {literal for action in actions for literal in action.delete_effects}
    stable = next((literal for literal in scenario.initial_state if literal not in deleted), None)
    if stable is None:
        raise MRBTPNativeError("MRBTP common observation requires one stable initial literal.")
    return stable


class _NodeCounter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.value = 0

    def next(self, label: str) -> str:
        self.value += 1
        return f"{self.prefix}.{self.value:05d}.{label}"
