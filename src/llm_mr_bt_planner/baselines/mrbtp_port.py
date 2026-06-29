"""Port our declarative scenarios into MRBTP's symbolic planning input.

MRBTP plans over *ground* ``PlanningAction`` objects (``name``, ``pre``, ``add``,
``del_set`` as sets of predicate strings) and exact-set state transitions. Our domain
is *lifted* (capabilities with parameter templates) and uses delete *patterns* (a
partial predicate clears any matching fact - the functional-fluent convention). This
module grounds each capability over the scenario's constants and expands every delete
pattern into the concrete facts it would remove, so MRBTP's exact-set semantics match
ours. The result is plain JSON (no MRBTP import needed); ``scripts/run_mrbtp.py`` turns
it into ``PlanningAction`` objects and runs the planner.
"""

from __future__ import annotations

from itertools import product
from typing import Any

from ..domain import Scenario
from ..predicates import matches_pattern, substitute


def port_scenario(scenario: Scenario) -> dict[str, Any]:
    """Return ``{task_id, start, goal, agents:[{robot, actions:[{name,pre,add,del_set,cost}]}]}``.

    Action names are bare with their bound parameters, ``action(p1,p2)`` - no robot
    token - so the converter can recover our action name + parameters and assign the
    action to the robot whose tree it appears in. Robot-scoped predicates keep their
    literal robot id (it is hardcoded in the template, not a parameter).
    """
    # Parameters in these scenarios range over objects and locations.
    domain = tuple(scenario.objects) + tuple(scenario.locations)

    agents: list[dict[str, Any]] = []
    grounded_by_robot: list[list[dict[str, Any]]] = []
    for robot in scenario.robots:
        ground_actions = _ground_robot_actions(robot, domain)
        grounded_by_robot.append(ground_actions)

    # Delete-relaxation reachability pruning: keep only ground actions whose
    # preconditions are reachable from the initial state (ignoring deletes). This drops
    # the nonsense bindings that exhaustive grounding produces, giving MRBTP a fair,
    # tight action set instead of one inflated by unsatisfiable actions.
    grounded_by_robot = _prune_unreachable(grounded_by_robot, set(scenario.initial_state))

    # Reachable fact universe: initial facts + every kept ground add-effect + goal facts.
    universe: set[str] = set(scenario.initial_state) | set(scenario.goal_state)
    for ground_actions in grounded_by_robot:
        for action in ground_actions:
            universe.update(action["add"])

    for robot, ground_actions in zip(scenario.robots, grounded_by_robot, strict=True):
        for action in ground_actions:
            patterns = action.pop("_del_patterns")
            action["del_set"] = sorted(
                fact for fact in universe
                if any(matches_pattern(fact, pattern) for pattern in patterns)
            )
        agents.append({"robot": robot.id, "actions": ground_actions})

    return {
        "task_id": scenario.task_id,
        "start": sorted(scenario.initial_state),
        "goal": sorted(scenario.goal_state),
        "agents": agents,
    }


def _prune_unreachable(
    grounded_by_robot: list[list[dict[str, Any]]], initial: set[str]
) -> list[list[dict[str, Any]]]:
    """Keep only actions whose preconditions are reachable under delete-relaxation.

    Standard grounder reachability: starting from ``initial``, repeatedly fire any
    action whose preconditions are already reachable and add its add-effects, to a
    fixpoint. Actions never fired have unsatisfiable preconditions and are dropped.
    """
    reachable = set(initial)
    flat = [(r, action) for r, actions in enumerate(grounded_by_robot) for action in actions]
    fired: set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, (_robot, action) in enumerate(flat):
            if index in fired or not set(action["pre"]) <= reachable:
                continue
            fired.add(index)
            new_facts = set(action["add"]) - reachable
            if new_facts:
                reachable |= new_facts
                changed = True
    kept: list[list[dict[str, Any]]] = [[] for _ in grounded_by_robot]
    for index, (robot, action) in enumerate(flat):
        if index in fired:
            kept[robot].append(action)
    return kept


def _ground_robot_actions(robot, domain: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    actions: list[dict[str, Any]] = []
    for cap in robot.capabilities:
        for combo in product(domain, repeat=len(cap.parameters)):
            bindings = dict(zip(cap.parameters, combo, strict=True))
            name = cap.name if not combo else f"{cap.name}({','.join(combo)})"
            if name in seen:
                continue
            seen.add(name)
            actions.append(
                {
                    "name": name,
                    "pre": sorted({substitute(p, bindings) for p in cap.preconditions}),
                    "add": sorted({substitute(a, bindings) for a in cap.effects.add}),
                    "cost": 1,
                    # Grounded delete patterns; expanded to concrete facts by port_scenario.
                    "_del_patterns": [substitute(d, bindings) for d in cap.effects.delete],
                }
            )
    return actions
