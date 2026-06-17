"""Symbolic multi-robot Behavior Tree executor.

Each robot's tree is ticked once per global tick (round-robin). Composites use
the standard memory semantics of reactive Behavior Trees:

* a ``Sequence`` ticks children left to right, stops at the first non-SUCCESS,
  and resumes there next tick;
* a ``Fallback`` ticks until the first non-FAILURE;
* a ``Parallel`` ticks all children and succeeds when ``success_threshold`` of
  them succeed.

Leaves model multi-robot synchronization as *blocking guards*: a ``Condition``
whose predicate does not hold, or an ``Action`` whose preconditions are not yet
met, returns ``RUNNING`` (the robot waits) rather than ``FAILURE``. A whole
global tick that changes nothing while some tree is still running is a deadlock -
the only thing that can change state is an executed action, so an unchanged
state guarantees every future tick would be identical.

To keep the execution trace a readable step-by-step timeline, each robot
executes at most ``actions_per_tick`` actions per global tick (default 1);
Conditions still resolve freely within a tick. So one global tick is one
synchronized round of robot actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bt import BTNode, Status
from .domain import Scenario, apply_grounded, ground_effects
from .plan import Plan
from .predicates import substitute


@dataclass
class SimulationReport:
    success: bool
    goal_success: bool
    final_state: list[str]
    trace: list[dict[str, Any]]
    errors: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal_success": self.goal_success,
            "final_state": self.final_state,
            "trace": self.trace,
            "errors": self.errors,
        }


@dataclass
class _Context:
    scenario: Scenario
    state: set[str]
    tick: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)
    blocked: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_budget: int = 1


def simulate(plan: Plan, scenario: Scenario, max_ticks: int = 80, actions_per_tick: int = 1) -> SimulationReport:
    ctx = _Context(scenario=scenario, state=set(scenario.initial_state))
    trees = plan.behavior_trees
    memories: dict[str, dict[int, Any]] = {robot: {} for robot in trees}
    done: dict[str, bool] = {robot: False for robot in trees}

    for tick in range(1, max_ticks + 1):
        if all(done.values()):
            return _result(ctx, scenario)

        ctx.tick = tick
        ctx.blocked.clear()
        snapshot = frozenset(ctx.state)

        for robot_id, tree in trees.items():
            if done[robot_id]:
                continue
            ctx.action_budget = actions_per_tick
            status = _tick(tree, robot_id, ctx, memories[robot_id])
            if status in (Status.SUCCESS, Status.FAILURE):
                done[robot_id] = True

        if all(done.values()):
            return _result(ctx, scenario)
        if frozenset(ctx.state) == snapshot:
            errors = [{"type": "deadlock", "waiting": list(ctx.blocked.values())}]
            return _result(ctx, scenario, errors)

    errors = [{"type": "timeout", "message": f"Simulation exceeded {max_ticks} ticks."}]
    return _result(ctx, scenario, errors)


def _tick(node: BTNode, robot_id: str, ctx: _Context, memory: dict[int, Any]) -> Status:
    if node.type == "Sequence":
        return _tick_sequence(node, robot_id, ctx, memory)
    if node.type == "Fallback":
        return _tick_fallback(node, robot_id, ctx, memory)
    if node.type == "Parallel":
        return _tick_parallel(node, robot_id, ctx, memory)
    if node.type == "Action":
        return _tick_action(node, robot_id, ctx)
    if node.type == "Condition":
        return _tick_condition(node, robot_id, ctx)
    return Status.FAILURE


def _tick_sequence(node: BTNode, robot_id: str, ctx: _Context, memory: dict[int, Any]) -> Status:
    start = memory.get(id(node), 0)
    for index in range(start, len(node.children)):
        status = _tick(node.children[index], robot_id, ctx, memory)
        if status is Status.RUNNING:
            memory[id(node)] = index
            return Status.RUNNING
        if status is Status.FAILURE:
            memory[id(node)] = 0
            return Status.FAILURE
    memory[id(node)] = 0
    return Status.SUCCESS


def _tick_fallback(node: BTNode, robot_id: str, ctx: _Context, memory: dict[int, Any]) -> Status:
    start = memory.get(id(node), 0)
    for index in range(start, len(node.children)):
        status = _tick(node.children[index], robot_id, ctx, memory)
        if status is Status.RUNNING:
            memory[id(node)] = index
            return Status.RUNNING
        if status is Status.SUCCESS:
            memory[id(node)] = 0
            return Status.SUCCESS
    memory[id(node)] = 0
    return Status.FAILURE


def _tick_parallel(node: BTNode, robot_id: str, ctx: _Context, memory: dict[int, Any]) -> Status:
    children = node.children
    threshold = node.success_threshold if node.success_threshold is not None else len(children)
    # Reactive Parallel *with memory*: latch children that have already returned a
    # terminal status. A succeeded/failed child is not re-ticked until the whole
    # Parallel resets, so its one-shot action effects aren't re-applied (which
    # could otherwise block it forever once its preconditions are consumed) and
    # its trace entry isn't duplicated on every later tick.
    completed: dict[int, Status] = memory.get(id(node), {})
    for index, child in enumerate(children):
        if index in completed:
            continue
        status = _tick(child, robot_id, ctx, memory)
        if status in (Status.SUCCESS, Status.FAILURE):
            completed[index] = status
    successes = sum(1 for status in completed.values() if status is Status.SUCCESS)
    failures = sum(1 for status in completed.values() if status is Status.FAILURE)
    if successes >= threshold:
        memory[id(node)] = {}
        return Status.SUCCESS
    if failures > len(children) - threshold:
        memory[id(node)] = {}
        return Status.FAILURE
    memory[id(node)] = completed
    return Status.RUNNING


def _tick_action(node: BTNode, robot_id: str, ctx: _Context) -> Status:
    if ctx.action_budget <= 0:
        # Already used this tick's action budget; resume at this action next tick.
        return Status.RUNNING

    capability = ctx.scenario.capability(robot_id, node.name or "")
    if capability is None:
        ctx.blocked[robot_id] = {"robot": robot_id, "action": node.label(), "missing_preconditions": ["unknown_capability"]}
        return Status.RUNNING

    bindings = dict(zip(capability.parameters, node.parameters))
    missing = [
        substitute(pre, bindings)
        for pre in capability.preconditions
        if substitute(pre, bindings) not in ctx.state
    ]
    if missing:
        ctx.blocked[robot_id] = {"robot": robot_id, "action": node.label(), "missing_preconditions": missing}
        return Status.RUNNING

    adds, deletes = ground_effects(capability.effects, bindings)
    apply_grounded(ctx.state, adds, deletes)
    ctx.action_budget -= 1
    ctx.trace.append(
        {
            "tick": ctx.tick,
            "robot": robot_id,
            "event": "action",
            "action": node.label(),
            "effects": {"add": adds, "delete": deletes},
        }
    )
    return Status.SUCCESS


def _tick_condition(node: BTNode, robot_id: str, ctx: _Context) -> Status:
    predicate = node.label()
    if predicate in ctx.state:
        ctx.trace.append({"tick": ctx.tick, "robot": robot_id, "event": "condition", "condition": predicate})
        return Status.SUCCESS
    ctx.blocked[robot_id] = {"robot": robot_id, "condition": predicate}
    return Status.RUNNING


def _result(ctx: _Context, scenario: Scenario, errors: list[dict[str, Any]] | None = None) -> SimulationReport:
    errors = errors or []
    goal_success = all(goal in ctx.state for goal in scenario.goal_state)
    return SimulationReport(
        success=not errors and goal_success,
        goal_success=goal_success,
        final_state=sorted(ctx.state),
        trace=ctx.trace,
        errors=errors,
    )


def skipped_simulation() -> SimulationReport:
    return SimulationReport(
        success=False,
        goal_success=False,
        final_state=[],
        trace=[],
        errors=[{"type": "validation_failed", "message": "Simulation skipped because validation failed."}],
    )
