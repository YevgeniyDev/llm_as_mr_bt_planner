"""Symbolic multi-robot Behavior Tree executor.

Each robot's tree is ticked once per global tick (round-robin). Composites use
the standard memory semantics of reactive Behavior Trees:

* a ``Sequence`` ticks children left to right, stops at the first non-SUCCESS,
  and resumes there next tick;
* a ``Fallback`` ticks until the first non-FAILURE;
* a ``Parallel`` ticks all children and succeeds when ``success_threshold`` of
  them succeed.

Leaves follow standard, explicit contracts. ``Condition`` returns FAILURE when
its predicate is false. ``WaitFor`` is the only leaf that returns RUNNING while
a predicate is false, and it has a mandatory finite timeout in validated plans.
An ``Action`` with unmet declared preconditions fails immediately; it never
silently becomes a synchronization primitive.

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
    runtime_errors: list[dict[str, Any]] = field(default_factory=list)
    wait_counts: dict[tuple[str, str], int] = field(default_factory=dict)
    action_progress: dict[tuple[str, str], int] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)


def simulate(
    plan: Plan,
    scenario: Scenario,
    max_ticks: int = 80,
    actions_per_tick: int = 1,
    cancel_at_tick: int | None = None,
) -> SimulationReport:
    """Tick the exact LLM-generated trees to completion or a proven failure."""
    ctx = _Context(scenario=scenario, state=set(scenario.initial_state))
    trees = plan.behavior_trees
    if not trees:
        return _result(ctx, scenario, [{"type": "no_behavior_trees", "message": "Plan has no behavior trees."}])
    memories: dict[str, dict[int, Any]] = {robot: {} for robot in trees}
    done: dict[str, bool] = {robot: False for robot in trees}

    for tick in range(1, max_ticks + 1):
        if cancel_at_tick is not None and tick >= cancel_at_tick:
            ctx.tick = tick
            _cancel_and_release(ctx)
            return _result(
                ctx,
                scenario,
                [{"type": "cancelled", "message": f"Simulation cancelled at tick {tick}."}],
            )
        if all(done.values()):
            return _result(ctx, scenario)

        ctx.tick = tick
        ctx.blocked.clear()
        for robot_id, tree in trees.items():
            if done[robot_id]:
                continue
            ctx.action_budget = actions_per_tick
            status = _tick(tree, robot_id, ctx, memories[robot_id])
            if status is Status.FAILURE:
                if not any(error.get("robot") == robot_id for error in ctx.runtime_errors):
                    ctx.runtime_errors.append(
                        {"type": "tree_failure", "robot": robot_id, "message": "Behavior tree returned FAILURE."}
                    )
                done[robot_id] = True
            elif status is Status.SUCCESS:
                done[robot_id] = True

        if all(done.values()):
            return _result(ctx, scenario)
        deadlock = _proven_wait_deadlock(plan, scenario, ctx.blocked, done, ctx.state)
        if deadlock is not None:
            return _result(ctx, scenario, [deadlock])

    errors = [{"type": "timeout", "message": f"Simulation exceeded {max_ticks} ticks."}]
    return _result(ctx, scenario, errors)


def _tick(node: BTNode, robot_id: str, ctx: _Context, memory: dict[int, Any]) -> Status:
    if node.type == "Sequence":
        return _tick_sequence(node, robot_id, ctx, memory)
    if node.type == "ReactiveSequence":
        return _tick_reactive_sequence(node, robot_id, ctx, memory)
    if node.type == "Fallback":
        return _tick_fallback(node, robot_id, ctx, memory)
    if node.type == "Parallel":
        return _tick_parallel(node, robot_id, ctx, memory)
    if node.type == "ParallelAll":
        return _tick_parallel(node, robot_id, ctx, memory)
    if node.type == "Action":
        return _tick_action(node, robot_id, ctx)
    if node.type == "Condition":
        return _tick_condition(node, robot_id, ctx)
    if node.type == "WaitFor":
        return _tick_wait_for(node, robot_id, ctx)
    if node.type == "AcquireResource":
        return _tick_acquire_resource(node, robot_id, ctx)
    if node.type == "ReleaseResource":
        return _tick_release_resource(node, robot_id, ctx)
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


def _tick_reactive_sequence(node: BTNode, robot_id: str, ctx: _Context, memory: dict[int, Any]) -> Status:
    for child in node.children:
        status = _tick(child, robot_id, ctx, memory)
        if status is not Status.SUCCESS:
            return status
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
        ctx.runtime_errors.append(
            {"type": "unknown_capability", "robot": robot_id, "action": node.label()}
        )
        return Status.FAILURE

    missing_resources = [
        resource for resource in capability.resources if ctx.resources.get(resource) != robot_id
    ]
    if missing_resources:
        error: dict[str, Any] = {
            "type": "resource_not_owned",
            "robot": robot_id,
            "action": node.label(),
            "resources": missing_resources,
        }
        ctx.runtime_errors.append(error)
        return Status.FAILURE

    bindings = dict(zip(capability.parameters, node.parameters))
    missing = [
        substitute(pre, bindings)
        for pre in capability.preconditions
        if substitute(pre, bindings) not in ctx.state
    ]
    if missing:
        error = {
            "type": "precondition_failure",
            "robot": robot_id,
            "action": node.label(),
            "missing_preconditions": missing,
        }
        ctx.runtime_errors.append(error)
        ctx.trace.append({"tick": ctx.tick, "event": "action_rejected", **error})
        return Status.FAILURE

    key = (robot_id, node.node_id or node.label())
    elapsed = ctx.action_progress.get(key, 0) + 1
    ctx.action_progress[key] = elapsed
    timeout = capability.timeout_ticks
    if elapsed > timeout:
        error = {
            "type": "action_timeout",
            "robot": robot_id,
            "action": node.label(),
            "elapsed_ticks": elapsed,
            "timeout_ticks": timeout,
        }
        ctx.runtime_errors.append(error)
        ctx.trace.append({"tick": ctx.tick, "event": "action_timeout", **error})
        return Status.FAILURE
    duration = capability.duration_ticks
    if elapsed < duration:
        ctx.action_budget -= 1
        ctx.trace.append(
            {
                "tick": ctx.tick,
                "robot": robot_id,
                "event": "action_running",
                "action": node.label(),
                "elapsed_ticks": elapsed,
                "duration_ticks": duration,
            }
        )
        return Status.RUNNING

    adds, deletes = ground_effects(capability.effects, bindings)
    event = {
        "tick": ctx.tick,
        "robot": robot_id,
        "event": "action",
        "action": node.label(),
        "name": node.name,
        "parameters": list(node.parameters),
        "effects": {"add": adds, "delete": deletes},
    }
    ctx.action_budget -= 1

    previous_state = set(ctx.state)
    apply_grounded(ctx.state, adds, deletes)
    invariant_errors = _state_invariant_errors(ctx.state, ctx.scenario)
    if invariant_errors:
        ctx.state.clear()
        ctx.state.update(previous_state)
        error = {
            "type": "state_invariant_violation",
            "robot": robot_id,
            "action": node.label(),
            "violations": invariant_errors,
        }
        ctx.runtime_errors.append(error)
        ctx.trace.append({"tick": ctx.tick, "event": "action_rejected", **error})
        ctx.action_progress.pop(key, None)
        return Status.FAILURE
    ctx.action_progress.pop(key, None)
    ctx.trace.append(event)
    return Status.SUCCESS


def _tick_condition(node: BTNode, robot_id: str, ctx: _Context) -> Status:
    predicate = node.label()
    if predicate in ctx.state:
        ctx.trace.append({"tick": ctx.tick, "robot": robot_id, "event": "condition", "condition": predicate})
        return Status.SUCCESS
    ctx.trace.append(
        {"tick": ctx.tick, "event": "condition_failed", "robot": robot_id, "condition": predicate}
    )
    return Status.FAILURE


def _tick_wait_for(node: BTNode, robot_id: str, ctx: _Context) -> Status:
    predicate = node.label()
    key = (robot_id, node.node_id or predicate)
    if predicate in ctx.state:
        ctx.wait_counts.pop(key, None)
        ctx.trace.append({"tick": ctx.tick, "robot": robot_id, "event": "wait_satisfied", "condition": predicate})
        return Status.SUCCESS
    count = ctx.wait_counts.get(key, 0) + 1
    ctx.wait_counts[key] = count
    waiting = {
        "robot": robot_id,
        "node_id": node.node_id,
        "condition": predicate,
        "waited_ticks": count,
        "timeout_ticks": node.timeout_ticks,
    }
    if node.timeout_ticks is not None and count >= node.timeout_ticks:
        error = {"type": "wait_timeout", **waiting}
        ctx.runtime_errors.append(error)
        ctx.trace.append({"tick": ctx.tick, "event": "wait_timeout", **waiting})
        return Status.FAILURE
    ctx.blocked[robot_id] = {"kind": "predicate", **waiting}
    return Status.RUNNING


def _tick_acquire_resource(node: BTNode, robot_id: str, ctx: _Context) -> Status:
    resource = node.name or ""
    if resource not in ctx.scenario.resource_ids:
        ctx.runtime_errors.append(
            {"type": "unknown_resource", "robot": robot_id, "resource": resource}
        )
        return Status.FAILURE
    owner = ctx.resources.get(resource)
    if owner in {None, robot_id}:
        if owner is None:
            ctx.resources[resource] = robot_id
            ctx.state.add(f"owns_resource({robot_id}, {resource})")
            ctx.trace.append(
                {"tick": ctx.tick, "robot": robot_id, "event": "resource_acquired", "resource": resource}
            )
        return Status.SUCCESS
    key = (robot_id, node.node_id or resource)
    count = ctx.wait_counts.get(key, 0) + 1
    ctx.wait_counts[key] = count
    waiting = {
        "kind": "resource",
        "robot": robot_id,
        "node_id": node.node_id,
        "resource": resource,
        "owner": owner,
        "waited_ticks": count,
        "timeout_ticks": node.timeout_ticks,
    }
    if node.timeout_ticks is not None and count >= node.timeout_ticks:
        error = {"type": "resource_timeout", **waiting}
        ctx.runtime_errors.append(error)
        ctx.trace.append({"tick": ctx.tick, "event": "resource_timeout", **waiting})
        return Status.FAILURE
    ctx.blocked[robot_id] = waiting
    return Status.RUNNING


def _tick_release_resource(node: BTNode, robot_id: str, ctx: _Context) -> Status:
    resource = node.name or ""
    owner = ctx.resources.get(resource)
    if owner != robot_id:
        ctx.runtime_errors.append(
            {
                "type": "invalid_resource_release",
                "robot": robot_id,
                "resource": resource,
                "owner": owner,
            }
        )
        return Status.FAILURE
    del ctx.resources[resource]
    ctx.state.discard(f"owns_resource({robot_id}, {resource})")
    ctx.trace.append(
        {"tick": ctx.tick, "robot": robot_id, "event": "resource_released", "resource": resource}
    )
    return Status.SUCCESS


def _proven_wait_deadlock(
    plan: Plan,
    scenario: Scenario,
    blocked: dict[str, dict[str, Any]],
    done: dict[str, bool],
    state: set[str],
) -> dict[str, Any] | None:
    """Return a deadlock only when every live tree is waiting and the wait graph
    has no live producer or contains a closed dependency cycle.

    Merely observing an unchanged world state is not proof of deadlock: control
    flow and per-tick action budgets may still advance on the next tick.
    """
    active = {robot for robot, finished in done.items() if not finished}
    unresolved = {
        robot: wait
        for robot, wait in blocked.items()
        if wait.get("kind") == "resource" or wait.get("condition") not in state
    }
    if not active or set(unresolved) != active:
        return None

    producer_edges: dict[str, set[str]] = {robot: set() for robot in active}
    missing: list[dict[str, Any]] = []
    for consumer, wait in unresolved.items():
        if wait.get("kind") == "resource":
            owner = wait.get("owner")
            if owner in active:
                producer_edges[consumer].add(str(owner))
            else:
                missing.append(wait)
            continue
        predicate = str(wait["condition"])
        producers: set[str] = set()
        for producer, tree in plan.behavior_trees.items():
            if done.get(producer, False):
                continue
            from .bt import iter_leaves
            from .domain import positive_effects

            for leaf in iter_leaves(tree):
                capability = scenario.capability(producer, leaf.name or "") if leaf.type == "Action" else None
                if capability is None:
                    continue
                bindings = dict(zip(capability.parameters, leaf.parameters))
                if predicate in positive_effects(capability.effects, bindings):
                    producers.add(producer)
        if not producers:
            missing.append(wait)
        producer_edges[consumer].update(producers)
    if missing:
        return {
            "type": "deadlock",
            "reason": "no_live_producer",
            "waiting": list(unresolved.values()),
            "unproducible_waits": missing,
        }

    visiting: set[str] = set()
    visited: set[str] = set()

    def cyclic(robot: str) -> bool:
        if robot in visiting:
            return True
        if robot in visited:
            return False
        visiting.add(robot)
        for producer in producer_edges.get(robot, set()):
            if producer in active and cyclic(producer):
                return True
        visiting.remove(robot)
        visited.add(robot)
        return False

    if any(cyclic(robot) for robot in active):
        return {
            "type": "deadlock",
            "reason": "closed_wait_cycle",
            "waiting": list(unresolved.values()),
            "wait_graph": {robot: sorted(producers) for robot, producers in producer_edges.items()},
        }
    return None


def _result(ctx: _Context, scenario: Scenario, errors: list[dict[str, Any]] | None = None) -> SimulationReport:
    errors = [*ctx.runtime_errors, *(errors or [])]
    if ctx.resources:
        errors.append(
            {
                "type": "resource_leak",
                "message": "Simulation terminated while resources were still owned.",
                "owners": dict(sorted(ctx.resources.items())),
            }
        )
    goal_success = all(goal in ctx.state for goal in scenario.goal_state)
    return SimulationReport(
        success=not errors and goal_success,
        goal_success=goal_success,
        final_state=sorted(ctx.state),
        trace=ctx.trace,
        errors=errors,
    )


def _state_invariant_errors(state: set[str], scenario: Scenario) -> list[str]:
    from .predicates import parse_predicate

    part_ids = {entity.id for entity in scenario.entities if entity.type == "part"}
    locations: dict[str, list[str]] = {part: [] for part in part_ids}
    holders: dict[str, list[str]] = {part: [] for part in part_ids}
    for fact in state:
        name, args = parse_predicate(fact)
        if name == "at" and len(args) == 2 and args[0] in part_ids:
            locations[args[0]].append(args[1])
        if name == "holding" and len(args) == 2 and args[1] in part_ids:
            holders[args[1]].append(args[0])
    errors: list[str] = []
    for part in sorted(part_ids):
        if len(locations[part]) > 1:
            errors.append(f"part '{part}' is at multiple locations: {locations[part]}")
        if len(holders[part]) > 1:
            errors.append(f"part '{part}' is held by multiple robots: {holders[part]}")
        if locations[part] and holders[part]:
            errors.append(f"part '{part}' is both placed and held")
    return errors


def _cancel_and_release(ctx: _Context) -> None:
    for resource, owner in sorted(ctx.resources.items()):
        ctx.state.discard(f"owns_resource({owner}, {resource})")
        ctx.trace.append(
            {
                "tick": ctx.tick,
                "robot": owner,
                "event": "resource_released_on_cancel",
                "resource": resource,
            }
        )
    ctx.resources.clear()
    ctx.action_progress.clear()


def skipped_simulation() -> SimulationReport:
    return SimulationReport(
        success=False,
        goal_success=False,
        final_state=[],
        trace=[],
        errors=[{"type": "validation_failed", "message": "Simulation skipped because validation failed."}],
    )
