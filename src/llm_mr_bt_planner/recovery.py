"""Execution-time recovery ladder: retry (same robot) -> reassign (another robot).

This is the runtime counterpart to the planning-time self-correction loop in
:mod:`llm_mr_bt_planner.planner`. The planner *repairs the whole plan with the
LLM before execution*; this module reacts to failures *during execution*, with
no LLM involved:

* an :class:`~llm_mr_bt_planner.simulation` ``action_oracle`` reports which
  action failed at runtime (physics, or an injected/stochastic model);
* **Tier 1 - retry**: re-attempt the failing action on the *same* robot, up to
  ``max_retries`` times;
* **Tier 2 - reassign**: if retries are exhausted, hand the failing action to
  *another* robot whose capability produces the same predicate (found via
  :func:`llm_mr_bt_planner.domain.candidate_producers`).

The controller drives repeated whole-episode simulations rather than mutating a
live tick: node memories are keyed by ``id(node)`` and per-robot ``done`` state
latches, so re-running a (possibly mutated) plan is both simpler and exactly the
seam :func:`~llm_mr_bt_planner.simulation.simulate` already exposes. Retry works
because the failure oracles keep a *persistent, monotonic* attempt tally, so an
action that has spent its failure budget stays successful when the episode is
replayed.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .bt import COMPOSITES, BTNode, Status
from .domain import ProducerSpec, Scenario, candidate_producers, positive_effects
from .plan import Plan, parse_plan
from .simulation import SimulationReport, simulate

FailureOracle = Callable[[dict[str, Any]], Status]


# --------------------------------------------------------------------------- #
# Failure oracles
# --------------------------------------------------------------------------- #


def _event_key(event: dict[str, Any], key: str) -> str:
    """Map a trace event to the oracle bucket key ('name' | 'label' | 'robot_action')."""
    if key == "label":
        return str(event.get("action", ""))
    if key == "robot_action":
        return f"{event.get('robot', '')}.{event.get('name', '')}"
    return str(event.get("name", ""))


@dataclass
class InjectedFailureOracle:
    """Deterministic oracle: fail the first ``fail_first[key]`` attempts of each
    matching action, then succeed. The attempt tally is *persistent and
    monotonic*, so replaying an episode never re-fails an action that has already
    exhausted its budget - which is what makes re-run-based retry converge.
    """

    fail_first: dict[str, int] = field(default_factory=dict)
    key: str = "name"
    _attempts: dict[str, int] = field(default_factory=dict, init=False)

    def __call__(self, event: dict[str, Any]) -> Status:
        bucket = _event_key(event, self.key)
        self._attempts[bucket] = self._attempts.get(bucket, 0) + 1
        return Status.FAILURE if self._attempts[bucket] <= self.fail_first.get(bucket, 0) else Status.SUCCESS


@dataclass
class StochasticFailureOracle:
    """Reproducible random oracle for robustness experiments.

    Each not-yet-succeeded action fails with probability ``prob`` (a float, or a
    per-key mapping). Draws are seeded by ``(seed, key, attempt)`` so a replay is
    stable, and once an action succeeds it is *sticky* (stays successful on
    replay) so re-run retry converges just like the injected oracle.
    """

    prob: float | dict[str, float] = 0.0
    seed: int = 0
    key: str = "name"
    _attempts: dict[str, int] = field(default_factory=dict, init=False)
    _succeeded: set[str] = field(default_factory=set, init=False)

    def _prob_for(self, bucket: str) -> float:
        return self.prob.get(bucket, 0.0) if isinstance(self.prob, dict) else self.prob

    def __call__(self, event: dict[str, Any]) -> Status:
        bucket = _event_key(event, self.key)
        if bucket in self._succeeded:
            return Status.SUCCESS
        self._attempts[bucket] = self._attempts.get(bucket, 0) + 1
        draw = random.Random(f"{self.seed}:{bucket}:{self._attempts[bucket]}").random()
        if draw < self._prob_for(bucket):
            return Status.FAILURE
        self._succeeded.add(bucket)
        return Status.SUCCESS


def parse_injection_spec(spec: str) -> dict[str, int]:
    """Parse a CLI ``"pick_tool:2,open_drawer:1"`` spec into ``{name: count}``."""
    out: dict[str, int] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, count = item.partition(":")
        name = name.strip()
        if not name:
            continue
        try:
            out[name] = int(count) if count.strip() else 1
        except ValueError:
            out[name] = 1
    return out


# --------------------------------------------------------------------------- #
# Recovery records
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryEvent:
    tier: str  # "retry" | "reassign"
    robot: str
    action: str
    parameters: list[str]
    attempt: int
    to_robot: str | None = None
    target_predicate: str | None = None
    outcome: str = ""  # "retried" | "reassigned" | "no_candidate"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "robot": self.robot,
            "action": self.action,
            "parameters": self.parameters,
            "attempt": self.attempt,
            "to_robot": self.to_robot,
            "target_predicate": self.target_predicate,
            "outcome": self.outcome,
        }


@dataclass
class RecoveryResult:
    success: bool
    goal_success: bool
    report: SimulationReport | None
    recovery_log: list[RecoveryEvent]
    plan: dict[str, Any]
    episodes: int
    error: str | None = None  # "retries_exhausted" | "unrecovered_failure" | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "goal_success": self.goal_success,
            "episodes": self.episodes,
            "error": self.error,
            "log": [event.to_dict() for event in self.recovery_log],
            "final_plan": self.plan,
        }


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #


@dataclass
class RecoveryController:
    """Re-run the plan under a failure oracle, escalating retry -> reassign."""

    oracle: FailureOracle
    max_retries: int = 2
    allow_reassign: bool = True
    max_ticks: int = 80
    max_episodes: int = 20

    def run(self, plan: Plan, scenario: Scenario) -> RecoveryResult:
        working = parse_plan(plan.to_dict())  # mutable clone we may rewrite on reassignment
        retries: dict[tuple[str, str, tuple[str, ...]], int] = {}
        tried: dict[tuple[str, str, tuple[str, ...]], set[str]] = {}
        log: list[RecoveryEvent] = []
        report: SimulationReport | None = None

        for episode in range(1, self.max_episodes + 1):
            report = simulate(working, scenario, max_ticks=self.max_ticks, action_oracle=self.oracle)
            if not report.failures:
                return RecoveryResult(
                    report.success, report.goal_success, report, log, working.to_dict(), episode
                )

            failure = report.failures[0]
            key = (failure["robot"], failure["name"], tuple(failure["parameters"]))
            retries[key] = retries.get(key, 0) + 1

            if retries[key] <= self.max_retries:  # Tier 1: retry, same robot
                log.append(_event("retry", failure, attempt=retries[key], outcome="retried"))
                continue

            if not self.allow_reassign:
                return RecoveryResult(
                    False, report.goal_success, report, log, working.to_dict(), episode,
                    error="retries_exhausted",
                )

            pick = self._pick_reassignment(failure, scenario, tried.get(key, set()))  # Tier 2: reassign
            if pick is None:
                log.append(_event("reassign", failure, attempt=retries[key], outcome="no_candidate"))
                return RecoveryResult(
                    False, report.goal_success, report, log, working.to_dict(), episode,
                    error="unrecovered_failure",
                )
            spec, target = pick
            _reassign(working, failure, spec)
            tried.setdefault(key, set()).add(spec.robot)
            retries[key] = 0  # fresh retry budget for the new robot
            log.append(_event(
                "reassign", failure, attempt=0, to_robot=spec.robot,
                target_predicate=target, outcome="reassigned",
            ))

        return RecoveryResult(
            False, report.goal_success if report else False, report, log,
            working.to_dict(), self.max_episodes, error="unrecovered_failure",
        )

    def _pick_reassignment(
        self, failure: dict[str, Any], scenario: Scenario, tried_robots: set[str]
    ) -> tuple[ProducerSpec, str] | None:
        capability = scenario.capability(failure["robot"], failure["name"])
        if capability is None:
            return None
        bindings = dict(zip(capability.parameters, failure["parameters"]))
        for target in positive_effects(capability.effects, bindings):
            for spec in candidate_producers(target, scenario):
                if spec.robot != failure["robot"] and spec.robot not in tried_robots:
                    return spec, target
        return None


def _event(
    tier: str,
    failure: dict[str, Any],
    *,
    attempt: int,
    to_robot: str | None = None,
    target_predicate: str | None = None,
    outcome: str,
) -> RecoveryEvent:
    return RecoveryEvent(
        tier=tier,
        robot=failure["robot"],
        action=failure["action"],
        parameters=list(failure["parameters"]),
        attempt=attempt,
        to_robot=to_robot,
        target_predicate=target_predicate,
        outcome=outcome,
    )


# --------------------------------------------------------------------------- #
# Plan mutation for reassignment
# --------------------------------------------------------------------------- #


def _reassign(plan: Plan, failure: dict[str, Any], spec: ProducerSpec) -> None:
    """Move the failed action off its robot and onto ``spec.robot``.

    Removing the failed leaf is essential - otherwise the old robot re-fails the
    same action every episode and recovery never converges. Cross-robot ordering
    is still enforced by the simulator (an action blocks until its preconditions
    hold), so the producer is simply front-loaded into the new robot's Sequence.
    """
    old_robot = failure["robot"]
    old_name = failure["name"]
    old_params = tuple(failure["parameters"])

    if old_robot in plan.behavior_trees:
        plan.behavior_trees[old_robot] = _remove_leaf(plan.behavior_trees[old_robot], old_name, old_params)

    producer = BTNode(type="Action", name=spec.action, parameters=tuple(spec.parameters))
    target = plan.behavior_trees.get(spec.robot)
    if target is None or target.type not in COMPOSITES:
        target = BTNode(type="Sequence", children=[])
        plan.behavior_trees[spec.robot] = target
    target.children.insert(0, producer)

    _retarget_assignment(plan, old_name, old_params, spec)


def _remove_leaf(node: BTNode, name: str | None, params: tuple[str, ...]) -> BTNode:
    """Return a copy of ``node`` with every matching Action leaf removed."""
    if node.type in COMPOSITES:
        kept = [
            _remove_leaf(child, name, params)
            for child in node.children
            if not (child.type == "Action" and child.name == name and tuple(child.parameters) == params)
        ]
        return BTNode(
            type=node.type, name=node.name, parameters=node.parameters,
            children=kept, success_threshold=node.success_threshold,
        )
    return node


def _retarget_assignment(
    plan: Plan, old_name: str | None, old_params: tuple[str, ...], spec: ProducerSpec
) -> None:
    """Best-effort bookkeeping so the emitted final plan stays coherent for
    inspection/visualization (not needed for re-simulation, which reads only the
    behavior trees).
    """
    for task in plan.task_graph:
        if task.get("action") == old_name and tuple(str(p) for p in task.get("parameters", [])) == old_params:
            task["action"] = spec.action
            task["parameters"] = list(spec.parameters)
            for assignment in plan.assignments:
                if assignment.get("task_id") == task.get("id"):
                    assignment["robot"] = spec.robot
            return
