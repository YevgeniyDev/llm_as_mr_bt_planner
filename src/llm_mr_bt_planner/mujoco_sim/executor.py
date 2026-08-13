"""Execute accepted multi-robot BTs using measured MuJoCo state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bt import BTNode, Status
from ..domain import Capability, Scenario
from ..plan import Plan
from ..predicates import canonical_predicate, parse_predicate, substitute
from .controllers import ArmController, ContactGaitController
from .world import (
    BEYOND_DOOR_X,
    DESTINATION_DOCK_X,
    DOOR_STAGING_X,
    CourierWorld,
)

# The scenario's symbolic tick duration is not a physical actuator period.  For
# this demo, one timeout tick maps conservatively to one simulated second.
BT_TICK_SECONDS = 1.0

# Local X approaches diagonally from the Go2 side and above, while local Z
# (the Z1 jaw-closing axis) remains horizontal across the payload.
Z1_GRASP_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-2**-0.5, 2**-0.5, 0.0],
        [-2**-0.5, -2**-0.5, 0.0],
    ]
)

# Keep each tabletop-mounted Panda's gripper vertical and preserve its local
# home-frame wrist heading.  Position-only IK allowed the long fingers to tilt
# through a workpiece when the robot bases were raised onto their benches.
PANDA_GRASP_ROTATIONS = {
    "franka_a": np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
    "franka_b": np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    ),
}


@dataclass
class RobotCursor:
    robot: str
    root: BTNode
    memory: dict[int, Any]
    entered_at: dict[str, float]
    action: "PhysicalAction | None" = None
    action_node_id: str | None = None
    last_failure: str | None = None
    status: str = "RUNNING"


@dataclass
class ExecutionReport:
    success: bool
    reason: str
    simulated_seconds: float
    final_goals: dict[str, bool]
    robot_status: dict[str, str]
    action_events: list[dict[str, Any]]
    locomotion: dict[str, Any]
    physics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "reason": self.reason,
            "simulated_seconds": self.simulated_seconds,
            "final_goals": self.final_goals,
            "robot_status": self.robot_status,
            "action_events": self.action_events,
            "locomotion": self.locomotion,
            "physics": self.physics,
            "scope": {
                "executed": (
                    "exact hierarchical BT control flow and selected leaves through MuJoCo controllers, "
                    "measured predicates, and explicitly verified blackboard signals"
                ),
                "not_claimed": [
                    "real-robot safety",
                    "sim-to-real controller validity",
                    "perception",
                    "collision-free operation outside this fixed scene",
                ],
                "grasp_model": (
                    "A runtime weld is enabled only after proximity, closed-gripper, and contact/proximity "
                    "checks; its relative pose is captured at activation so the payload is not teleported."
                ),
            },
        }


class PhysicalExecutor:
    def __init__(
        self,
        world: CourierWorld,
        scenario: Scenario,
        plan: Plan,
        arms: dict[str, ArmController],
        gait: ContactGaitController,
        *,
        progress=print,
    ) -> None:
        self.world = world
        self.scenario = scenario
        self.plan = plan
        self.arms = arms
        self.gait = gait
        self.progress = progress
        self.resources: dict[str, str] = {}
        self.events: list[dict[str, Any]] = []
        self.cursors = {
            robot: RobotCursor(robot, tree, {}, {}) for robot, tree in plan.behavior_trees.items()
        }
        self.part_id = next(
            (entity.id for entity in scenario.entities if entity.type == "part"),
            "payload",
        )
        # These are blackboard inputs, not inferred world effects. Geometric
        # predicates below always use measured MuJoCo state instead.
        self.signals = {canonical_predicate(literal) for literal in scenario.initial_state}
        self.failed_reason: str | None = None
        self.used_arms: set[str] = set()

    @property
    def complete(self) -> bool:
        return all(cursor.status == "SUCCESS" for cursor in self.cursors.values())

    @property
    def failed(self) -> bool:
        return self.failed_reason is not None

    def step(self, dt: float) -> None:
        if self.failed or self.complete:
            self.gait.step()
            for arm in self.arms.values():
                arm.hold()
            return
        self.used_arms.clear()
        for cursor in self.cursors.values():
            if cursor.status == "RUNNING":
                status = self._tick_node(cursor, cursor.root, dt)
                if status is Status.SUCCESS:
                    cursor.status = "SUCCESS"
                    cursor.last_failure = None
                    self._event(cursor.robot, "tree_success", "Behavior Tree completed")
                elif status is Status.FAILURE:
                    cursor.status = "FAILURE"
                    self._fail(
                        cursor.last_failure
                        or f"{cursor.robot} Behavior Tree returned FAILURE at its root."
                    )
        for robot, arm in self.arms.items():
            if robot not in self.used_arms:
                arm.hold()
        self.gait.step()
        if not self.world.finite():
            self._fail("MuJoCo produced a non-finite position or velocity.")
        elif not self.gait.upright():
            self._fail("The Go2 fell or exceeded the allowed base tilt.")

    def make_report(self, reason: str | None = None) -> ExecutionReport:
        goals = {predicate: self.observe_literal(predicate) for predicate in self.scenario.goal_state}
        resources_released = not self.resources
        if reason is None:
            reason = self.failed_reason or (
                "All physical goals were observed and every resource was released."
                if all(goals.values()) and resources_released
                else "Goals are missing or a physical resource remains owned."
            )
        success = self.complete and not self.failed and all(goals.values()) and resources_released
        packaging = self.scenario.task_id == "three_robot_packaging_delivery"
        return ExecutionReport(
            success=success,
            reason=reason,
            simulated_seconds=round(float(self.world.data.time), 4),
            final_goals=goals,
            robot_status={robot: cursor.status for robot, cursor in self.cursors.items()},
            action_events=self.events,
            locomotion=self.gait.metrics(),
            physics={
                "engine": "MuJoCo",
                "timestep_seconds": float(self.world.model.opt.timestep),
                "final_payload_position_m": self.world.payload_position.round(6).tolist(),
                "final_go2_base_position_m": self.world.base_position.round(6).tolist(),
                "final_go2_base_velocity": self.world.base_velocity.round(6).tolist(),
                "payload_dynamic_free_body": True,
                "arm_link_gravity_compensation": True,
                "go2_and_payload_gravity_enabled": True,
                "transport_mode": "payload retained in the Z1 grasp during Go2 locomotion",
                "resources_released": resources_released,
                "packaging_delivery_evidence": {
                    "package_base_position_m": self.world.object_position("payload")
                    .round(6)
                    .tolist(),
                    "package_lid_position_m": self.world.object_position("package_lid")
                    .round(6)
                    .tolist(),
                    "package_seal_constraint_active": self.world.equality_active("package_seal"),
                    "door_initially_closed": canonical_predicate("door_closed(room_door)")
                    in self.scenario.initial_state,
                    "final_door_angle_radians": round(self.world.door_angle, 6),
                    "door_physically_open": self.world.door_open(),
                    "parcel_physically_delivered": self.observe_literal(
                        "delivered(package_base,delivery_station)"
                    ),
                }
                if packaging
                else None,
            },
        )

    def observe_literal(self, literal: str) -> bool:
        name, parameters = parse_predicate(literal)
        if name in {"system_ready", "robot_ready"}:
            return True
        if name == "base_stationary":
            velocity = self.world.base_velocity
            linear = float(np.linalg.norm(velocity[:3]))
            angular = float(np.linalg.norm(velocity[3:6]))
            return linear < 0.35 and angular < 0.5 and self.gait.upright()
        if name == "arm_stowed":
            arm = self.arms[parameters[0]]
            return bool(np.max(np.abs(arm.q - arm.home)) < 0.09)
        if name == "arm_home":
            arm = self.arms[parameters[0]]
            return bool(np.max(np.abs(arm.q - arm.home)) < 0.09)
        if name == "docked":
            xy = self.world.base_position[:2]
            return bool(
                np.linalg.norm(xy - self.world.dock_position(parameters[1])) < 0.18
                and self.observe_literal("base_stationary(unitree_go2_z1)")
            )
        if name == "holding":
            grip_key = self._grip_key(parameters[0], parameters[1])
            return grip_key is not None and self.world.equality_active(grip_key)
        if name == "gripper_empty":
            return not self.world.robot_holding_any(parameters[0])
        if name == "installed":
            fixture = parameters[1]
            return self.world.equality_active(fixture) and self._payload_near(fixture, 0.08)
        if name == "at" and parameters[0] in {self.part_id, "package_lid"}:
            object_id, location = parameters
            if self._object_held(object_id):
                return False
            return self._object_near(object_id, location, 0.085)
        if name in {"package_sealed", "attached"}:
            return (
                self.scenario.task_id == "three_robot_packaging_delivery"
                and self.world.equality_active("package_seal")
            )
        if name == "door_open" and parameters == ("room_door",):
            return self.world.door_open()
        if name == "door_closed" and parameters == ("room_door",):
            return self.world.door_closed()
        if name == "delivered" and parameters == (self.part_id, "delivery_station"):
            return self._payload_near("delivery_station", 0.085) and not self._object_held(
                self.part_id
            )
        return self._has_signal(literal)

    def _tick_node(self, cursor: RobotCursor, node: BTNode, dt: float) -> Status:
        if node.type == "Sequence":
            return self._tick_sequence(cursor, node, dt)
        if node.type == "ReactiveSequence":
            for child in node.children:
                status = self._tick_node(cursor, child, dt)
                if status is not Status.SUCCESS:
                    return status
            return Status.SUCCESS
        if node.type == "Fallback":
            return self._tick_fallback(cursor, node, dt)
        if node.type in {"Parallel", "ParallelAll"}:
            return self._tick_parallel(cursor, node, dt)
        if node.type == "Condition":
            literal = node.label()
            passed = self.observe_literal(literal)
            self._event(
                cursor.robot,
                "condition_success" if passed else "condition_failure",
                literal,
                node_id=node.node_id,
            )
            return Status.SUCCESS if passed else Status.FAILURE
        if node.type == "WaitFor":
            return self._tick_wait(cursor, node)
        if node.type == "AcquireResource":
            return self._tick_acquire(cursor, node)
        if node.type == "ReleaseResource":
            return self._tick_release(cursor, node)
        if node.type == "Action":
            return self._tick_action(cursor, node, dt)
        cursor.last_failure = f"Physical executor does not support BT node type {node.type}."
        return Status.FAILURE

    def _tick_sequence(self, cursor: RobotCursor, node: BTNode, dt: float) -> Status:
        start = int(cursor.memory.get(id(node), 0))
        for index in range(start, len(node.children)):
            status = self._tick_node(cursor, node.children[index], dt)
            if status is Status.RUNNING:
                cursor.memory[id(node)] = index
                return status
            if status is Status.FAILURE:
                cursor.memory[id(node)] = 0
                return status
        cursor.memory[id(node)] = 0
        return Status.SUCCESS

    def _tick_fallback(self, cursor: RobotCursor, node: BTNode, dt: float) -> Status:
        start = int(cursor.memory.get(id(node), 0))
        for index in range(start, len(node.children)):
            status = self._tick_node(cursor, node.children[index], dt)
            if status is Status.RUNNING:
                cursor.memory[id(node)] = index
                return status
            if status is Status.SUCCESS:
                cursor.memory[id(node)] = 0
                cursor.last_failure = None
                return status
        cursor.memory[id(node)] = 0
        return Status.FAILURE

    def _tick_parallel(self, cursor: RobotCursor, node: BTNode, dt: float) -> Status:
        completed: dict[int, Status] = cursor.memory.get(id(node), {})
        threshold = node.success_threshold if node.success_threshold is not None else len(node.children)
        for index, child in enumerate(node.children):
            if index in completed:
                continue
            status = self._tick_node(cursor, child, dt)
            if status in {Status.SUCCESS, Status.FAILURE}:
                completed[index] = status
        successes = sum(status is Status.SUCCESS for status in completed.values())
        failures = sum(status is Status.FAILURE for status in completed.values())
        if successes >= threshold:
            cursor.memory[id(node)] = {}
            return Status.SUCCESS
        if failures > len(node.children) - threshold:
            cursor.memory[id(node)] = {}
            return Status.FAILURE
        cursor.memory[id(node)] = completed
        return Status.RUNNING

    def _tick_wait(self, cursor: RobotCursor, node: BTNode) -> Status:
        self._enter(cursor, node)
        literal = node.label()
        if self.observe_literal(literal):
            self._event(cursor.robot, "wait_satisfied", literal, node_id=node.node_id)
            self._leave(cursor, node)
            return Status.SUCCESS
        if self._timed_out(cursor, node):
            cursor.last_failure = (
                f"{cursor.robot} timed out waiting for measured predicate {literal}."
            )
            self._leave(cursor, node)
            return Status.FAILURE
        return Status.RUNNING

    def _tick_acquire(self, cursor: RobotCursor, node: BTNode) -> Status:
        self._enter(cursor, node)
        resource = node.name or ""
        owner = self.resources.get(resource)
        if owner in {None, cursor.robot}:
            self.resources[resource] = cursor.robot
            self._event(cursor.robot, "resource_acquired", resource, node_id=node.node_id)
            self._leave(cursor, node)
            return Status.SUCCESS
        if self._timed_out(cursor, node):
            cursor.last_failure = (
                f"{cursor.robot} timed out acquiring physical resource {resource}."
            )
            self._leave(cursor, node)
            return Status.FAILURE
        return Status.RUNNING

    def _tick_release(self, cursor: RobotCursor, node: BTNode) -> Status:
        resource = node.name or ""
        if self.resources.get(resource) != cursor.robot:
            cursor.last_failure = (
                f"{cursor.robot} attempted to release unowned resource {resource}."
            )
            return Status.FAILURE
        del self.resources[resource]
        self._event(cursor.robot, "resource_released", resource, node_id=node.node_id)
        return Status.SUCCESS

    def _tick_action(self, cursor: RobotCursor, node: BTNode, dt: float) -> Status:
        self._enter(cursor, node)
        capability = self.scenario.capability(cursor.robot, node.name or "")
        if capability is None:
            cursor.last_failure = f"No scenario capability backs {cursor.robot}/{node.name}."
            self._leave(cursor, node)
            return Status.FAILURE
        if cursor.action is None:
            missing = self._missing_preconditions(cursor.robot, capability, node.parameters)
            if missing:
                # Physical plants need to settle between commands.  Keep this
                # Action leaf RUNNING while its measured guard is false; do not
                # apply a symbolic effect or skip the guard.
                if self._elapsed(cursor, node) > capability.timeout_ticks:
                    cursor.last_failure = (
                        f"Physical preconditions for {cursor.robot}/{node.label()} remained false for "
                        f"{capability.timeout_ticks}s: {', '.join(missing)}."
                    )
                    self._leave(cursor, node)
                    return Status.FAILURE
                return Status.RUNNING
            cursor.action = PhysicalAction(self, cursor.robot, node.name or "", node.parameters)
            cursor.action_node_id = self._node_key(node)
            self._event(cursor.robot, "action_started", node.label(), node_id=node.node_id)
        elif cursor.action_node_id != self._node_key(node):
            cursor.last_failure = (
                f"{cursor.robot} attempted concurrent physical Actions on one controller."
            )
            return Status.FAILURE
        result, detail = cursor.action.step(dt)
        if result == "SUCCESS":
            self._record_measured_action_result(cursor.robot, node.name or "", node.parameters, cursor.action)
            self._event(cursor.robot, "action_success", node.label(), detail=detail, node_id=node.node_id)
            cursor.action = None
            cursor.action_node_id = None
            self._leave(cursor, node)
            return Status.SUCCESS
        elif result == "FAILURE":
            cursor.last_failure = f"Physical action {cursor.robot}/{node.label()} failed: {detail}"
            self._event(cursor.robot, "action_failure", node.label(), detail=detail, node_id=node.node_id)
            cursor.action = None
            cursor.action_node_id = None
            self._leave(cursor, node)
            return Status.FAILURE
        elif self._elapsed(cursor, node) > capability.timeout_ticks:
            cursor.last_failure = (
                f"Physical action {cursor.robot}/{node.label()} exceeded its {capability.timeout_ticks}s timeout; "
                f"last stage: {cursor.action.stage}."
            )
            self._event(
                cursor.robot,
                "action_failure",
                node.label(),
                detail=cursor.last_failure,
                node_id=node.node_id,
            )
            cursor.action = None
            cursor.action_node_id = None
            self._leave(cursor, node)
            return Status.FAILURE
        return Status.RUNNING

    def _missing_preconditions(
        self, robot: str, capability: Capability, parameters: tuple[str, ...]
    ) -> list[str]:
        bindings = dict(zip(capability.parameters, parameters))
        missing: list[str] = []
        for template in capability.preconditions:
            grounded = substitute(template, bindings)
            if not self.observe_literal(grounded):
                missing.append(grounded)
        for resource in capability.resources:
            if self.resources.get(resource) != robot:
                missing.append(f"resource_owned({resource},{robot})")
        return missing

    def _node_key(self, node: BTNode) -> str:
        return node.node_id or f"anonymous:{id(node)}"

    def _enter(self, cursor: RobotCursor, node: BTNode) -> None:
        key = self._node_key(node)
        if key not in cursor.entered_at:
            cursor.entered_at[key] = float(self.world.data.time)
            self._event(cursor.robot, "node_started", node.label(), node_id=node.node_id)

    def _leave(self, cursor: RobotCursor, node: BTNode) -> None:
        cursor.entered_at.pop(self._node_key(node), None)

    def _elapsed(self, cursor: RobotCursor, node: BTNode) -> float:
        return float(self.world.data.time) - cursor.entered_at.get(
            self._node_key(node), float(self.world.data.time)
        )

    def _timed_out(self, cursor: RobotCursor, node: BTNode) -> bool:
        if node.timeout_ticks is None:
            return False
        return self._elapsed(cursor, node) > node.timeout_ticks * BT_TICK_SECONDS

    def _record_measured_action_result(
        self,
        robot: str,
        name: str,
        parameters: tuple[str, ...],
        _action: "PhysicalAction",
    ) -> None:
        if name in {"verify_transport_readiness", "verify_delivery_readiness"}:
            signal = (
                "transport_ready"
                if name == "verify_transport_readiness"
                else "delivery_ready"
            )
            self._add_signal(f"{signal}({robot})")
        elif name == "fit_and_seal_package_lid":
            base, lid = parameters
            if self.world.equality_active("package_seal"):
                self._add_signal(f"package_sealed({base},{lid})")
                self._add_signal(f"attached({lid},{base})")
        elif name == "push_open_door_and_cross" and self.world.door_open():
            self._discard_signal(f"door_closed({parameters[1]})")
            self._add_signal(f"door_open({parameters[1]})")
        elif name == "place_parcel_at_delivery_station":
            self._add_signal(f"delivered({parameters[0]},{parameters[1]})")

    def _has_signal(self, literal: str) -> bool:
        return canonical_predicate(literal) in self.signals

    def _add_signal(self, literal: str) -> None:
        self.signals.add(canonical_predicate(literal))

    def _discard_signal(self, literal: str) -> None:
        self.signals.discard(canonical_predicate(literal))

    def _object_name(self, object_id: str) -> str:
        return "package_lid" if object_id == "package_lid" else "payload"

    def _grip_key(self, robot: str, object_id: str) -> str | None:
        if object_id == "package_lid":
            key = f"{robot}:package_lid"
            return key if key in self.world.grip_equalities else None
        return robot if robot in self.world.grip_equalities else None

    def _object_held(self, object_id: str) -> bool:
        if object_id == "package_lid":
            return any(
                self.world.equality_active(key)
                for key in self.world.grip_equalities
                if key.endswith(":package_lid")
            )
        return any(
            self.world.equality_active(robot)
            for robot in self.arms
            if robot in self.world.grip_equalities
        )

    def _object_near(self, object_id: str, location: str, tolerance: float) -> bool:
        return bool(
            np.linalg.norm(
                self.world.object_position(self._object_name(object_id))
                - self.world.site_position(location)
            )
            < tolerance
        )

    def _payload_near(self, location: str, tolerance: float) -> bool:
        return bool(np.linalg.norm(self.world.payload_position - self.world.site_position(location)) < tolerance)

    def _event(self, robot: str, kind: str, message: str, **extra: Any) -> None:
        event = {
            "time": round(float(self.world.data.time), 4),
            "robot": robot,
            "kind": kind,
            "message": message,
            **extra,
        }
        self.events.append(event)
        if kind in {
            "action_started",
            "action_success",
            "action_failure",
            "condition_success",
            "condition_failure",
            "resource_acquired",
            "resource_released",
            "tree_success",
        }:
            self.progress(f"[{event['time']:7.2f}s] {robot}: {kind.replace('_', ' ')} — {message}")

    def _fail(self, message: str) -> None:
        if self.failed_reason is None:
            self.failed_reason = message
            self.events.append(
                {"time": round(float(self.world.data.time), 4), "robot": "system", "kind": "failure", "message": message}
            )
            self.progress(f"[{self.world.data.time:7.2f}s] FAILURE — {message}")


class PhysicalAction:
    """Controller-backed action leaf for the two supported physical missions."""

    def __init__(
        self,
        executor: PhysicalExecutor,
        robot: str,
        name: str,
        parameters: tuple[str, ...],
    ) -> None:
        self.executor = executor
        self.world = executor.world
        self.robot = robot
        self.name = name
        self.parameters = parameters
        self.arm = executor.arms.get(robot)
        self.stage = "start"
        self.stage_started = float(self.world.data.time)
        self.pick_origin: np.ndarray | None = None
        self.place_tool_offset: np.ndarray | None = None
        self.object_id = "package_lid" if name in {
            "pick_package_lid",
            "fit_and_seal_package_lid",
        } else executor.part_id
        self.object_name = executor._object_name(self.object_id)

    @property
    def grip_key(self) -> str:
        key = self.executor._grip_key(self.robot, self.object_id)
        if key is None:
            raise RuntimeError(
                f"No physical grasp constraint maps {self.robot} to {self.object_id}."
            )
        return key

    def step(self, dt: float) -> tuple[str, str]:
        pick_locations = {
            "pick_source": "source_bin",
            "pick_source_cradle": "source_cradle",
            "pick_destination_cradle": "destination_cradle",
            "pick_loaded_package_base": "base_supply",
            "pick_package_lid": "lid_supply",
            "pick_sealed_parcel": "packing_station",
        }
        if self.name in pick_locations:
            return self._pick(pick_locations[self.name], dt)

        place_targets: dict[str, tuple[str, str | None]] = {
            "place_source_cradle": ("source_cradle", None),
            "place_destination_cradle": ("destination_cradle", None),
            "install_target": ("target_fixture", "target_fixture"),
            "place_base_at_packing_station": ("packing_station", None),
            "fit_and_seal_package_lid": ("lid_seal_target", "package_seal"),
            "place_parcel_at_delivery_station": ("delivery_station", None),
        }
        if self.name in place_targets:
            location, lock = place_targets[self.name]
            return self._place(location, lock, dt)

        if self.name.startswith("stow_arm") or self.name in {
            "stow_transport_arm",
            "stow_after_delivery",
        }:
            return self._stow(dt)
        if self.name in {
            "navigate_destination",
            "approach_closed_room_door",
            "push_open_door_and_cross",
            "cross_already_open_door",
            "navigate_delivery_room",
        }:
            return self._navigate(dt)
        if self.name in {"verify_transport_readiness", "verify_delivery_readiness"}:
            return self._verify_transport_readiness(dt)
        return "FAILURE", f"No physical controller adapter exists for action {self.name}"

    def _pick(self, location: str, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        object_position = self.world.object_position(self.object_name)
        if self.pick_origin is None:
            self.pick_origin = object_position.copy()
        grasp_target = (
            object_position if self.stage in {"descend", "close"} else self.pick_origin
        )
        if self.stage == "start":
            if self.name == "pick_sealed_parcel" and not self.world.equality_active(
                "package_seal"
            ):
                return "FAILURE", "the package lid was not physically sealed to the base"
            self.arm.set_gripper(closed=False)
            self._next("open")
        if self.stage == "open":
            self.arm.hold()
            if self.arm.gripper_opened():
                self._next("prealign" if self.robot == "unitree_go2_z1" else "approach")
            elif self._elapsed() > 1.5:
                return "FAILURE", "gripper did not reach its measured open position"
        elif self.stage == "prealign":
            prealign_target = grasp_target + [0, 0.10, 0.10]
            self._move_tool(prealign_target, dt)
            if self.arm.tool_pose_reached(
                prealign_target,
                target_rotation=Z1_GRASP_ROTATION,
                position_tolerance=0.025,
                rotation_tolerance=0.20,
            ):
                self._next("descend")
        elif self.stage == "approach":
            approach_offset = np.array([0.0, 0.0, 0.11])
            if self._move_tool(grasp_target + approach_offset, dt):
                self._next("descend")
        elif self.stage == "descend":
            reached = self._move_tool(grasp_target, dt)
            distance = float(np.linalg.norm(self.arm.tool_position - object_position))
            object_drift = float(np.linalg.norm(object_position - self.pick_origin))
            if object_drift > 0.06:
                return "FAILURE", (
                    f"{self.object_id} moved {object_drift:.3f}m before a valid grasp was established"
                )
            ready_to_close = (
                self.arm.tool_pose_reached(
                    grasp_target,
                    target_rotation=Z1_GRASP_ROTATION,
                    position_tolerance=0.020,
                    rotation_tolerance=0.12,
                )
                if self.robot == "unitree_go2_z1"
                else reached or distance < 0.052
            )
            if ready_to_close:
                self.arm.set_gripper(closed=True)
                self._next("close")
        elif self.stage == "close":
            self._move_tool(grasp_target, dt)
            distance = float(np.linalg.norm(self.arm.tool_position - object_position))
            object_drift = float(np.linalg.norm(object_position - grasp_target))
            contact = self.world.object_contact_with(self.object_name, self.arm.body_prefix)
            z1_pinch = self.world.payload_contact_with_z1_finger_pad(
                "fixed"
            ) and self.world.payload_contact_with_z1_finger_pad("moving")
            proximity_limit = 0.060 if self.robot == "unitree_go2_z1" else 0.032
            if (
                self._elapsed() > 0.35
                and distance < (0.10 if self.robot == "unitree_go2_z1" else 0.045)
                and (
                    z1_pinch
                    if self.robot == "unitree_go2_z1"
                    else contact or distance < proximity_limit
                )
                and self.arm.gripper_closed()
            ):
                self.world.activate_weld(self.grip_key)
                self.world.set_cradle_holding_friction(location, enabled=False)
                self._next("lift")
            elif object_drift > 0.06:
                return "FAILURE", (
                    f"{self.object_id} moved {object_drift:.3f}m before a valid grasp was established"
                )
            elif self._elapsed() > 3.0:
                return "FAILURE", (
                    f"grasp checks failed for {self.object_id} (tool distance {distance:.3f}m, "
                    f"contact={contact}, opposing_finger_contact="
                    f"{z1_pinch if self.robot == 'unitree_go2_z1' else 'n/a'}, "
                    f"gripper_closed={self.arm.gripper_closed()}, "
                    f"gripper_position={self.arm.gripper_position})"
                )
        elif self.stage == "lift":
            assert self.pick_origin is not None
            lift_offset = (
                np.array([0.0, 0.06, 0.07])
                if self.robot == "unitree_go2_z1"
                else np.array([0.0, 0.0, 0.12])
            )
            reached = self._move_tool(self.pick_origin + lift_offset, dt)
            lift_height = float(
                self.world.object_position(self.object_name)[2] - self.pick_origin[2]
            )
            if self.world.equality_active(self.grip_key) and lift_height > 0.05 and (
                reached or self.robot == "unitree_go2_z1"
            ):
                return "SUCCESS", (
                    f"contact-qualified {self.object_id} grasp remained constrained during lift"
                )
        return "RUNNING", self.stage

    def _place(self, location: str, lock: str | None, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        target = self.world.site_position(location)
        object_position = self.world.object_position(self.object_name)
        if self.stage == "start":
            if not self.world.equality_active(self.grip_key):
                return "FAILURE", f"{self.robot} no longer held {self.object_id} before placement"
            self.arm.set_gripper(closed=True)
            self.place_tool_offset = (
                np.zeros(3)
                if self.robot == "unitree_go2_z1"
                else self.arm.tool_position - object_position
            )
            self._next("approach")
        assert self.place_tool_offset is not None
        tool_target = target + self.place_tool_offset
        if self.stage == "approach":
            approach_offset = (
                np.array([0.0, 0.08, 0.08])
                if self.robot == "unitree_go2_z1"
                else np.array([0.0, 0.0, 0.11])
            )
            if self._move_tool(tool_target + approach_offset, dt):
                if self.robot == "unitree_go2_z1":
                    self.place_tool_offset = (
                        self.arm.tool_position
                        - self.world.object_position(self.object_name)
                    )
                self._next("lower")
        elif self.stage == "lower":
            object_error = target - self.world.object_position(self.object_name)
            tool_target = target + self.place_tool_offset
            self._move_tool(tool_target, dt)
            if float(np.linalg.norm(object_error)) < 0.035:
                self.world.set_cradle_holding_friction(location, enabled=True)
                if lock:
                    self.world.activate_weld(lock)
                if self.robot != "unitree_go2_z1":
                    self.world.deactivate_weld(self.grip_key)
                self.arm.set_gripper(closed=False)
                self._next("open_release" if self.robot == "unitree_go2_z1" else "release")
        elif self.stage == "release":
            self._move_tool(tool_target, dt)
            if self.arm.gripper_opened() and self._elapsed() > 0.30:
                self._next("retreat")
            elif self._elapsed() > 1.5:
                return "FAILURE", "gripper did not reach its measured open position during release"
        elif self.stage == "open_release":
            self._move_tool(tool_target, dt)
            if self.arm.gripper_opened() and self._elapsed() > 0.30:
                if not lock:
                    self.world.deactivate_weld(self.grip_key)
                self._next("settle_release")
            elif self._elapsed() > 1.5:
                return "FAILURE", "gripper did not reach its measured open position during release"
        elif self.stage == "settle_release":
            self._move_tool(tool_target, dt)
            if self._elapsed() > 0.35:
                if self._placement_observed(location, lock):
                    self._next("retreat")
                else:
                    return "FAILURE", (
                        f"{self.object_id} did not settle at {location} after measured gripper opening"
                    )
        elif self.stage == "retreat":
            retreat_offset = (
                np.array([0.0, 0.08, 0.08])
                if self.robot == "unitree_go2_z1"
                else np.array([0.0, 0.0, 0.10])
            )
            if self._move_tool(tool_target + retreat_offset, dt):
                if self._placement_observed(location, lock):
                    if self.robot.startswith("franka_"):
                        self._next("home")
                    else:
                        return "SUCCESS", (
                            f"{self.object_id} pose was observed at {location} after measured release"
                        )
                elif self._elapsed() > 1.8:
                    return "FAILURE", (
                        f"{self.object_id} was not physically observed at {location} after release"
                    )
        elif self.stage == "home":
            self.arm.set_gripper(closed=False)
            if self.arm.move_home(dt):
                return "SUCCESS", (
                    f"{self.object_id} was observed at {location} and {self.robot} returned home"
                )
        return "RUNNING", self.stage

    def _placement_observed(self, location: str, lock: str | None) -> bool:
        if lock == "package_seal":
            return self.executor.observe_literal(
                f"package_sealed({self.executor.part_id},package_lid)"
            )
        if lock is not None:
            return self.executor.observe_literal(
                f"installed({self.executor.part_id},{lock})"
            )
        return self.executor.observe_literal(f"at({self.object_id},{location})")

    def _stow(self, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        self.arm.set_gripper(closed=False)
        if self.arm.move_home(dt):
            return "SUCCESS", f"{self.robot} joint state is within the measured home tolerance"
        return "RUNNING", "moving to measured home posture"

    def _move_tool(self, target: np.ndarray, dt: float) -> bool:
        assert self.arm is not None
        rotation = (
            Z1_GRASP_ROTATION
            if self.robot == "unitree_go2_z1"
            else PANDA_GRASP_ROTATIONS.get(self.robot)
        )
        return self.arm.move_tool(target, dt, target_rotation=rotation)

    def _navigate(self, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        if not self.world.equality_active("unitree_go2_z1"):
            return "FAILURE", "the parcel left the Z1 grasp before or during locomotion"
        if self.scenario_is_packaging and not self.world.equality_active("package_seal"):
            return "FAILURE", "the package seal constraint opened before or during locomotion"
        self.arm.set_gripper(closed=True)
        target = {
            "navigate_destination": DESTINATION_DOCK_X,
            "approach_closed_room_door": DOOR_STAGING_X,
            "push_open_door_and_cross": BEYOND_DOOR_X,
            "cross_already_open_door": BEYOND_DOOR_X,
            "navigate_delivery_room": DESTINATION_DOCK_X,
        }[self.name]
        if self.stage == "start":
            if self.name == "cross_already_open_door" and not self.world.door_open():
                return "FAILURE", "the alternate doorway branch requires a physically open door"
            self._next("secure_grasp")
        if self.stage == "secure_grasp":
            self.arm.hold()
            if self.arm.gripper_closed() and self._elapsed() > 0.30:
                self._next("move_to_carry_pose")
            elif self._elapsed() > 1.5:
                return "FAILURE", "Z1 did not retain its measured closed position before locomotion"
        elif self.stage == "move_to_carry_pose":
            if self.arm.move_home(dt):
                self._next("settle_carry_pose")
        elif self.stage == "settle_carry_pose":
            self.arm.hold()
            if self._elapsed() > 0.40:
                self.executor.gait.navigate_to(target)
                self._next("walk")
        elif self.stage == "walk":
            self.arm.hold()
            if not self.arm.gripper_closed():
                return "FAILURE", "the measured Z1 gripper position opened during locomotion"
            if self.executor.gait.reached_target():
                if self.name == "push_open_door_and_cross" and not self.world.door_open():
                    return "FAILURE", (
                        "Go2 reached the far side, but the measured hinge angle did not prove the door opened"
                    )
                payload_evidence = (
                    "the sealed parcel remained in the Z1 grasp"
                    if self.scenario_is_packaging
                    else "the payload remained in the closed Z1 grasp"
                )
                return "SUCCESS", f"Go2 reached x={target:.2f}m while {payload_evidence}"
        return "RUNNING", self.stage

    @property
    def scenario_is_packaging(self) -> bool:
        return self.executor.scenario.task_id == "three_robot_packaging_delivery"

    def _verify_transport_readiness(self, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        self.arm.set_gripper(closed=False)
        home = self.arm.move_home(dt)
        stationary = self.executor.observe_literal("base_stationary(unitree_go2_z1)")
        empty = self.executor.observe_literal("gripper_empty(unitree_go2_z1)")
        if home and stationary and empty:
            return "SUCCESS", (
                "upright base, stationary velocity, stowed Z1, and open empty gripper were measured"
            )
        return "RUNNING", "measuring delivery readiness and returning Z1 to stow"

    def _next(self, stage: str) -> None:
        self.stage = stage
        self.stage_started = float(self.world.data.time)

    def _elapsed(self) -> float:
        return float(self.world.data.time) - self.stage_started
