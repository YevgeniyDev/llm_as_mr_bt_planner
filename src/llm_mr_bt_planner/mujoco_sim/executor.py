"""Execute the accepted courier BT using measured MuJoCo state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bt import BTNode, iter_leaves
from ..domain import Capability, Scenario
from ..plan import Plan
from ..predicates import parse_predicate, substitute
from .controllers import ArmController, ContactGaitController
from .world import DESTINATION_DOCK_X, DOCK_Y, SOURCE_DOCK_X, CourierWorld

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
    leaves: list[BTNode]
    index: int = 0
    entered_at: float | None = None
    action: "PhysicalAction | None" = None
    status: str = "RUNNING"

    @property
    def current(self) -> BTNode | None:
        return self.leaves[self.index] if self.index < len(self.leaves) else None


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
                "executed": "exact BT leaves through MuJoCo controllers and observed predicates",
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
            robot: RobotCursor(robot, list(iter_leaves(tree))) for robot, tree in plan.behavior_trees.items()
        }
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
                self._tick_cursor(cursor, dt)
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
        if reason is None:
            reason = self.failed_reason or ("All physical goals were observed." if all(goals.values()) else "Goals missing.")
        success = self.complete and not self.failed and all(goals.values())
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
        if name == "docked":
            dock_x = SOURCE_DOCK_X if parameters[1] == "source_dock" else DESTINATION_DOCK_X
            xy = self.world.base_position[:2]
            return bool(np.linalg.norm(xy - [dock_x, DOCK_Y]) < 0.18 and self.observe_literal("base_stationary(unitree_go2_z1)"))
        if name == "holding":
            return parameters[1] == "payload" and self.world.equality_active(parameters[0])
        if name == "gripper_empty":
            return not self.world.equality_active(parameters[0])
        if name == "installed":
            return self.world.equality_active("target_fixture") and self._payload_near("target_fixture", 0.08)
        if name == "at" and parameters[0] == "payload":
            location = parameters[1]
            if any(self.world.equality_active(owner) for owner in self.arms):
                return False
            return self._payload_near(location, 0.085)
        return False

    def _tick_cursor(self, cursor: RobotCursor, dt: float) -> None:
        # A cursor can consume multiple immediate wait/resource leaves in one physics step.
        for _ in range(4):
            node = cursor.current
            if node is None:
                cursor.status = "SUCCESS"
                self._event(cursor.robot, "tree_success", "Behavior Tree completed")
                return
            if cursor.entered_at is None:
                cursor.entered_at = float(self.world.data.time)
                self._event(cursor.robot, "node_started", node.label(), node_id=node.node_id)

            if node.type in {"Condition", "WaitFor"}:
                literal = node.label()
                if self.observe_literal(literal):
                    self._advance(cursor, node)
                    continue
                if self._timed_out(cursor, node.timeout_ticks):
                    self._fail(f"{cursor.robot} timed out waiting for measured predicate {literal}.")
                return

            if node.type == "AcquireResource":
                owner = self.resources.get(node.name or "")
                if owner in {None, cursor.robot}:
                    self.resources[node.name or ""] = cursor.robot
                    self._event(cursor.robot, "resource_acquired", node.name or "")
                    self._advance(cursor, node)
                    continue
                if self._timed_out(cursor, node.timeout_ticks):
                    self._fail(f"{cursor.robot} timed out acquiring physical resource {node.name}.")
                return

            if node.type == "ReleaseResource":
                resource = node.name or ""
                if self.resources.get(resource) != cursor.robot:
                    self._fail(f"{cursor.robot} attempted to release unowned resource {resource}.")
                    return
                del self.resources[resource]
                self._event(cursor.robot, "resource_released", resource)
                self._advance(cursor, node)
                continue

            if node.type == "Action":
                self._tick_action(cursor, node, dt)
                return

            self._fail(f"Physical executor does not support BT node type {node.type}.")
            return

    def _tick_action(self, cursor: RobotCursor, node: BTNode, dt: float) -> None:
        capability = self.scenario.capability(cursor.robot, node.name or "")
        if capability is None:
            self._fail(f"No scenario capability backs {cursor.robot}/{node.name}.")
            return
        if cursor.action is None:
            missing = self._missing_preconditions(cursor.robot, capability, node.parameters)
            if missing:
                # Physical plants need to settle between commands.  Keep this
                # Action leaf RUNNING while its measured guard is false; do not
                # apply a symbolic effect or skip the guard.
                if float(self.world.data.time) - (cursor.entered_at or 0.0) > capability.timeout_ticks:
                    self._fail(
                        f"Physical preconditions for {cursor.robot}/{node.label()} remained false for "
                        f"{capability.timeout_ticks}s: {', '.join(missing)}."
                    )
                return
            cursor.action = PhysicalAction(self, cursor.robot, node.name or "", node.parameters)
            self._event(cursor.robot, "action_started", node.label(), node_id=node.node_id)
        result, detail = cursor.action.step(dt)
        if result == "SUCCESS":
            self._event(cursor.robot, "action_success", node.label(), detail=detail, node_id=node.node_id)
            cursor.action = None
            self._advance(cursor, node)
        elif result == "FAILURE":
            self._fail(f"Physical action {cursor.robot}/{node.label()} failed: {detail}")
        elif float(self.world.data.time) - (cursor.entered_at or 0.0) > capability.timeout_ticks:
            self._fail(
                f"Physical action {cursor.robot}/{node.label()} exceeded its {capability.timeout_ticks}s timeout; "
                f"last stage: {cursor.action.stage}."
            )

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

    def _advance(self, cursor: RobotCursor, node: BTNode) -> None:
        self._event(cursor.robot, "node_success", node.label(), node_id=node.node_id)
        cursor.index += 1
        cursor.entered_at = None

    def _timed_out(self, cursor: RobotCursor, timeout_ticks: int | None) -> bool:
        if timeout_ticks is None:
            return False
        return float(self.world.data.time) - (cursor.entered_at or 0.0) > timeout_ticks * BT_TICK_SECONDS

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
        if kind in {"action_started", "action_success", "resource_acquired", "resource_released", "tree_success"}:
            self.progress(f"[{event['time']:7.2f}s] {robot}: {kind.replace('_', ' ')} — {message}")

    def _fail(self, message: str) -> None:
        if self.failed_reason is None:
            self.failed_reason = message
            self.events.append(
                {"time": round(float(self.world.data.time), 4), "robot": "system", "kind": "failure", "message": message}
            )
            self.progress(f"[{self.world.data.time:7.2f}s] FAILURE — {message}")


class PhysicalAction:
    def __init__(self, executor: PhysicalExecutor, robot: str, name: str, parameters: tuple[str, ...]) -> None:
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

    def step(self, dt: float) -> tuple[str, str]:
        if self.name.startswith("pick_"):
            location = {
                "pick_source": "source_bin",
                "pick_source_cradle": "source_cradle",
                "pick_destination_cradle": "destination_cradle",
            }[self.name]
            return self._pick(location, dt)
        if self.name in {"place_source_cradle", "place_destination_cradle", "install_target"}:
            location = {
                "place_source_cradle": "source_cradle",
                "place_destination_cradle": "destination_cradle",
                "install_target": "target_fixture",
            }[self.name]
            lock = "target_fixture" if location == "target_fixture" else None
            return self._place(location, lock, dt)
        if self.name.startswith("stow_arm"):
            assert self.arm is not None
            self.executor.used_arms.add(self.robot)
            self.arm.set_gripper(closed=False)
            if self.arm.move_home(dt):
                return "SUCCESS", "Z1 joint state is within the measured stow tolerance"
            return "RUNNING", "moving to stow posture"
        if self.name == "navigate_destination":
            return self._navigate(dt)
        return "FAILURE", f"No physical controller adapter exists for action {self.name}"

    def _pick(self, location: str, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        payload = self.world.payload_position
        if self.pick_origin is None:
            self.pick_origin = payload.copy()
        grasp_target = payload if self.stage in {"descend", "close"} else self.pick_origin
        if self.stage == "start":
            self.arm.set_gripper(closed=False)
            self._next("open")
        if self.stage == "open":
            self.arm.hold()
            if self.arm.gripper_opened():
                self._next("prealign" if self.robot == "unitree_go2_z1" else "approach")
            elif self._elapsed() > 1.5:
                return "FAILURE", "gripper did not reach its measured open position"
        elif self.stage == "prealign":
            # Complete the Z1 wrist rotation on the Go2 side of the table,
            # then approach horizontally with both open fingers clear.
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
            approach_offset = np.array([0.0, 0.07, 0.0]) if self.robot == "unitree_go2_z1" else np.array([0.0, 0.0, 0.11])
            approach_target = grasp_target + approach_offset
            reached = self._move_tool(approach_target, dt)
            pose_aligned = reached or (
                self.robot == "unitree_go2_z1"
                and self.arm.tool_pose_reached(
                    approach_target,
                    target_rotation=Z1_GRASP_ROTATION,
                    position_tolerance=0.018,
                    rotation_tolerance=0.12,
                )
            )
            if pose_aligned:
                self._next("descend")
        elif self.stage == "descend":
            reached = self._move_tool(grasp_target, dt)
            distance = float(np.linalg.norm(self.arm.tool_position - payload))
            payload_drift = float(np.linalg.norm(payload - self.pick_origin))
            if payload_drift > 0.06:
                return "FAILURE", f"payload moved {payload_drift:.3f}m before a valid grasp was established"
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
            distance = float(np.linalg.norm(self.arm.tool_position - payload))
            payload_drift = float(np.linalg.norm(payload - grasp_target))
            contact = self.world.payload_contact_with(self.arm.body_prefix)
            z1_pinch = self.world.payload_contact_with_z1_finger_pad(
                "fixed"
            ) and self.world.payload_contact_with_z1_finger_pad("moving")
            proximity_limit = 0.060 if self.robot == "unitree_go2_z1" else 0.032
            if (
                self._elapsed() > 0.35
                and distance < (0.10 if self.robot == "unitree_go2_z1" else 0.045)
                and (z1_pinch if self.robot == "unitree_go2_z1" else contact or distance < proximity_limit)
                and self.arm.gripper_closed()
            ):
                self.world.activate_weld(self.robot)
                self.world.set_cradle_holding_friction(location, enabled=False)
                self._next("lift")
            elif payload_drift > 0.06:
                return "FAILURE", f"payload moved {payload_drift:.3f}m before a valid grasp was established"
            elif self._elapsed() > 3.0:
                return "FAILURE", (
                    f"grasp checks failed (tool distance {distance:.3f}m, contact={contact}, "
                    f"opposing_finger_contact={z1_pinch if self.robot == 'unitree_go2_z1' else 'n/a'}, "
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
            lift_height = float(self.world.payload_position[2] - self.pick_origin[2])
            if self.world.equality_active(self.robot) and lift_height > 0.05 and (
                reached or self.robot == "unitree_go2_z1"
            ):
                return "SUCCESS", "contact-qualified grasp remained constrained during lift"
        return "RUNNING", self.stage

    def _place(self, location: str, lock: str | None, dt: float) -> tuple[str, str]:
        assert self.arm is not None
        self.executor.used_arms.add(self.robot)
        target = self.world.site_position(location)
        if self.stage == "start":
            self.arm.set_gripper(closed=True)
            self.place_tool_offset = (
                np.zeros(3)
                if self.robot == "unitree_go2_z1"
                else self.arm.tool_position - self.world.payload_position
            )
            self._next("approach")
        assert self.place_tool_offset is not None
        tool_target = target + self.place_tool_offset
        if self.stage == "approach":
            approach_offset = np.array([0.0, 0.08, 0.08]) if self.robot == "unitree_go2_z1" else np.array([0.0, 0.0, 0.11])
            if self._move_tool(tool_target + approach_offset, dt):
                if self.robot == "unitree_go2_z1":
                    self.place_tool_offset = self.arm.tool_position - self.world.payload_position
                self._next("lower")
        elif self.stage == "lower":
            payload_error = target - self.world.payload_position
            # Close the loop on the payload pose, not merely the gripper site:
            # the captured grasp orientation can rotate their relative offset.
            tool_target = target + self.place_tool_offset
            self._move_tool(tool_target, dt)
            alignment_limit = 0.035 if self.robot == "unitree_go2_z1" else 0.035
            payload_aligned = float(np.linalg.norm(payload_error)) < alignment_limit
            if payload_aligned:
                self.world.set_cradle_holding_friction(location, enabled=True)
                if lock:
                    self.world.activate_weld(lock)
                if self.robot != "unitree_go2_z1":
                    self.world.deactivate_weld(self.robot)
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
                    self.world.deactivate_weld(self.robot)
                self._next("settle_release")
            elif self._elapsed() > 1.5:
                return "FAILURE", "gripper did not reach its measured open position during release"
        elif self.stage == "settle_release":
            self._move_tool(tool_target, dt)
            if self._elapsed() > 0.35:
                if self.executor.observe_literal(f"at(payload,{location})") or (
                    location == "target_fixture" and self.executor.observe_literal("installed(payload,target_fixture)")
                ):
                    self._next("retreat")
                else:
                    return "FAILURE", f"payload did not settle at {location} after measured gripper opening"
        elif self.stage == "retreat":
            retreat_offset = np.array([0.0, 0.08, 0.08]) if self.robot == "unitree_go2_z1" else np.array([0.0, 0.0, 0.10])
            if self._move_tool(tool_target + retreat_offset, dt):
                if self.executor.observe_literal(f"at(payload,{location})") or (
                    location == "target_fixture" and self.executor.observe_literal("installed(payload,target_fixture)")
                ):
                    if self.robot.startswith("franka_"):
                        self._next("home")
                    else:
                        return "SUCCESS", f"payload pose was observed at {location} after measured release"
                else:
                    return "FAILURE", f"payload was not physically observed at {location} after release"
        elif self.stage == "home":
            self.arm.set_gripper(closed=False)
            if self.arm.move_home(dt):
                return "SUCCESS", f"payload was observed at {location} and {self.robot} returned home"
        return "RUNNING", self.stage

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
            return "FAILURE", "the payload left the Z1 grasp before or during locomotion"
        self.arm.set_gripper(closed=True)
        if self.stage == "start":
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
                self.executor.gait.navigate_to_destination()
                self._next("walk")
        elif self.stage == "walk":
            self.arm.hold()
            if not self.arm.gripper_closed():
                return "FAILURE", "the measured Z1 gripper position opened during locomotion"
            if self.executor.gait.reached_destination():
                return "SUCCESS", "Go2 reached the destination while the payload remained in the closed Z1 grasp"
        return "RUNNING", self.stage

    def _next(self, stage: str) -> None:
        self.stage = stage
        self.stage_started = float(self.world.data.time)

    def _elapsed(self) -> float:
        return float(self.world.data.time) - self.stage_started
