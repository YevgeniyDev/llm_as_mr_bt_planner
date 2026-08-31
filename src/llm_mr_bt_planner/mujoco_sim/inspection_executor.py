"""Measured BT execution adapter for five-agent inspection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ..domain import ground_effects
from ..predicates import canonical_predicate, parse_predicate
from .executor import ExecutionReport, PhysicalExecutor, RobotCursor
from .inspection_controllers import InspectionMotionController
from .inspection_world import (
    PANDA_HOME,
    PANDA_WORK,
    Z1_DEPLOY,
    Z1_HOME,
    InspectionWorld,
)


class InspectionExecutionReport(ExecutionReport):
    def to_dict(self) -> dict[str, object]:
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
                    "the exact supplied hierarchical BTs, actuator-driven arm joint poses, measured base "
                    "docks, a measured isolation-switch joint, and pose/range-gated thermal observations"
                ),
                "abstracted": [
                    "inspection-kit and marker attachment state after the corresponding arm pose is reached",
                    "seeded deterministic temperature readings; MuJoCo does not simulate heat transfer",
                    "task-level position-servo mobile-base translation",
                ],
                "not_claimed": [
                    "real-robot safety or sim-to-real controller validity",
                    "contact-valid grasping or collision-free manipulation",
                    "B2 gait control or Husky navigation-stack fidelity",
                    "thermal perception accuracy or fault diagnosis from images",
                ],
            },
        }


class InspectionExecutor(PhysicalExecutor):
    def __init__(self, world, scenario, plan, motion, *, progress=print) -> None:
        super().__init__(cast(Any, world), scenario, plan, {}, cast(Any, motion), progress=progress)
        self.inspection_world: InspectionWorld = world
        self.motion: InspectionMotionController = motion

    def observe_literal(self, literal: str) -> bool:
        name, parameters = parse_predicate(literal)
        if name in {"system_ready", "robot_ready"}:
            return True
        if name == "stationary":
            return self.inspection_world.robot_speed(parameters[0]) < 0.10
        if name == "docked":
            return self.inspection_world.at_dock(parameters[0], parameters[1])
        if name == "stowed":
            robot = parameters[0]
            target = Z1_HOME if robot == "z1_thermal_arm" else PANDA_HOME
            return bool(np.max(np.abs(self.inspection_world.arm_q(robot) - target)) < 0.075)
        if name == "arm_home":
            return bool(np.max(np.abs(self.inspection_world.arm_q(parameters[0]) - PANDA_HOME)) < 0.075)
        if name == "asset_isolated":
            return self.inspection_world.isolated()
        if name == "leak_repaired":
            return self.inspection_world.leak_repaired()
        if name == "at" and parameters == ("repair_tool", "husky_tool_rack"):
            return self.inspection_world.kit_state == "tool_rack" and self._has_signal(literal)
        if name == "at" and parameters == ("inspection_kit", "search_floor"):
            return self.inspection_world.fallen_tool_settled() and self._has_signal(literal)
        if name == "tool_localized" and parameters == ("inspection_kit",):
            return self.inspection_world.fallen_tool_settled() and self._has_signal(literal)
        if name == "marker_attached":
            return self._has_signal(literal) and bool(
                np.linalg.norm(
                    self.inspection_world.data.mocap_pos[
                        int(self.inspection_world.model.body("inspection_marker").mocapid[0])
                    ]
                    - (
                        self.inspection_world.site_position(self.inspection_world.hidden_anomaly_site)
                        + [0, -0.10, 0]
                    )
                )
                < 0.03
            )
        return self._has_signal(literal)

    def _tick_action(self, cursor: RobotCursor, node, dt):
        # Reuse the exact BT action lifecycle/timeouts, replacing only the
        # courier-specific action implementation.
        self._enter(cursor, node)
        capability = self.scenario.capability(cursor.robot, node.name or "")
        if capability is None:
            cursor.last_failure = f"No scenario capability backs {cursor.robot}/{node.name}."
            self._leave(cursor, node)
            from ..bt import Status
            return Status.FAILURE
        from ..bt import Status
        if cursor.action is None:
            missing = self._missing_preconditions(cursor.robot, capability, node.parameters)
            if missing:
                if self._elapsed(cursor, node) > capability.timeout_ticks:
                    cursor.last_failure = f"Measured preconditions remained false: {', '.join(missing)}."
                    self._leave(cursor, node)
                    return Status.FAILURE
                return Status.RUNNING
            cursor.action = cast(Any, InspectionAction(self, cursor.robot, node.name or "", node.parameters))
            cursor.action_node_id = self._node_key(node)
            self._event(cursor.robot, "action_started", node.label(), node_id=node.node_id)
        result, detail = cursor.action.step(dt)
        if result == "SUCCESS":
            self._record_measured_action_result(cursor.robot, node.name or "", node.parameters, cursor.action)
            self._event(cursor.robot, "action_success", node.label(), detail=detail, node_id=node.node_id)
            cursor.action = None
            cursor.action_node_id = None
            self._leave(cursor, node)
            return Status.SUCCESS
        if result == "FAILURE" or self._elapsed(cursor, node) > capability.timeout_ticks:
            cursor.last_failure = f"Physical action {cursor.robot}/{node.label()} failed: {detail}"
            cursor.action = None
            cursor.action_node_id = None
            self._leave(cursor, node)
            return Status.FAILURE
        return Status.RUNNING

    def _record_measured_action_result(self, robot, name, parameters, action) -> None:
        capability = self.scenario.capability(robot, name)
        assert capability is not None
        bindings = dict(zip(capability.parameters, parameters))
        adds, deletes = ground_effects(capability.effects, bindings)
        for literal in deletes:
            self.signals.discard(canonical_predicate(literal))
        for literal in adds:
            # Isolation is observed from the physical switch, not asserted.
            if not literal.startswith(("asset_isolated(", "leak_repaired(")):
                self.signals.add(canonical_predicate(literal))

    def make_report(self, reason: str | None = None) -> InspectionExecutionReport:
        goals = {predicate: self.observe_literal(predicate) for predicate in self.scenario.goal_state}
        released = not self.resources
        success = self.complete and not self.failed and all(goals.values()) and released
        if reason is None:
            reason = self.failed_reason or (
                "All five supplied BTs completed against measured MuJoCo state."
                if success
                else "Goals are missing or a resource remains owned."
            )
        return InspectionExecutionReport(
            success=success,
            reason=reason,
            simulated_seconds=round(float(self.inspection_world.data.time), 4),
            final_goals=goals,
            robot_status={robot: cursor.status for robot, cursor in self.cursors.items()},
            action_events=self.events,
            locomotion=self.motion.metrics(),
            physics={
                "engine": "MuJoCo",
                "timestep_seconds": float(self.inspection_world.model.opt.timestep),
                "b2_position_m": self.inspection_world.data.xpos[
                    self.inspection_world.model.body("b2_carriage").id
                ].round(5).tolist(),
                "husky_position_m": self.inspection_world.data.xpos[
                    self.inspection_world.model.body("husky_base").id
                ].round(5).tolist(),
                "thermal_evidence": self.inspection_world.evidence,
                "localized_site": self.inspection_world.hidden_anomaly_site,
                "isolation_switch_measured": self.inspection_world.isolated(),
                "leak_repair_collar_measured": self.inspection_world.leak_repaired(),
                "fallen_tool_active": self.inspection_world.fallen_tool_active,
                "fallen_tool_position_m": self.inspection_world.fallen_tool_position().round(5).tolist(),
                "resources_released": released,
                "scope": "Task-level motion controllers; no claim of sim-to-real B2 gait or Husky navigation control.",
            },
        )


class InspectionAction:
    def __init__(self, executor: InspectionExecutor, robot: str, name: str, parameters: tuple[str, ...]) -> None:
        self.executor = executor
        self.world = executor.inspection_world
        self.robot = robot
        self.name = name
        self.parameters = parameters
        self.stage = "start"
        self.stage_started = float(self.world.data.time)

    def step(self, _dt: float) -> tuple[str, str]:
        navigation = {
            "navigate_b2_solar_view": ("b2_base", "solar_view"),
            "navigate_b2_pipe_view": ("b2_base", "pipe_view"),
            "return_b2_home": ("b2_base", "b2_home"),
            "navigate_husky_reference_dock": ("husky_base", "reference_dock"),
            "navigate_husky_anomaly_dock": ("husky_base", "anomaly_service_dock"),
            "return_husky_home": ("husky_base", "husky_home"),
            "navigate_b2_pipe_inspection": ("b2_base", "pipe_view"),
            "return_b2_home_after_repair": ("b2_base", "b2_home"),
            "navigate_husky_leak_repair_dock": ("husky_base", "leak_repair_dock"),
            "return_husky_home_after_repair": ("husky_base", "husky_home"),
            "navigate_b2_tool_search": ("b2_base", "tool_search_view"),
            "navigate_b2_search_to_solar": ("b2_base", "solar_view"),
            "navigate_husky_tool_recovery": ("husky_base", "tool_recovery_dock"),
            "navigate_husky_recovery_to_reference": ("husky_base", "reference_dock"),
        }
        if self.name in navigation:
            robot, dock = navigation[self.name]
            self.executor.motion.navigate(robot, dock)
            return ("SUCCESS", f"measured at {dock}") if self.executor.motion.reached(robot, dock) else ("RUNNING", "servo navigation")
        if self.name.startswith("deploy_camera_"):
            reached = self.world.command_arm("z1_thermal_arm", Z1_DEPLOY)
            return ("SUCCESS", "thermal camera reached deployed joint pose") if reached else ("RUNNING", "deploying camera")
        if self.name.startswith("stow_z1_"):
            reached = self.world.command_arm("z1_thermal_arm", Z1_HOME)
            return ("SUCCESS", "Z1 reached measured stow pose") if reached else ("RUNNING", "stowing Z1")
        if self.name == "localize_fallen_tool":
            self.world.command_arm("z1_thermal_arm", Z1_DEPLOY)
            if self._elapsed() < 0.8:
                return "RUNNING", "searching the shared corridor for the dropped tool"
            measurement = self.world.localize_fallen_tool()
            if not cast(bool, measurement["settled"]):
                return "FAILURE", "the dropped tool had not reached a stable floor pose"
            if cast(float, measurement["range_m"]) > 4.5:
                return "FAILURE", "fallen tool outside validated search range"
            return "SUCCESS", f"fallen tool localized at {measurement['tool_position_m']}"
        if self.name in {
            "scan_solar_panel",
            "scan_pipe_rig",
            "verify_isolation_thermal",
            "detect_pipe_leak",
            "verify_pipe_repair",
        }:
            self.world.command_arm("z1_thermal_arm", Z1_DEPLOY)
            if self._elapsed() < 0.8:
                return "RUNNING", "integrating thermal frames"
            phase = {
                "scan_solar_panel": "solar",
                "verify_isolation_thermal": "verification",
                "detect_pipe_leak": "leak",
                "verify_pipe_repair": "repair_verification",
            }.get(self.name, "pipe")
            measurement = self.world.thermal_measurement(phase)
            if cast(float, measurement["range_m"]) > 4.5:
                return "FAILURE", "thermal target outside validated sensing range"
            if phase in {"verification", "repair_verification"} and cast(float, measurement["peak_c"]) > 35.0:
                return "FAILURE", "post-intervention temperature remained unsafe"
            if phase == "leak" and cast(float, measurement["peak_c"]) < 60.0:
                return "FAILURE", "seeded leak did not produce the expected thermal evidence"
            return "SUCCESS", f"thermal evidence archived ({measurement['peak_c']} C peak)"

        if self.name == "prepare_inspection_kit":
            return self._arm_to("static_franka", PANDA_WORK, "inspection kit grasp pose")
        if self.name == "prepare_repair_tool":
            return self._arm_to("static_franka", PANDA_WORK, "repair tool grasp pose")
        if self.name == "place_inspection_kit_handoff":
            self.world.set_kit_pose("handoff")
            return self._arm_to("static_franka", PANDA_HOME, "kit placed in handoff tray")
        if self.name == "place_repair_tool_handoff":
            self.world.set_kit_pose("handoff")
            return self._arm_to("static_franka", PANDA_HOME, "repair tool placed in handoff tray")
        if self.name == "load_inspection_kit":
            return self._work_then_home("husky_franka", "husky", "kit secured on Husky")
        if self.name == "load_repair_tool":
            return self._work_then_home("husky_franka", "husky", "repair tool secured on Husky")
        if self.name == "recover_localized_tool":
            if not self.world.fallen_tool_settled() and self.stage == "start":
                return "FAILURE", "localized tool is not in a stable floor pose"
            if self.stage == "start":
                if self.world.command_arm("husky_franka", PANDA_WORK):
                    self.world.attach_fallen_tool_to_husky()
                    self._next("home")
                return "RUNNING", "reaching to the measured fallen-tool pose"
            reached = self.world.command_arm("husky_franka", PANDA_HOME)
            return ("SUCCESS", "fallen tool recovered and secured on Husky") if reached else ("RUNNING", "returning arm home with recovered tool")
        if self.name == "install_thermal_reference":
            return self._work_then_home("husky_franka", "reference", "thermal reference installed")
        if self.name == "confirm_reported_anomaly":
            return self._work_then_home("husky_franka", None, "reported hot joint physically revisited")
        if self.name == "attach_inspection_marker":
            return self._work_then_home("husky_franka", "marker", "marker attached at localized joint")
        if self.name == "isolate_energy_rig":
            if self.stage == "start":
                if self.world.command_arm("husky_franka", PANDA_WORK):
                    self.world.isolate()
                    self._next("switch")
                return "RUNNING", "approaching isolation switch"
            if self.stage == "switch":
                self.world.isolate()
                if self.world.isolated():
                    self._next("home")
                return "RUNNING", "depressing measured isolation switch"
            reached = self.world.command_arm("husky_franka", PANDA_HOME)
            return ("SUCCESS", "isolation switch latched") if reached else ("RUNNING", "returning arm home")
        if self.name == "repair_pipe_leak":
            if self.stage == "start":
                if self.world.command_arm("husky_franka", PANDA_WORK):
                    self.world.repair_leak()
                    self._next("collar")
                return "RUNNING", "bringing the supplied tool to the leaking joint"
            if self.stage == "collar":
                self.world.repair_leak()
                if self.world.leak_repaired():
                    self.world.set_kit_pose("tool_rack")
                    self._next("home")
                return "RUNNING", "closing the measured repair collar"
            reached = self.world.command_arm("husky_franka", PANDA_HOME)
            return ("SUCCESS", "repair collar closed and tool returned to rack") if reached else ("RUNNING", "returning arm home")
        return "FAILURE", f"no inspection action adapter for {self.name}"

    def _arm_to(self, robot: str, target: np.ndarray, detail: str) -> tuple[str, str]:
        reached = self.world.command_arm(robot, target)
        return ("SUCCESS", detail) if reached else ("RUNNING", detail)

    def _work_then_home(self, robot: str, prop: str | None, detail: str) -> tuple[str, str]:
        if self.stage == "start":
            if self.world.command_arm(robot, PANDA_WORK):
                if prop == "marker":
                    self.world.set_marker_visible(True)
                elif prop is not None:
                    self.world.set_kit_pose(prop)
                self._next("home")
            return "RUNNING", detail
        reached = self.world.command_arm(robot, PANDA_HOME)
        return ("SUCCESS", detail) if reached else ("RUNNING", "returning arm home")

    def diagnostic_detail(self) -> str:
        return self.stage

    def _next(self, stage: str) -> None:
        self.stage = stage
        self.stage_started = float(self.world.data.time)

    def _elapsed(self) -> float:
        return float(self.world.data.time) - self.stage_started
