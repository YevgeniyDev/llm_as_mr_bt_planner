"""Stable actuator controllers for the inspection demonstration plant."""

from __future__ import annotations

import numpy as np

from .inspection_world import B2_DOCK_X, B2_HOME, HUSKY_DOCK_X, InspectionWorld


class InspectionMotionController:
    def __init__(self, world: InspectionWorld) -> None:
        self.world = world
        self.targets = {"b2_base": B2_DOCK_X["b2_home"], "husky_base": HUSKY_DOCK_X["husky_home"]}
        self.start_x = {robot: world.robot_x(robot) for robot in self.targets}
        self.max_b2_torque = 0.0
        self.max_speed = {robot: 0.0 for robot in self.targets}
        self.command_steps = {robot: 0 for robot in self.targets}
        self._b2_actuators: list[int] = []
        self._b2_qpos: list[int] = []
        self._b2_dofs: list[int] = []
        for leg in ("FR", "FL", "RR", "RL"):
            for joint in ("hip", "thigh", "calf"):
                name = f"b2_{leg}_{joint}"
                self._b2_actuators.append(world.model.actuator(name).id)
                joint_model = world.model.joint(f"{name}_joint")
                self._b2_qpos.append(int(joint_model.qposadr[0]))
                self._b2_dofs.append(int(joint_model.dofadr[0]))

    def navigate(self, robot: str, dock: str) -> None:
        targets = B2_DOCK_X if robot == "b2_base" else HUSKY_DOCK_X
        self.targets[robot] = targets[dock]

    def reached(self, robot: str, dock: str) -> bool:
        return self.world.at_dock(robot, dock)

    def step(self) -> None:
        for robot, target in self.targets.items():
            self.world.command_base(robot, target)
            speed = self.world.robot_speed(robot)
            self.max_speed[robot] = max(self.max_speed[robot], speed)
            if abs(self.world.robot_x(robot) - target) > 0.055:
                self.command_steps[robot] += 1
        q = np.asarray(self.world.data.qpos[self._b2_qpos])
        qvel = np.asarray(self.world.data.qvel[self._b2_dofs])
        torque = np.clip(125.0 * (B2_HOME - q) - 6.0 * qvel, -180.0, 180.0)
        self.world.data.ctrl[self._b2_actuators] = torque
        self.max_b2_torque = max(self.max_b2_torque, float(np.max(np.abs(torque))))
        self.world.update_kinematic_props()

    def upright(self) -> bool:
        rotation = self.world.data.xmat[self.world.model.body("b2_base_link").id].reshape(3, 3)
        return bool(rotation[2, 2] > 0.98)

    def metrics(self) -> dict[str, object]:
        return {
            "controller": "position-servo task-space base motion; B2 joint-torque stance hold",
            "b2_displacement_m": round(self.world.robot_x("b2_base") - self.start_x["b2_base"], 5),
            "husky_displacement_m": round(self.world.robot_x("husky_base") - self.start_x["husky_base"], 5),
            "max_base_speed_m_s": {key: round(value, 5) for key, value in self.max_speed.items()},
            "base_motion_command_steps": self.command_steps,
            "max_b2_stance_torque_nm": round(self.max_b2_torque, 4),
            "b2_upright": self.upright(),
            "direct_qpos_writes_after_reset": self.world.qpos_writes_after_reset,
        }
