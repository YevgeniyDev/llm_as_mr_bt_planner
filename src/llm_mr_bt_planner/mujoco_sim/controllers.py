"""Model-scoped physical controllers for the bundled three-robot scenes.

The differential-IK implementation follows the damped least-squares and
null-space structure demonstrated by kevinzakka/mjctrl (Apache-2.0), adapted
for prefixed robots inside one model.  The quadruped controller follows the
alternating-contact, stance/swing and torque-control structure of
elijah-waichong-chan/go2-convex-mpc (MIT), but uses MuJoCo state directly so it
can account for the mounted Z1 and carrier without the upstream standalone
Pinocchio state layout.  It is a contact gait controller, not a claim that the
upstream convex MPC was copied unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .world import DESTINATION_DOCK_X, CourierWorld


@dataclass
class ArmController:
    world: CourierWorld
    robot: str
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    site_name: str
    home: np.ndarray
    gripper_actuator: str
    gripper_open_value: float
    gripper_closed_value: float
    body_prefix: str
    gripper_joint: str | None = None
    gripper_open_position: float | None = None
    gripper_closed_position: float | None = None
    gripper_position_tolerance: float = 0.08
    gripper_command_step: float | None = None
    max_joint_velocity: float = 1.1
    damping: float = 2e-3
    feedback_gain: float = 0.012
    position_tolerance: float = 0.025
    _joint_ids: np.ndarray = field(init=False)
    _qpos_ids: np.ndarray = field(init=False)
    _dof_ids: np.ndarray = field(init=False)
    _actuator_ids: np.ndarray = field(init=False)
    _site_id: int = field(init=False)
    _gripper_id: int = field(init=False)
    _gripper_qpos_id: int | None = field(init=False, default=None)
    _desired_q: np.ndarray = field(init=False)
    _gripper_value: float = field(init=False)
    _applied_gripper_value: float = field(init=False)
    _ik_target_position: np.ndarray | None = field(init=False, default=None)
    _ik_target_rotation: np.ndarray | None = field(init=False, default=None)
    _ik_target_q: np.ndarray | None = field(init=False, default=None)
    _ik_base_position: np.ndarray | None = field(init=False, default=None)
    _ik_base_rotation: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        model = self.world.model
        self._joint_ids = np.array([model.joint(name).id for name in self.joint_names], dtype=int)
        self._qpos_ids = np.array([int(model.joint(name).qposadr[0]) for name in self.joint_names], dtype=int)
        self._dof_ids = np.array([int(model.joint(name).dofadr[0]) for name in self.joint_names], dtype=int)
        self._actuator_ids = np.array([model.actuator(name).id for name in self.actuator_names], dtype=int)
        self._site_id = model.site(self.site_name).id
        self._gripper_id = model.actuator(self.gripper_actuator).id
        if self.gripper_joint is not None:
            self._gripper_qpos_id = int(model.joint(self.gripper_joint).qposadr[0])
        self._desired_q = self.q.copy()
        self._gripper_value = self.gripper_open_value
        self._applied_gripper_value = float(self.world.data.ctrl[self._gripper_id])

    @property
    def q(self) -> np.ndarray:
        return np.asarray(self.world.data.qpos[self._qpos_ids]).copy()

    @property
    def tool_position(self) -> np.ndarray:
        return self.world.data.site_xpos[self._site_id].copy()

    def set_gripper(self, *, closed: bool) -> None:
        self._gripper_value = self.gripper_closed_value if closed else self.gripper_open_value

    def gripper_closed(self) -> bool:
        return self._gripper_at(self.gripper_closed_value, self.gripper_closed_position)

    def gripper_opened(self) -> bool:
        return self._gripper_at(self.gripper_open_value, self.gripper_open_position)

    @property
    def gripper_position(self) -> float | None:
        if self._gripper_qpos_id is None:
            return None
        return float(self.world.data.qpos[self._gripper_qpos_id])

    def _gripper_at(self, expected_command: float, expected_position: float | None) -> bool:
        actual_command = float(self.world.data.ctrl[self._gripper_id])
        scale = max(1.0, abs(self.gripper_open_value - self.gripper_closed_value))
        command_reached = abs(actual_command - expected_command) <= 0.08 * scale
        if self._gripper_qpos_id is None or expected_position is None:
            return command_reached
        actual_position = float(self.world.data.qpos[self._gripper_qpos_id])
        return command_reached and abs(actual_position - expected_position) <= self.gripper_position_tolerance

    def hold(self) -> None:
        self._apply(self._desired_q)

    def move_home(self, dt: float) -> bool:
        self._ik_target_position = None
        self._ik_target_rotation = None
        self._ik_target_q = None
        self._ik_base_position = None
        self._ik_base_rotation = None
        current = self.q
        delta = np.clip(
            self.home - self._desired_q,
            -self.max_joint_velocity * dt,
            self.max_joint_velocity * dt,
        )
        self._desired_q = self._desired_q + delta
        self._apply(self._desired_q)
        return bool(np.max(np.abs(self.home - current)) < 0.045)

    def move_tool(
        self,
        target: np.ndarray,
        dt: float,
        *,
        target_rotation: np.ndarray | None = None,
        track_mobile_base: bool = True,
    ) -> bool:
        target = np.asarray(target, dtype=float)
        target_rotation = None if target_rotation is None else np.asarray(target_rotation, dtype=float)
        rotation_changed = (self._ik_target_rotation is None) != (target_rotation is None) or (
            self._ik_target_rotation is not None
            and target_rotation is not None
            and np.linalg.norm(target_rotation - self._ik_target_rotation) > 1e-4
        )
        base_position = self.world.base_position
        base_rotation = self.world.data.xmat[self.world.base_body_id].reshape(3, 3).copy()
        mobile_base_changed = track_mobile_base and self.robot == "unitree_go2_z1" and (
            self._ik_base_position is None
            or self._ik_base_rotation is None
            or np.linalg.norm(base_position - self._ik_base_position) > 0.003
            or np.linalg.norm(base_rotation - self._ik_base_rotation) > 0.015
        )
        if (
            self._ik_target_position is None
            or np.linalg.norm(target - self._ik_target_position) > 0.004
            or rotation_changed
            or mobile_base_changed
        ):
            self._ik_target_q = self._solve_ik(target, target_rotation)
            self._ik_target_position = target.copy()
            self._ik_target_rotation = None if target_rotation is None else target_rotation.copy()
            self._ik_base_position = base_position
            self._ik_base_rotation = base_rotation
        assert self._ik_target_q is not None
        error = target - self.tool_position
        # Low-gain task-space feedback compensates finite actuator stiffness and
        # payload/contact loads that the kinematic scratch solve cannot predict.
        jacobian_position = np.zeros((3, self.world.model.nv))
        jacobian_rotation = np.zeros((3, self.world.model.nv)) if target_rotation is not None else None
        mujoco.mj_jacSite(
            self.world.model,
            self.world.data,
            jacobian_position,
            jacobian_rotation,
            self._site_id,
        )
        if jacobian_rotation is None:
            jacobian = jacobian_position[:, self._dof_ids]
            task_error = error
        else:
            rotation_error = self._rotation_error(target_rotation)
            assert rotation_error is not None
            jacobian = np.vstack(
                (jacobian_position[:, self._dof_ids], jacobian_rotation[:, self._dof_ids])
            )
            task_error = np.concatenate((error, rotation_error))
        correction = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + self.damping * np.eye(len(task_error)),
            task_error,
        )
        correction_peak = float(np.max(np.abs(correction)))
        if correction_peak > 0.12:
            correction *= 0.12 / correction_peak
        self._ik_target_q += self.feedback_gain * correction
        ranges = self.world.model.jnt_range[self._joint_ids]
        self._ik_target_q = np.clip(
            self._ik_target_q,
            ranges[:, 0] + 1e-4,
            ranges[:, 1] - 1e-4,
        )
        delta = np.clip(
            self._ik_target_q - self._desired_q,
            -self.max_joint_velocity * dt,
            self.max_joint_velocity * dt,
        )
        self._desired_q = self._desired_q + delta
        self._apply(self._desired_q)
        return self.tool_pose_reached(target, target_rotation=target_rotation)

    def tool_pose_reached(
        self,
        target: np.ndarray,
        *,
        target_rotation: np.ndarray | None = None,
        position_tolerance: float | None = None,
        rotation_tolerance: float = 0.12,
    ) -> bool:
        position_limit = self.position_tolerance if position_tolerance is None else position_tolerance
        rotation_error = self._rotation_error(target_rotation)
        return bool(
            np.linalg.norm(np.asarray(target, dtype=float) - self.tool_position) < position_limit
            and (rotation_error is None or np.linalg.norm(rotation_error) < rotation_tolerance)
        )

    def _solve_ik(self, target: np.ndarray, target_rotation: np.ndarray | None) -> np.ndarray:
        """Damped differential IK on a scratch state; execution remains actuator driven."""
        model = self.world.model
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = self.world.data.qpos
        scratch.qvel[:] = 0
        if target_rotation is not None:
            return self._solve_pose_ik(scratch, target, target_rotation)

        ranges = model.jnt_range[self._joint_ids]

        def residual(joints: np.ndarray) -> np.ndarray:
            scratch.qpos[self._qpos_ids] = joints
            mujoco.mj_forward(model, scratch)
            return (target - scratch.site_xpos[self._site_id]) * 5.0

        starts = (
            (
                self.q,
                np.array([-1.65, 2.90, -0.75, -0.20, -0.30, 0.30]),
                np.array([-1.90, 2.55, -0.60, -0.20, -0.20, -0.90]),
                np.array([-1.30, 2.40, -0.50, -0.40, -0.40, 1.20]),
            )
            if len(self._joint_ids) == 6
            else (self.q, self.home)
        )
        bounds = (ranges[:, 0] + 1e-4, ranges[:, 1] - 1e-4)
        results = [
            least_squares(
                residual,
                np.clip(start, *bounds),
                bounds=bounds,
                max_nfev=100,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            for start in starts
        ]
        evaluated = [
            (candidate, float(np.linalg.norm(residual(candidate.x))))
            for candidate in results
        ]
        best_task_error = min(task_error for _, task_error in evaluated)
        acceptable_task_error = max(0.02, best_task_error + 0.002)
        continuous_candidates = [
            candidate
            for candidate, task_error in evaluated
            if task_error <= acceptable_task_error
        ]
        joint_span = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)
        result = min(
            continuous_candidates,
            key=lambda candidate: float(
                np.linalg.norm((candidate.x - self.q) / joint_span)
            ),
        )
        return np.asarray(result.x).copy()

    def _solve_pose_ik(
        self, scratch: mujoco.MjData, target: np.ndarray, target_rotation: np.ndarray
    ) -> np.ndarray:
        model = self.world.model
        target_quat = np.empty(4)
        mujoco.mju_mat2Quat(target_quat, target_rotation.reshape(-1))
        ranges = model.jnt_range[self._joint_ids]

        def residual(joints: np.ndarray) -> np.ndarray:
            scratch.qpos[self._qpos_ids] = joints
            mujoco.mj_forward(model, scratch)
            current_quat = np.empty(4)
            rotation_error = np.empty(3)
            mujoco.mju_mat2Quat(current_quat, scratch.site_xmat[self._site_id])
            mujoco.mju_subQuat(rotation_error, target_quat, current_quat)
            return np.concatenate(((target - scratch.site_xpos[self._site_id]) * 5.0, rotation_error))

        bounds = (ranges[:, 0] + 1e-4, ranges[:, 1] - 1e-4)
        # Pose IK has elbow-up/down local minima.  Use dimension-appropriate
        # seeds for the six-axis Z1 and redundant seven-axis Pandas.
        if len(self._joint_ids) == 6:
            starts = (
                self.q,
                np.array([-1.008, 1.639, -1.514, 0.743, -0.387, -1.152]),
                np.array([-1.933, 1.457, -1.327, 0.689, 0.253, -1.832]),
                np.array([-0.900, 1.750, -1.650, 0.850, -0.550, -1.000]),
                np.array([-1.300, 1.650, -1.450, 0.650, 0.0, -1.500]),
            )
        else:
            starts = (
                self.q,
                self.home,
                np.array([0.0, -0.55, 0.0, -2.05, 0.0, 1.55, -0.785]),
                np.array([0.45, 0.35, 0.0, -1.85, 0.0, 1.35, -0.35]),
                np.array([-0.45, 0.35, 0.0, -1.85, 0.0, 1.35, -1.15]),
            )
        results = [
            least_squares(
                residual,
                np.clip(start, *bounds),
                bounds=bounds,
                max_nfev=120,
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
            )
            for start in starts
        ]
        evaluated = [
            (candidate, float(np.linalg.norm(residual(candidate.x)))) for candidate in results
        ]
        best_task_error = min(task_error for _, task_error in evaluated)
        # Several elbow-up/down solutions can have effectively identical tool
        # pose error.  Selecting only by the last few residual digits can jump
        # to a distant branch whenever the floating base moves, making the tool
        # sweep through the workspace despite a smooth Cartesian command.
        # Retain all solutions that are practically as accurate as the best
        # one, then choose the smallest normalized displacement from the
        # measured joint state.
        acceptable_task_error = max(0.02, best_task_error + 0.002)
        continuous_candidates = [
            candidate
            for candidate, task_error in evaluated
            if task_error <= acceptable_task_error
        ]
        joint_span = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)
        result = min(
            continuous_candidates,
            key=lambda candidate: float(np.linalg.norm((candidate.x - self.q) / joint_span)),
        )
        final_residual = residual(result.x)
        if not np.isfinite(final_residual).all():
            raise RuntimeError(
                f"{self.robot} pose IK produced a non-finite solution and physical execution was stopped."
            )
        return np.asarray(result.x).copy()

    def _rotation_error(self, target_rotation: np.ndarray | None) -> np.ndarray | None:
        if target_rotation is None:
            return None
        target_quat = np.empty(4)
        current_quat = np.empty(4)
        error = np.empty(3)
        mujoco.mju_mat2Quat(target_quat, target_rotation.reshape(-1))
        mujoco.mju_mat2Quat(current_quat, self.world.data.site_xmat[self._site_id])
        mujoco.mju_subQuat(error, target_quat, current_quat)
        return error

    def _apply(self, desired: np.ndarray) -> None:
        self.world.data.ctrl[self._actuator_ids] = desired
        if self.gripper_command_step is None:
            self._applied_gripper_value = self._gripper_value
        else:
            self._applied_gripper_value += float(
                np.clip(
                    self._gripper_value - self._applied_gripper_value,
                    -self.gripper_command_step,
                    self.gripper_command_step,
                )
            )
        self.world.data.ctrl[self._gripper_id] = self._applied_gripper_value


class ContactGaitController:
    """Alternating diagonal trot driven only through the 12 Go2 leg motors."""

    def __init__(self, world: CourierWorld) -> None:
        self.world = world
        self.legs = ("FL", "FR", "RL", "RR")
        self.phase_offsets = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}
        self.home = {
            leg: np.array([0.10 if leg in {"FL", "RL"} else -0.10, 0.9, -1.8])
            for leg in self.legs
        }
        self.actuator_ids: dict[str, np.ndarray] = {}
        self.qpos_ids: dict[str, np.ndarray] = {}
        self.dof_ids: dict[str, np.ndarray] = {}
        model = world.model
        for leg in self.legs:
            joints = [model.joint(f"go2_{leg}_{name}_joint") for name in ("hip", "thigh", "calf")]
            self.actuator_ids[leg] = np.array(
                [model.actuator(f"go2_{leg}_{name}").id for name in ("hip", "thigh", "calf")], dtype=int
            )
            self.qpos_ids[leg] = np.array([int(joint.qposadr[0]) for joint in joints], dtype=int)
            self.dof_ids[leg] = np.array([int(joint.dofadr[0]) for joint in joints], dtype=int)
        self.target_x: float | None = None
        self.started_at = 0.0
        self.contact_steps = 0
        self.multi_diagonal_contact_steps = 0
        self.max_abs_torque = 0.0
        self.max_tilt_radians = 0.0
        self.start_x = float(world.base_position[0])

    @property
    def navigating(self) -> bool:
        return self.target_x is not None

    def navigate_to_destination(self) -> None:
        self.navigate_to(DESTINATION_DOCK_X)

    def navigate_to(self, target_x: float) -> None:
        if self.target_x is None or abs(self.target_x - target_x) > 1e-6:
            self._set_foot_friction(0.8)
            self.target_x = float(target_x)
            self.started_at = float(self.world.data.time)

    def reached_destination(self) -> bool:
        return self.reached_target()

    def reached_target(self) -> bool:
        if self.target_x is None:
            return False
        error = abs(float(self.world.base_position[0]) - self.target_x)
        speed = float(np.linalg.norm(self.world.base_velocity[:2]))
        reached = error < 0.16 and speed < 0.35 and self.upright()
        if reached:
            self._set_foot_friction(1.6)
        return reached

    def upright(self) -> bool:
        rotation = self.world.data.xmat[self.world.base_body_id].reshape(3, 3)
        return bool(self.world.base_position[2] > 0.20 and rotation[2, 2] > 0.78)

    def step(self) -> None:
        time_now = float(self.world.data.time)
        gait = False
        direction = 1.0
        if self.target_x is not None:
            error = self.target_x - float(self.world.base_position[0])
            direction = 1.0 if error >= 0 else -1.0
            # Begin braking before the dock; body motion still results from contacts and inertia.
            gait = abs(error) > 0.10

        for leg in self.legs:
            desired = self.home[leg].copy()
            if gait:
                desired[0] = 0.0
                phase = 2.0 * math.pi * 1.5 * (time_now - self.started_at) + self.phase_offsets[leg]
                sinusoid = math.sin(phase)
                desired[1] += direction * 0.25 * sinusoid
                desired[2] += direction * 0.35 * sinusoid
            q = np.asarray(self.world.data.qpos[self.qpos_ids[leg]])
            velocity = np.asarray(self.world.data.qvel[self.dof_ids[leg]])
            kp = 60.0 if gait else 105.0
            kd = 2.5 if gait else 4.5
            torque = kp * (desired - q) - kd * velocity
            torque = np.clip(torque, [-23.7, -23.7, -45.43], [23.7, 23.7, 45.43])
            self.world.data.ctrl[self.actuator_ids[leg]] = torque
            self.max_abs_torque = max(self.max_abs_torque, float(np.max(np.abs(torque))))

        contacts = self._grounded_legs()
        if contacts:
            self.contact_steps += 1
        diagonal_a = {"FL", "RR"}
        diagonal_b = {"FR", "RL"}
        if contacts & diagonal_a and contacts & diagonal_b:
            self.multi_diagonal_contact_steps += 1
        rotation = self.world.data.xmat[self.world.base_body_id].reshape(3, 3)
        tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
        self.max_tilt_radians = max(self.max_tilt_radians, tilt)

    def metrics(self) -> dict[str, float | int | bool]:
        displacement = float(self.world.base_position[0]) - self.start_x
        return {
            "contact_driven_displacement_m": displacement,
            "ground_contact_steps": self.contact_steps,
            "multi_diagonal_contact_steps": self.multi_diagonal_contact_steps,
            "max_commanded_leg_torque_nm": self.max_abs_torque,
            "max_base_tilt_deg": math.degrees(self.max_tilt_radians),
            "base_upright": self.upright(),
            "direct_base_state_writes": self.world.qpos_writes_after_reset,
        }

    def _grounded_legs(self) -> set[str]:
        grounded: set[str] = set()
        model, data = self.world.model, self.world.data
        for index in range(data.ncon):
            contact = data.contact[index]
            for geom_id in (contact.geom1, contact.geom2):
                body_name = model.body(int(model.geom_bodyid[geom_id])).name
                for leg in self.legs:
                    if body_name.startswith(f"go2_{leg}_calf"):
                        grounded.add(leg)
        return grounded

    def _set_foot_friction(self, sliding: float) -> None:
        for name in ("go2_FL", "go2_FR", "go2_RL", "go2_RR"):
            self.world.model.geom_friction[self.world.model.geom(name).id] = [sliding, 0.02, 0.01]


def build_arm_controllers(world: CourierWorld) -> dict[str, ArmController]:
    panda_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
    controllers: dict[str, ArmController] = {}
    for robot in ("franka_a", "franka_b"):
        prefix = f"{robot}_"
        robot_home = panda_home.copy()
        if world.task_id == "three_robot_packaging_delivery":
            robot_home[0] = 0.8 if robot == "franka_a" else -0.8
        controllers[robot] = ArmController(
            world=world,
            robot=robot,
            joint_names=tuple(f"{prefix}joint{i}" for i in range(1, 8)),
            actuator_names=tuple(f"{prefix}actuator{i}" for i in range(1, 8)),
            site_name=f"{prefix}grasp_site",
            home=robot_home,
            gripper_actuator=f"{prefix}actuator8",
            gripper_open_value=255.0,
            gripper_closed_value=0.0,
            body_prefix=prefix,
            feedback_gain=0.0,
        )
    controllers["unitree_go2_z1"] = ArmController(
        world=world,
        robot="unitree_go2_z1",
        joint_names=tuple(f"go2_z1_joint{i}" for i in range(1, 7)),
        actuator_names=tuple(f"go2_z1_motor{i}" for i in range(1, 7)),
        site_name="go2_z1_grasp_site",
        home=np.array([0.0, 0.785, -0.261, -0.523, 0.0, 0.0]),
        gripper_actuator="go2_z1_motorGripper",
        # This hinge is closed at 0 and opens by rotating negatively.  At
        # -0.50 rad the pad inner faces are about 41.5 mm apart, matching the
        # 40 mm payload without forced penetration.
        gripper_open_value=-0.75,
        gripper_closed_value=-0.50,
        body_prefix="go2_z1_",
        gripper_joint="go2_z1_jointGripper",
        gripper_open_position=-0.75,
        gripper_closed_position=-0.50,
        gripper_position_tolerance=0.06,
        gripper_command_step=0.0008,
        max_joint_velocity=0.65,
        damping=4e-3,
        feedback_gain=0.0,
        position_tolerance=0.015,
    )
    return controllers
