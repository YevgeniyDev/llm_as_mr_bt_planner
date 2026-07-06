"""Manipulation controller for the MuJoCo backend ("ik" fidelity).

Inverse kinematics is solved with `mink <https://github.com/kevinzakka/mink>`_, a
differential-IK library that formulates each step as a quadratic program. This
buys us, for free and from a well-tested implementation, the things that are hard
to get right by hand:

* an **orientation task** that keeps each gripper pointed straight down,
* a **posture task** that regularises the arm to a natural configuration (no
  twisted, flailing joints),
* **configuration limits** so joints never exceed their range, and
* **collision avoidance** between the arms, so the manipulators no longer drive
  through one another when they reach a shared workspace.

mink produces a smooth, collision-free joint trajectory; we apply it kinematically
(the whole team stays deterministic and stable -- nothing flings, tips, or falls
through the table) and carry a grasped part rigidly with the gripper. The Panda
fingers and the Z1 jaw close on the part when grasped and open on release.
"""

from __future__ import annotations

from collections.abc import Callable

import mink
import mujoco
import numpy as np

from .mujoco_scene import SceneInfo

_PICK_WORDS = ("pick", "grasp", "grab", "hold")
_PLACE_WORDS = ("place", "put", "return", "mount", "insert", "fasten",
                "attach", "install", "deliver", "stabilize")

_APPROACH = 0.14           # clearance above a part before descending
_CARRY_Z = 0.16            # how high a carried part rides above the table
_MAX_IK_STEPS = 90         # mink iterations per motion segment
_DT = 0.02                 # integration timestep for the diff-IK solver
_POS_TOL = 3e-3
_ORI_TOL = 2e-2


class ArmController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo,
                 step_cb: Callable[[], None] | None = None) -> None:
        self.model = model
        self.data = data
        self.info = info
        self.step_cb = step_cb
        self.configuration = mink.Configuration(model)

        self.home_R: dict[str, np.ndarray] = {}     # clean "hand-down" orientation
        self.offset: dict[str, float] = {}          # EE origin -> grasp point
        self.arm_qadr: dict[str, list[int]] = {}     # arm-joint qpos addresses
        self.gripper: dict[str, list[tuple[int, float, float]]] = {}  # (qadr, open, closed)
        self.closed: dict[str, bool] = {}

        # Read the clean ready pose (arms folded, hands down) from the home config.
        saved = data.qpos.copy()
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        arm_geoms: dict[str, list[int]] = {}
        for robot, arm in info.arms.items():
            ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)
            self.home_R[robot] = np.array(data.xmat[ee]).reshape(3, 3).copy()
            self.offset[robot] = 0.10 if len(arm.joint_names) == 7 else 0.12
            self.arm_qadr[robot] = [
                int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
                for jn in arm.joint_names
            ]
            self.gripper[robot] = self._find_gripper(robot)
            self.closed[robot] = False
            arm_geoms[robot] = mink.get_subtree_geom_ids(model, self._base_body(robot))
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)

        # One frame task per arm (position + orientation), plus a posture regulariser.
        self.frames = {
            robot: mink.FrameTask(frame_name=info.arms[robot].ee_body, frame_type="body",
                                  position_cost=1.0, orientation_cost=1.0, lm_damping=1.0)
            for robot in info.arms
        }
        self.posture = mink.PostureTask(model, cost=1e-2)

        # Avoid collisions between every pair of arms.
        robots = list(info.arms)
        pairs = [
            (arm_geoms[robots[i]], arm_geoms[robots[j]])
            for i in range(len(robots)) for j in range(i + 1, len(robots))
        ]
        self.limits: list = [mink.ConfigurationLimit(model=model)]
        if pairs:
            self.limits.append(mink.CollisionAvoidanceLimit(
                model=model, geom_pairs=pairs,
                minimum_distance_from_collisions=0.05, collision_detection_distance=0.20,
            ))

    # -- setup helpers ------------------------------------------------------

    def _base_body(self, robot: str) -> int:
        prefix = self.info.ee_prefix.get(robot, "")
        ids = [b for b in range(self.model.nbody)
               if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)]
        return min(ids) if ids else 0

    def _find_gripper(self, robot: str) -> list[tuple[int, float, float]]:
        prefix = self.info.ee_prefix.get(robot, "")
        jaws: list[tuple[int, float, float]] = []
        for j in range(self.model.njnt):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if not nm.startswith(prefix) or not ("finger" in nm or "gripper" in nm.lower()):
                continue
            if self.model.jnt_limited[j]:
                lo, hi = float(self.model.jnt_range[j][0]), float(self.model.jnt_range[j][1])
            else:
                lo, hi = 0.0, 0.04
            jaws.append((int(self.model.jnt_qposadr[j]), hi, lo + 0.22 * (hi - lo)))
        return jaws

    # -- kinematics helpers -------------------------------------------------

    def _body_pos(self, body: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        return np.array(self.data.xpos[bid]) if bid >= 0 else np.zeros(3)

    def ee_pos(self, robot: str) -> np.ndarray:
        ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.info.arms[robot].ee_body)
        return np.array(self.data.xpos[ee])

    def _ee_R(self, robot: str) -> np.ndarray:
        ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.info.arms[robot].ee_body)
        return np.array(self.data.xmat[ee]).reshape(3, 3)

    def _snap_held(self) -> None:
        for obj, robot in self.info.held.items():
            if robot not in self.info.arms:
                continue
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
            if bid < 0:
                continue
            jadr = self.model.body_jntadr[bid]
            if jadr < 0:
                continue
            qadr = int(self.model.jnt_qposadr[jadr])
            ee = self.ee_pos(robot)
            off = self.offset[robot]
            self.data.qpos[qadr:qadr + 3] = [float(ee[0]), float(ee[1]), float(ee[2] - off)]
            self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
            dadr = int(self.model.jnt_dofadr[jadr])
            self.data.qvel[dadr:dadr + 6] = 0.0

    def _refresh(self) -> None:
        self._snap_held()
        for robot, jaws in self.gripper.items():
            for qadr, open_v, closed_v in jaws:
                self.data.qpos[qadr] = closed_v if self.closed[robot] else open_v
        mujoco.mj_forward(self.model, self.data)
        if self.step_cb is not None:
            self.step_cb()

    def _set_targets(self, active: str, target_pos: np.ndarray) -> None:
        for robot in self.info.arms:
            if robot == active:
                rot, pos = self.home_R[robot], target_pos
            else:  # hold idle arms exactly where they are
                rot, pos = self._ee_R(robot), self.ee_pos(robot)
            self.frames[robot].set_target(
                mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rot), pos))

    # -- motion -------------------------------------------------------------

    def move(self, robot: str, target) -> None:
        """Follow mink's collision-free trajectory to ``target`` with the gripper
        pointed down, applying each configuration kinematically and carrying any
        held part with the hand."""
        self.configuration.update(self.data.qpos)
        self.posture.set_target_from_configuration(self.configuration)
        self._set_targets(robot, np.asarray(target, float))
        tasks = [*self.frames.values(), self.posture]
        for _ in range(_MAX_IK_STEPS):
            vel = mink.solve_ik(self.configuration, tasks, _DT, "daqp", limits=self.limits, damping=1e-3)
            self.configuration.integrate_inplace(vel, _DT)
            # Apply *every* arm's joints from the solution: the collision-avoidance
            # QP may nudge an idle arm aside, and that must reach the sim too.
            for qadrs in self.arm_qadr.values():
                for qa in qadrs:
                    self.data.qpos[qa] = self.configuration.q[qa]
            self._refresh()
            err = self.frames[robot].compute_error(self.configuration)
            if np.linalg.norm(err[:3]) < _POS_TOL and np.linalg.norm(err[3:]) < _ORI_TOL:
                break

    # -- grasp / release ----------------------------------------------------

    def grasp(self, robot: str, obj: str) -> None:
        self.info.held[obj] = robot
        self.closed[robot] = True
        for _ in range(6):
            self._refresh()

    def release(self, robot: str, obj: str, xy: tuple[float, float]) -> None:
        self.info.held.pop(obj, None)
        self.closed[robot] = False
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, obj)
        if bid >= 0:
            jadr = self.model.body_jntadr[bid]
            if jadr >= 0:
                qadr = int(self.model.jnt_qposadr[jadr])
                rest = self.info.surface_z + self.info.object_half_z.get(obj, 0.03)
                self.data.qpos[qadr:qadr + 3] = [float(xy[0]), float(xy[1]), float(rest)]
                self.data.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                dadr = int(self.model.jnt_dofadr[jadr])
                self.data.qvel[dadr:dadr + 6] = 0.0
        for _ in range(4):
            self._refresh()

    # -- action dispatch ----------------------------------------------------

    def _object(self, event: dict) -> str | None:
        for p in event.get("parameters", []):
            if p in self.info.object_bodies:
                return p
        return None

    def _target_xy(self, event: dict) -> tuple[float, float]:
        for p in event.get("parameters", []):
            if p in self.info.location_xy:
                return self.info.location_xy[p]
        loc = self.info.robot_location.get(event.get("robot", ""))
        return self.info.location_xy.get(loc, (0.0, 0.0)) if loc else (0.0, 0.0)

    def execute(self, event: dict) -> str:
        name = (event.get("name") or "").lower()
        robot = event.get("robot", "")
        if robot not in self.info.arms:
            return "idle"
        if "drawer" in name:
            return self._drawer(robot, "open" in name)
        obj = self._object(event)
        if obj is not None and any(w in name for w in _PICK_WORDS) and obj not in self.info.held:
            return self._pick(robot, obj)
        if obj is not None and any(w in name for w in _PLACE_WORDS):
            return self._place(robot, obj, self._target_xy(event))
        return "settle"

    def _pick(self, robot: str, obj: str) -> str:
        off = self.offset[robot]
        self.closed[robot] = False
        p = self._body_pos(obj)
        self.move(robot, [p[0], p[1], p[2] + off + _APPROACH])
        p = self._body_pos(obj)
        self.move(robot, [p[0], p[1], p[2] + off])
        self.grasp(robot, obj)
        self.move(robot, [p[0], p[1], self.info.surface_z + off + _CARRY_Z])
        return "ik_pick"

    def _place(self, robot: str, obj: str, xy: tuple[float, float]) -> str:
        off = self.offset[robot]
        tx, ty = xy
        half = self.info.object_half_z.get(obj, 0.03)
        self.move(robot, [tx, ty, self.info.surface_z + off + _CARRY_Z])
        self.move(robot, [tx, ty, self.info.surface_z + off + half])
        self.release(robot, obj, xy)
        self.move(robot, [tx, ty, self.info.surface_z + off + _APPROACH])
        return "ik_place"

    def _drawer(self, robot: str, opening: bool) -> str:
        if not self.info.drawer_object or not self.info.drawer_joint:
            return "drawer"
        d = self._body_pos(self.info.drawer_object)
        off = self.offset[robot]
        self.move(robot, [d[0], d[1] - 0.20, self.info.surface_z + off + 0.02])
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.info.drawer_joint)
        qadr = int(self.model.jnt_qposadr[jid])
        start = float(self.data.qpos[qadr])
        end = self.info.drawer_open_pos if opening else 0.0
        for f in range(1, 25):
            t = f / 24.0
            self.data.qpos[qadr] = (1.0 - t) * start + t * end
            self._refresh()
        return "open_drawer" if opening else "close_drawer"
