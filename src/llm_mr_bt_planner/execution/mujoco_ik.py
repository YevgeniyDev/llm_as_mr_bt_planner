"""Legacy hand-rolled inverse kinematics + kinematic grasp for the MuJoCo backend.

.. note::
   This module is **superseded and no longer used at runtime**. The ``ik`` fidelity
   now runs through :class:`llm_mr_bt_planner.execution.mujoco_physics.ArmController`
   (``mink`` differential IK with collision avoidance); ``mujoco_backend`` no longer
   imports this file. It is kept for reference; ``docs/roadmap.md`` tracks removing
   or archiving it. Full friction-based grasping remains the next rung either way.

``ik_reach`` drives a robot's arm joints so its end-effector approaches a world
target, using damped least squares on the body Jacobian (``mj_jacBody``). Grasping
is modeled kinematically: ``skill_ik_pick`` reaches the object then marks it held,
and each held object is snapped to its gripper every physics step; ``skill_ik_place``
reaches the drop target and releases, so the part falls/rests under gravity.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .mujoco_scene import ArmHandles, SceneInfo
from .mujoco_skills import (
    _object_param,
    _set_body_xyz,
    _target_xy,
    rest_z,
    skill_close_drawer,
    skill_open_drawer,
    skill_settle,
)

_PICK_WORDS = ("pick", "grasp", "grab", "hold")
_PLACE_WORDS = ("place", "put", "return", "mount", "insert", "fasten", "attach", "install", "deliver")


def _body_xpos(model: mujoco.MjModel, data: mujoco.MjData, body: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return np.array(data.xpos[bid]) if bid >= 0 else np.zeros(3)


def ik_reach(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmHandles,
    target: np.ndarray,
    iters: int = 60,
    tol: float = 0.03,
    damping: float = 0.12,
) -> bool:
    """Damped-least-squares IK: move ``arm``'s joints so its EE reaches ``target``.
    Mutates ``data.qpos`` in place and returns True if it got within ``tol``."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)
    if ee_id < 0:
        return False
    dof_adr = []
    qpos_adr = []
    jnt_range = []
    for name in arm.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        dof_adr.append(int(model.jnt_dofadr[jid]))
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        jnt_range.append(tuple(model.jnt_range[jid]) if model.jnt_limited[jid] else (-np.inf, np.inf))
    if not dof_adr:
        return False

    jacp = np.zeros((3, model.nv))
    target = np.asarray(target, dtype=float)
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        err = target - np.array(data.xpos[ee_id])
        if np.linalg.norm(err) < tol:
            return True
        mujoco.mj_jacBody(model, data, jacp, None, ee_id)
        jac = jacp[:, dof_adr]  # 3 x k
        # dq = J^T (J J^T + λ²I)^-1 err
        lam2 = damping * damping
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(3), err)
        for k, qadr in enumerate(qpos_adr):
            lo, hi = jnt_range[k]
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + dq[k], lo, hi))
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - np.array(data.xpos[ee_id]))) < tol


def snap_held_to_gripper(model: mujoco.MjModel, data: mujoco.MjData, info: SceneInfo) -> None:
    """Place every held object at its holder's end-effector (kinematic grasp)."""
    for obj, robot in info.held.items():
        arm = info.arms.get(robot)
        if arm is None:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj)
        if bid < 0:
            continue
        jadr = model.body_jntadr[bid]
        if jadr < 0:
            continue
        qadr = int(model.jnt_qposadr[jadr])
        ee = _body_xpos(model, data, arm.ee_body)
        data.qpos[qadr : qadr + 3] = [float(ee[0]), float(ee[1]), float(ee[2] - 0.05)]
        data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        dadr = int(model.jnt_dofadr[jadr])
        data.qvel[dadr : dadr + 6] = 0.0


# --- IK skills (signature matches mujoco_skills.Skill) -----------------------

def skill_ik_pick(model, data, event, info: SceneInfo) -> None:
    obj = _object_param(event, info)
    robot = event.get("robot", "")
    arm = info.arms.get(robot)
    if obj is None or arm is None:
        return
    ik_reach(model, data, arm, _body_xpos(model, data, obj))
    info.held[obj] = robot


def skill_ik_place(model, data, event, info: SceneInfo) -> None:
    obj = _object_param(event, info)
    arm = info.arms.get(event.get("robot", ""))
    if obj is None:
        return
    x, y = _target_xy(event, info)
    if arm is not None:
        ik_reach(model, data, arm, np.array([x, y, info.surface_z + 0.16]))
    info.held.pop(obj, None)  # release the grasp
    _set_body_xyz(model, data, obj, x, y, rest_z(info, obj))  # set it down on the table


def resolve_skill_ik(action_name: str | None):
    """Like resolve_skill but pick/place reach with IK; drawer/settle unchanged."""
    name = (action_name or "").lower()
    if "drawer" in name and ("close" in name or "shut" in name):
        return skill_close_drawer
    if "drawer" in name and "open" in name:
        return skill_open_drawer
    if any(w in name for w in _PICK_WORDS):
        return skill_ik_pick
    if any(w in name for w in _PLACE_WORDS):
        return skill_ik_place
    return skill_settle


def ik_reach_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: ArmHandles,
    target_pos: np.ndarray,
    target_quat: np.ndarray | None = None,
    iters: int = 120,
    tol: float = 0.008,
    damping: float = 0.12,
    ori_weight: float = 0.5,
) -> bool:
    """6-DOF damped-least-squares IK: move ``arm`` so its end-effector reaches
    ``target_pos`` and (optionally) ``target_quat`` orientation. Keeps the gripper
    pointed the way we ask (e.g. straight down) instead of twisting. Mutates
    ``data.qpos`` for the arm joints only; returns True if within ``tol``."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.ee_body)
    if ee_id < 0:
        return False
    dof_adr, qpos_adr, jnt_range = [], [], []
    for name in arm.joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        dof_adr.append(int(model.jnt_dofadr[jid]))
        qpos_adr.append(int(model.jnt_qposadr[jid]))
        jnt_range.append(tuple(model.jnt_range[jid]) if model.jnt_limited[jid] else (-np.inf, np.inf))
    if not dof_adr:
        return False

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    target_pos = np.asarray(target_pos, float)
    rot_err = np.zeros(3)
    for _ in range(iters):
        mujoco.mj_forward(model, data)
        perr = target_pos - np.array(data.xpos[ee_id])
        if target_quat is not None:
            # mju_subQuat gives the error in the EE's *local* frame; rotate it into
            # world frame to match mj_jacBody's world-frame rotational Jacobian.
            local = np.zeros(3)
            mujoco.mju_subQuat(local, np.asarray(target_quat, float), np.array(data.xquat[ee_id]))
            rot_err = ori_weight * (np.array(data.xmat[ee_id]).reshape(3, 3) @ local)
        err = np.concatenate([perr, rot_err])
        if np.linalg.norm(perr) < tol and (target_quat is None or np.linalg.norm(rot_err) < 0.05):
            return True
        mujoco.mj_jacBody(model, data, jacp, jacr, ee_id)
        jac = np.vstack([jacp[:, dof_adr], jacr[:, dof_adr]])  # 6 x k
        lam2 = damping * damping
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam2 * np.eye(6), err)
        for k, qadr in enumerate(qpos_adr):
            lo, hi = jnt_range[k]
            data.qpos[qadr] = float(np.clip(data.qpos[qadr] + dq[k], lo, hi))
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target_pos - np.array(data.xpos[ee_id]))) < tol
