"""Per-action skills that turn a symbolic BT action into a physical motion.

Each skill mutates ``data`` (joint targets / free-body poses) for the object an
action manipulates; the backend then steps physics so the change settles and is
visible in the viewer. This is **Stage 1**: motions are scripted teleports +
gravity settling, chosen so the *structure and ordering* of the generated BT
play out on the real robot models. Stage 2 would replace these with Jacobian-IK
reaches and gripper grasps.

A skill has the signature ``skill(model, data, event, info) -> None`` where
``event`` is the simulation action event (``name``, ``parameters``, ``robot``,
...) and ``info`` is the :class:`SceneInfo` from the scene builder.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco

from .mujoco_scene import SceneInfo

Skill = Callable[[mujoco.MjModel, mujoco.MjData, dict, SceneInfo], None]

LIFT = 0.22   # how high above the table a picked part is raised (Stage-1 teleport)
REST = 0.035  # half-size of a part: its centre height when resting on the table


def _set_body_xyz(model, data: mujoco.MjData, body: str, x: float, y: float, z: float) -> bool:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        return False
    jadr = model.body_jntadr[bid]
    if jadr < 0:
        return False
    qadr = int(model.jnt_qposadr[jadr])
    data.qpos[qadr : qadr + 3] = [x, y, z]
    data.qpos[qadr + 3 : qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    # Zero just this free body's 6 velocity DOFs so it settles instead of flinging.
    dadr = int(model.jnt_dofadr[jadr])
    data.qvel[dadr : dadr + 6] = 0.0
    return True


def _set_slide(model: mujoco.MjModel, data: mujoco.MjData, joint: str, value: float) -> bool:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0:
        return False
    data.qpos[model.jnt_qposadr[jid]] = value
    return True


def _object_param(event: dict, info: SceneInfo) -> str | None:
    for p in event.get("parameters", []):
        if p in info.object_bodies:
            return p
    return None


def _target_xy(event: dict, info: SceneInfo) -> tuple[float, float]:
    for p in event.get("parameters", []):
        if p in info.location_xy:
            return info.location_xy[p]
    loc = info.robot_location.get(event.get("robot", ""))
    return info.location_xy.get(loc, (0.0, 0.0)) if loc else (0.0, 0.0)


def rest_z(info: SceneInfo, obj: str) -> float:
    """Table-top height at which this part's centre rests (depends on its shape)."""
    return info.surface_z + info.object_half_z.get(obj, REST)


# --- skills -----------------------------------------------------------------

def skill_open_drawer(model, data, event, info: SceneInfo) -> None:
    if info.drawer_joint:
        _set_slide(model, data, info.drawer_joint, info.drawer_open_pos)


def skill_close_drawer(model, data, event, info: SceneInfo) -> None:
    if info.drawer_joint:
        _set_slide(model, data, info.drawer_joint, 0.0)


def skill_pick(model, data, event, info: SceneInfo) -> None:
    obj = _object_param(event, info)
    if obj is None:
        return
    x, y = _target_xy(event, info)  # lift it in place, above the table
    _set_body_xyz(model, data, obj, x, y, info.surface_z + LIFT)


def skill_place(model, data, event, info: SceneInfo) -> None:
    obj = _object_param(event, info)
    if obj is None:
        return
    x, y = _target_xy(event, info)
    _set_body_xyz(model, data, obj, x, y, rest_z(info, obj))


def skill_settle(model, data, event, info: SceneInfo) -> None:
    """No explicit motion; the backend still steps physics to settle."""
    return None


# Exact-name registry for the capabilities used by the bundled scenarios, plus a
# keyword fallback so any new capability still maps to a reasonable motion.
SKILL_REGISTRY: dict[str, Skill] = {
    "open_drawer": skill_open_drawer,
    "close_drawer": skill_close_drawer,
    "pick_tray": skill_pick,
    "pick_tool": skill_pick,
    "pick_gear": skill_pick,
    "place_tray": skill_place,
    "place_tool": skill_place,
    "return_tool": skill_place,
    "mount_gear": skill_place,
    "fasten_screw": skill_place,
}

_PICK_WORDS = ("pick", "grasp", "grab", "hold")
_PLACE_WORDS = ("place", "put", "return", "mount", "insert", "fasten", "attach", "install", "deliver")


def resolve_skill(action_name: str | None) -> Skill:
    name = (action_name or "").lower()
    if name in SKILL_REGISTRY:
        return SKILL_REGISTRY[name]
    if "drawer" in name and ("close" in name or "shut" in name):
        return skill_close_drawer
    if "drawer" in name and "open" in name:
        return skill_open_drawer
    if any(w in name for w in _PICK_WORDS):
        return skill_pick
    if any(w in name for w in _PLACE_WORDS):
        return skill_place
    return skill_settle
