"""Composition and measured state for the heterogeneous courier scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SOURCE_DOCK_X = 0.00
DESTINATION_DOCK_X = 3.00
DOCK_Y = 0.54
PAYLOAD_HALF_SIZE = 0.020
WORKBENCH_TOP_Z = 0.490
PANDA_MOUNT_Z = 0.502

PANDA_MOUNT_POSES: dict[str, np.ndarray] = {
    "franka_a": np.array([-0.48, -0.25, PANDA_MOUNT_Z]),
    "franka_b": np.array([3.48, -0.25, PANDA_MOUNT_Z]),
}

STATION_POSES: dict[str, np.ndarray] = {
    "source_bin": np.array([0.00, -0.48, 0.510]),
    "source_cradle": np.array([0.00, 0.15, 0.550]),
    "destination_cradle": np.array([3.00, 0.15, 0.550]),
    "target_fixture": np.array([3.00, -0.48, 0.510]),
}

STATION_PAD_HALF_EXTENTS: dict[str, np.ndarray] = {
    "source_bin": np.array([0.085, 0.075, 0.008]),
    "source_cradle": np.array([0.085, 0.075, 0.028]),
    "destination_cradle": np.array([0.10, 0.085, 0.028]),
    "target_fixture": np.array([0.085, 0.075, 0.012]),
}

_ARENA_XML = """
<mujoco model="three_robot_courier">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="100"/>
  <visual><global azimuth="135" elevation="-24"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.45 0.65" rgb2="0.02 0.03 0.05" width="512" height="3072"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.18 0.20 0.23" rgb2="0.10 0.11 0.13" width="512" height="512"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="8 8" reflectance="0.08"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <light pos="1 -3 2.5" dir="0 1 -0.5" diffuse="0.45 0.45 0.45"/>
    <geom name="floor" type="plane" size="4 3 0.1" material="floor_mat"
          friction="1.0 0.02 0.005" condim="6"/>
    <camera name="overview" pos="5.5 -0.20 3.2" xyaxes="0 1 0 -0.55 0 0.835"/>
  </worldbody>
</mujoco>
"""


@dataclass
class CourierWorld:
    model: mujoco.MjModel
    data: mujoco.MjData
    station_sites: dict[str, int]
    grip_equalities: dict[str, int]
    z1_finger_pad_geoms: dict[str, frozenset[int]]
    payload_body_id: int
    base_body_id: int
    initial_base_xy: np.ndarray
    qpos_writes_after_reset: int = 0

    @classmethod
    def build(cls, assets: Path) -> "CourierWorld":
        spec = mujoco.MjSpec.from_string(_ARENA_XML)
        _add_stations(spec)

        go2 = _load_child(assets / "unitree_go2" / "go2.xml", spec)
        z1 = _load_child(assets / "unitree_z1" / "z1_gripper.xml", spec)
        z1.body("link06").add_site(
            name="grasp_site",
            # Midpoint between the finger-pad contact surfaces at the
            # -0.50 rad command used for the 40 mm payload.
            pos=[0.18205, 0.0, 0.01145],
            size=[0.012],
            rgba=[0.95, 0.2, 0.1, 0.8],
        )
        mount = go2.body("base").add_frame(name="z1_mount", pos=[0.05, 0.0, 0.10])
        mount.attach_body(z1.body("link00"), prefix="z1_")

        spec.worldbody.add_frame(name="go2_start", pos=[SOURCE_DOCK_X, DOCK_Y, 0.0]).attach_body(
            go2.body("base"), prefix="go2_"
        )

        _attach_panda(spec, assets, "franka_a_", PANDA_MOUNT_POSES["franka_a"].tolist(), 0.0)
        _attach_panda(spec, assets, "franka_b_", PANDA_MOUNT_POSES["franka_b"].tolist(), np.pi)

        spec.add_equality(
            name="grip_franka_a",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=0,
            name1="franka_a_hand",
            name2="payload",
            solref=[0.005, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )
        spec.add_equality(
            name="grip_unitree_go2_z1",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=0,
            name1="go2_z1_link06",
            name2="payload",
            solref=[0.005, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )
        spec.add_equality(
            name="grip_franka_b",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=0,
            name1="franka_b_hand",
            name2="payload",
            solref=[0.005, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )
        spec.add_equality(
            name="install_fixture",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            active=0,
            name1="target_fixture_body",
            name2="payload",
            solref=[0.005, 1.0],
            solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        )

        model = spec.compile()
        # The Menagerie gripper includes conservative collision meshes for the
        # stator/mover housings as well as dedicated box geoms for the actual
        # finger pads.  Use the pads as grasp contacts so the housing cannot
        # eject an object before the jaws surround it; visuals remain intact.
        for geom_id in range(model.ngeom):
            body_name = model.body(int(model.geom_bodyid[geom_id])).name
            if body_name in {"go2_z1_link06", "go2_z1_gripperMover"} and model.geom_type[
                geom_id
            ] == mujoco.mjtGeom.mjGEOM_MESH:
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
            if (
                body_name in {"go2_z1_link06", "go2_z1_gripperMover"}
                and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                and np.allclose(model.geom_size[geom_id], [0.014, 0.015, 0.004])
            ):
                model.geom_solref[geom_id] = [0.02, 1.0]
                model.geom_solimp[geom_id] = [0.85, 0.95, 0.002, 0.5, 2.0]
            if model.geom(geom_id).name in {"go2_FL", "go2_FR", "go2_RL", "go2_RR"}:
                # Rubber-foot contact: prevent arm reaction forces from sliding
                # a stationary Go2 while preserving fully dynamic contacts.
                model.geom_friction[geom_id] = [1.6, 0.02, 0.01]
        # Match mjctrl's arm setup: compensate arm-link gravity while contacts and
        # actuator dynamics remain active.  The Go2 trunk/legs and payload retain
        # their full gravity so locomotion and transport stay physically coupled.
        for body_id in range(model.nbody):
            body_name = model.body(body_id).name
            if body_name.startswith("franka_a_") or body_name.startswith("franka_b_") or body_name.startswith(
                "go2_z1_"
            ):
                model.body_gravcomp[body_id] = 1.0
        data = mujoco.MjData(model)
        world = cls(
            model=model,
            data=data,
            station_sites={name: model.site(f"station_{name}").id for name in STATION_POSES},
            grip_equalities={
                "franka_a": model.equality("grip_franka_a").id,
                "unitree_go2_z1": model.equality("grip_unitree_go2_z1").id,
                "franka_b": model.equality("grip_franka_b").id,
                "target_fixture": model.equality("install_fixture").id,
            },
            z1_finger_pad_geoms={
                side: frozenset(
                    geom_id
                    for geom_id in range(model.ngeom)
                    if model.body(int(model.geom_bodyid[geom_id])).name == body_name
                    and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                    and np.allclose(model.geom_size[geom_id], [0.014, 0.015, 0.004])
                )
                for side, body_name in {
                    "fixed": "go2_z1_link06",
                    "moving": "go2_z1_gripperMover",
                }.items()
            },
            payload_body_id=model.body("payload").id,
            base_body_id=model.body("go2_base").id,
            initial_base_xy=np.array([SOURCE_DOCK_X, DOCK_Y]),
        )
        world.reset()
        return world

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.eq_active[:] = 0

        base_joint_id = int(self.model.body(self.base_body_id).jntadr[0])
        base_qadr = int(self.model.jnt_qposadr[base_joint_id])
        self.data.qpos[base_qadr : base_qadr + 7] = [SOURCE_DOCK_X, DOCK_Y, 0.445, 1, 0, 0, 0]

        for leg in ("FL", "FR", "RL", "RR"):
            hip = 0.10 if leg in {"FL", "RL"} else -0.10
            for joint, value in zip(("hip", "thigh", "calf"), (hip, 0.9, -1.8)):
                self._set_qpos(f"go2_{leg}_{joint}_joint", value)
        for index, value in enumerate((0.0, 0.785, -0.261, -0.523, 0.0, 0.0), 1):
            self._set_qpos(f"go2_z1_joint{index}", value)
            self.data.ctrl[self.model.actuator(f"go2_z1_motor{index}").id] = value
        self._set_qpos("go2_z1_jointGripper", -0.75)
        self.data.ctrl[self.model.actuator("go2_z1_motorGripper").id] = -0.75

        panda_home = (0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853)
        for prefix in ("franka_a_", "franka_b_"):
            for index, value in enumerate(panda_home, 1):
                self._set_qpos(f"{prefix}joint{index}", value)
                self.data.ctrl[self.model.actuator(f"{prefix}actuator{index}").id] = value
            self._set_qpos(f"{prefix}finger_joint1", 0.04)
            self._set_qpos(f"{prefix}finger_joint2", 0.04)
            self.data.ctrl[self.model.actuator(f"{prefix}actuator8").id] = 255.0

        payload_joint = int(self.model.body(self.payload_body_id).jntadr[0])
        payload_qadr = int(self.model.jnt_qposadr[payload_joint])
        self.data.qpos[payload_qadr : payload_qadr + 7] = [*STATION_POSES["source_bin"], 1, 0, 0, 0]
        self.data.qvel[:] = 0
        self.data.ctrl[:12] = 0
        mujoco.mj_forward(self.model, self.data)
        self.qpos_writes_after_reset = 0

    @property
    def payload_position(self) -> np.ndarray:
        return self.data.xpos[self.payload_body_id].copy()

    @property
    def base_position(self) -> np.ndarray:
        return self.data.xpos[self.base_body_id].copy()

    @property
    def base_velocity(self) -> np.ndarray:
        base_joint_id = int(self.model.body(self.base_body_id).jntadr[0])
        address = int(self.model.jnt_dofadr[base_joint_id])
        return self.data.qvel[address : address + 6].copy()

    def site_position(self, name: str) -> np.ndarray:
        return self.data.site(f"station_{name}").xpos.copy()

    def equality_active(self, owner: str) -> bool:
        return bool(self.data.eq_active[self.grip_equalities[owner]])

    def activate_weld(self, owner: str) -> None:
        """Activate a weld at the current relative pose, avoiding any object snap."""
        equality_id = self.grip_equalities[owner]
        body1 = int(self.model.eq_obj1id[equality_id])
        body2 = int(self.model.eq_obj2id[equality_id])
        rotation1 = self.data.xmat[body1].reshape(3, 3)
        relative_position = rotation1.T @ (self.data.xpos[body2] - self.data.xpos[body1])
        inverse_q1 = np.empty(4)
        relative_quat = np.empty(4)
        mujoco.mju_negQuat(inverse_q1, self.data.xquat[body1])
        mujoco.mju_mulQuat(relative_quat, inverse_q1, self.data.xquat[body2])
        self.model.eq_data[equality_id, :6] = 0
        self.model.eq_data[equality_id, 3:6] = relative_position
        self.model.eq_data[equality_id, 6:10] = relative_quat
        self.model.eq_data[equality_id, 10] = 0.05
        self.data.eq_active[equality_id] = 1

    def deactivate_weld(self, owner: str) -> None:
        self.data.eq_active[self.grip_equalities[owner]] = 0

    def set_cradle_holding_friction(self, location: str, *, enabled: bool) -> None:
        if location not in {"source_cradle", "destination_cradle"}:
            return
        geom_id = self.model.geom(f"{location}_pad").id
        self.model.geom_friction[geom_id] = [4.0, 0.08, 0.015] if enabled else [1.2, 0.02, 0.005]
        payload_geom_id = self.model.geom("payload_geom").id
        self.model.geom_friction[payload_geom_id] = (
            [3.0, 0.06, 0.012] if enabled else [1.4, 0.03, 0.008]
        )

    def payload_contact_with(self, body_prefix: str) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if self.payload_body_id not in {body1, body2}:
                continue
            other = body2 if body1 == self.payload_body_id else body1
            if self.model.body(other).name.startswith(body_prefix):
                return True
        return False

    def payload_contact_with_z1_finger_pad(self, side: str) -> bool:
        pad_geoms = self.z1_finger_pad_geoms[side]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if self.payload_body_id not in {body1, body2}:
                continue
            other_geom = int(contact.geom2 if body1 == self.payload_body_id else contact.geom1)
            if other_geom in pad_geoms:
                return True
        return False

    def finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def _set_qpos(self, joint_name: str, value: float) -> None:
        joint = self.model.joint(joint_name)
        self.data.qpos[int(joint.qposadr[0])] = value


def _load_child(path: Path, parent: mujoco.MjSpec) -> mujoco.MjSpec:
    child = mujoco.MjSpec.from_file(str(path))
    for key in list(child.keys):
        child.delete(key)
    child.option.integrator = parent.option.integrator
    child.option.cone = parent.option.cone
    child.option.impratio = parent.option.impratio
    child.option.timestep = parent.option.timestep
    return child


def _attach_panda(spec: mujoco.MjSpec, assets: Path, prefix: str, pos: list[float], yaw: float) -> None:
    panda = _load_child(assets / "franka_emika_panda" / "panda.xml", spec)
    panda.body("hand").add_site(
        name="grasp_site",
        pos=[0.0, 0.0, 0.10],
        size=[0.012],
        rgba=[0.95, 0.2, 0.1, 0.8],
    )
    spec.worldbody.add_frame(name=f"{prefix}mount", pos=pos, euler=[0.0, 0.0, yaw]).attach_body(
        panda.body("link0"), prefix=prefix
    )


def _add_stations(spec: mujoco.MjSpec) -> None:
    _add_workbench(
        spec,
        "source",
        center_x=0.00,
        half_x=0.65,
        panda_mount=PANDA_MOUNT_POSES["franka_a"][:2],
    )
    _add_workbench(
        spec,
        "destination",
        center_x=3.00,
        half_x=0.65,
        panda_mount=PANDA_MOUNT_POSES["franka_b"][:2],
    )

    colors = {
        "source_bin": [0.75, 0.45, 0.10, 1.0],
        "source_cradle": [0.15, 0.48, 0.42, 1.0],
        "destination_cradle": [0.16, 0.62, 0.30, 1.0],
        "target_fixture": [0.68, 0.16, 0.18, 1.0],
    }
    for name, position in STATION_POSES.items():
        body = spec.worldbody.add_body(name=f"{name}_body", pos=position)
        body.add_geom(
            name=f"{name}_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, -(PAYLOAD_HALF_SIZE + float(STATION_PAD_HALF_EXTENTS[name][2]))],
            size=STATION_PAD_HALF_EXTENTS[name],
            rgba=colors[name],
            friction=[1.2, 0.02, 0.005],
        )
        # Sites remain exact controller/predicate references but are invisible
        # in the finished scene.  Unlike geoms, sites never create contacts.
        body.add_site(name=f"station_{name}", size=[0.006], rgba=[1.0, 0.9, 0.1, 0.0])

    payload = spec.worldbody.add_body(name="payload", pos=STATION_POSES["source_bin"])
    payload.add_freejoint(name="payload_freejoint")
    payload.add_geom(
        name="payload_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PAYLOAD_HALF_SIZE] * 3,
        mass=0.12,
        rgba=[0.95, 0.80, 0.12, 1.0],
        friction=[1.4, 0.03, 0.008],
        condim=6,
    )


def _add_workbench(
    spec: mujoco.MjSpec,
    name: str,
    *,
    center_x: float,
    half_x: float,
    panda_mount: np.ndarray,
) -> None:
    """Add a laboratory workbench that physically supports a Panda base."""
    center_y = -0.25
    half_y = 0.50
    top = spec.worldbody.add_body(name=f"{name}_worktable", pos=[center_x, center_y, 0.0])
    top.add_geom(
        name=f"{name}_worktable_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.455],
        size=[half_x, half_y, 0.035],
        rgba=[0.43, 0.27, 0.14, 1.0],
        friction=[1.1, 0.02, 0.005],
    )
    top.add_geom(
        name=f"{name}_worktable_apron",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.405],
        size=[half_x - 0.03, half_y - 0.03, 0.018],
        rgba=[0.18, 0.19, 0.20, 1.0],
    )
    mount_local = np.asarray(panda_mount, dtype=float) - np.array([center_x, center_y])
    top.add_geom(
        name=f"{name}_panda_mounting_plate",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[float(mount_local[0]), float(mount_local[1]), 0.496],
        size=[0.15, 0.006],
        rgba=[0.20, 0.22, 0.24, 1.0],
        friction=[1.2, 0.02, 0.005],
    )
    for x_side, x in (("left", -(half_x - 0.06)), ("right", half_x - 0.06)):
        for y_side, y in (("rear", -(half_y - 0.06)), ("front", half_y - 0.06)):
            top.add_geom(
                name=f"{name}_worktable_leg_{x_side}_{y_side}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, 0.205],
                size=[0.026, 0.026, 0.205],
                rgba=[0.14, 0.15, 0.16, 1.0],
            )
