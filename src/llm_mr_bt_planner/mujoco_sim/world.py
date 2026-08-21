"""Composition and measured state for the heterogeneous three-robot scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SOURCE_DOCK_X = 0.00
DOOR_STAGING_X = 1.02
BEYOND_DOOR_X = 1.92
DESTINATION_DOCK_X = 3.00
DOCK_Y = 0.54
ROOM_ROUTE_Y = 0.76
PAYLOAD_HALF_SIZE = 0.020
PACKAGE_LID_HALF_SIZE = np.array([0.018, 0.018, 0.006])
RECOVERY_TASK_ID = "three_robot_spare_part_recovery"
WORKBENCH_TOP_Z = 0.490
PANDA_MOUNT_Z = 0.502

PANDA_MOUNT_POSES: dict[str, np.ndarray] = {
    "franka_a": np.array([-0.48, -0.25, PANDA_MOUNT_Z]),
    "franka_b": np.array([3.48, -0.25, PANDA_MOUNT_Z]),
}

PACKAGING_PANDA_MOUNT_POSES: dict[str, np.ndarray] = {
    "franka_a": np.array([-0.52, -0.25, PANDA_MOUNT_Z]),
    "franka_b": np.array([0.52, -0.25, PANDA_MOUNT_Z]),
}

STATION_POSES: dict[str, np.ndarray] = {
    "source_bin": np.array([0.00, -0.48, 0.510]),
    "source_cradle": np.array([0.00, 0.15, 0.550]),
    "destination_cradle": np.array([3.00, 0.15, 0.550]),
    "target_fixture": np.array([3.00, -0.48, 0.510]),
}

PACKAGING_STATION_POSES: dict[str, np.ndarray] = {
    "base_supply": np.array([-0.24, -0.48, 0.510]),
    "lid_supply": np.array([0.24, -0.48, 0.496]),
    "packing_station": np.array([0.00, 0.15, 0.510]),
    "lid_seal_target": np.array([0.00, 0.15, 0.536]),
    "delivery_station": np.array([3.00, 0.37, 0.550]),
}

RECOVERY_STATION_POSES: dict[str, np.ndarray] = {
    "primary_bin": np.array([-0.18, -0.48, 0.510]),
    "backup_bin": np.array([0.26, -0.48, 0.510]),
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

PACKAGING_STATION_PAD_HALF_EXTENTS: dict[str, np.ndarray] = {
    "base_supply": np.array([0.085, 0.075, 0.008]),
    "lid_supply": np.array([0.075, 0.065, 0.003]),
    "packing_station": np.array([0.10, 0.085, 0.008]),
    "delivery_station": np.array([0.12, 0.10, 0.028]),
}

RECOVERY_STATION_PAD_HALF_EXTENTS: dict[str, np.ndarray] = {
    "primary_bin": np.array([0.085, 0.075, 0.008]),
    "backup_bin": np.array([0.085, 0.075, 0.008]),
    "source_cradle": np.array([0.085, 0.075, 0.028]),
    "destination_cradle": np.array([0.10, 0.085, 0.028]),
    "target_fixture": np.array([0.085, 0.075, 0.012]),
}

DOCK_POSES: dict[str, np.ndarray] = {
    "source_dock": np.array([SOURCE_DOCK_X, DOCK_Y]),
    "door_staging": np.array([DOOR_STAGING_X, DOCK_Y]),
    "beyond_door": np.array([BEYOND_DOOR_X, DOCK_Y]),
    "destination_dock": np.array([DESTINATION_DOCK_X, DOCK_Y]),
}

PACKAGING_DOCK_POSES: dict[str, np.ndarray] = {
    "source_dock": np.array([SOURCE_DOCK_X, DOCK_Y]),
    "door_staging": np.array([DOOR_STAGING_X, DOCK_Y]),
    "beyond_door": np.array([BEYOND_DOOR_X, ROOM_ROUTE_Y]),
    "destination_dock": np.array([DESTINATION_DOCK_X, ROOM_ROUTE_Y]),
}

_ARENA_XML = """
<mujoco model="three_robot_team">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="100"/>
  <visual><global azimuth="135" elevation="-24" offwidth="1920" offheight="1080"/></visual>
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
    <camera name="packaging_recording" pos="1.5 -4.5 2.9" xyaxes="1 0 0 0 0.48 0.88"/>
    <camera name="courier_source" pos="0 -3.2 2.1" xyaxes="1 0 0 0 0.342 0.940" fovy="38"/>
    <camera name="courier_route" pos="1.5 4.5 3.2" xyaxes="-1 0 0 0 -0.500 0.866" fovy="42"/>
    <camera name="courier_destination" pos="3 -3.2 2.1" xyaxes="1 0 0 0 0.342 0.940" fovy="38"/>
    <camera name="packaging_assembly" pos="0 -3.2 2.1" xyaxes="1 0 0 0 0.342 0.940" fovy="38"/>
    <camera name="packaging_door" pos="0.8 3.5 2.2" xyaxes="-1 0 0 0 -0.400 0.916" fovy="38"/>
    <camera name="packaging_route" pos="2.3 3.5 2.2" xyaxes="-1 0 0 0 -0.400 0.916" fovy="38"/>
    <camera name="packaging_delivery" pos="3 -3.2 2.1" xyaxes="1 0 0 0 0.342 0.940" fovy="38"/>
    <camera name="recovery_source" pos="0 -3.0 1.9" xyaxes="1 0 0 0 0.330 0.944" fovy="36"/>
    <camera name="recovery_floor" pos="-1.3 1.4 1.25"
            xyaxes="-0.693109 -0.720833 0 0.349553 -0.336108 0.874554" fovy="42"/>
    <camera name="recovery_route" pos="1.5 4.5 3.2" xyaxes="-1 0 0 0 -0.500 0.866" fovy="42"/>
    <camera name="recovery_destination" pos="3 -3.2 2.1" xyaxes="1 0 0 0 0.342 0.940" fovy="38"/>
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
    object_body_ids: dict[str, int]
    payload_body_id: int
    base_body_id: int
    initial_base_xy: np.ndarray
    task_id: str = "three_robot_courier"
    source_location: str = "source_bin"
    door_joint_id: int | None = None
    qpos_writes_after_reset: int = 0
    active_payload_name: str = "payload"
    reset_count: int = 0

    @classmethod
    def build(cls, assets: Path, *, task_id: str = "three_robot_courier") -> "CourierWorld":
        packaging = task_id == "three_robot_packaging_delivery"
        recovery = task_id == RECOVERY_TASK_ID
        station_poses = (
            PACKAGING_STATION_POSES
            if packaging
            else RECOVERY_STATION_POSES
            if recovery
            else STATION_POSES
        )
        mount_poses = PACKAGING_PANDA_MOUNT_POSES if packaging else PANDA_MOUNT_POSES
        spec = mujoco.MjSpec.from_string(_ARENA_XML)
        if packaging:
            _add_packaging_scene(spec)
        elif recovery:
            _add_recovery_scene(spec)
        else:
            _add_courier_scene(spec)

        go2 = _load_child(assets / "unitree_go2" / "go2.xml", spec)
        z1 = _load_child(assets / "unitree_z1" / "z1_gripper.xml", spec)
        z1.body("link06").add_site(
            name="grasp_site",
            pos=[0.18205, 0.0, 0.01145],
            size=[0.012],
            rgba=[0.95, 0.2, 0.1, 0.8],
        )
        mount = go2.body("base").add_frame(name="z1_mount", pos=[0.05, 0.0, 0.10])
        mount.attach_body(z1.body("link00"), prefix="z1_")
        spec.worldbody.add_frame(name="go2_start", pos=[SOURCE_DOCK_X, DOCK_Y, 0.0]).attach_body(
            go2.body("base"), prefix="go2_"
        )

        _attach_panda(spec, assets, "franka_a_", mount_poses["franka_a"].tolist(), 0.0)
        _attach_panda(spec, assets, "franka_b_", mount_poses["franka_b"].tolist(), np.pi)

        if recovery:
            for object_name in ("primary_part", "spare_part"):
                _add_weld(
                    spec,
                    f"grip_franka_a_{object_name}",
                    "franka_a_hand",
                    object_name,
                )
                _add_weld(
                    spec,
                    f"grip_unitree_go2_z1_{object_name}",
                    "go2_z1_link06",
                    object_name,
                )
                _add_weld(
                    spec,
                    f"grip_franka_b_{object_name}",
                    "franka_b_hand",
                    object_name,
                )
                _add_weld(
                    spec,
                    f"install_fixture_{object_name}",
                    "target_fixture_body",
                    object_name,
                )
        else:
            _add_weld(spec, "grip_franka_a", "franka_a_hand", "payload")
            _add_weld(spec, "grip_unitree_go2_z1", "go2_z1_link06", "payload")
            _add_weld(spec, "grip_franka_b", "franka_b_hand", "payload")
        if packaging:
            _add_weld(spec, "grip_franka_b_package_lid", "franka_b_hand", "package_lid")
            _add_weld(spec, "package_seal", "payload", "package_lid")
        elif not recovery:
            _add_weld(spec, "install_fixture", "target_fixture_body", "payload")

        model = spec.compile()
        _configure_contacts_and_gravity(model)
        data = mujoco.MjData(model)
        grip_equalities = {}
        if recovery:
            for object_name in ("primary_part", "spare_part"):
                for robot in ("franka_a", "unitree_go2_z1", "franka_b"):
                    grip_equalities[f"{robot}:{object_name}"] = model.equality(
                        f"grip_{robot}_{object_name}"
                    ).id
                grip_equalities[f"target_fixture:{object_name}"] = model.equality(
                    f"install_fixture_{object_name}"
                ).id
        else:
            grip_equalities = {
                "franka_a": model.equality("grip_franka_a").id,
                "unitree_go2_z1": model.equality("grip_unitree_go2_z1").id,
                "franka_b": model.equality("grip_franka_b").id,
            }
        if packaging:
            grip_equalities.update(
                {
                    "franka_b:package_lid": model.equality("grip_franka_b_package_lid").id,
                    "package_seal": model.equality("package_seal").id,
                }
            )
        elif not recovery:
            grip_equalities["target_fixture"] = model.equality("install_fixture").id

        object_body_ids = (
            {
                "primary_part": model.body("primary_part").id,
                "spare_part": model.body("spare_part").id,
            }
            if recovery
            else {"payload": model.body("payload").id}
        )
        if packaging:
            object_body_ids["package_lid"] = model.body("package_lid").id

        world = cls(
            model=model,
            data=data,
            station_sites={name: model.site(f"station_{name}").id for name in station_poses},
            grip_equalities=grip_equalities,
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
            object_body_ids=object_body_ids,
            payload_body_id=(
                model.body("primary_part").id if recovery else model.body("payload").id
            ),
            base_body_id=model.body("go2_base").id,
            initial_base_xy=np.array([SOURCE_DOCK_X, DOCK_Y]),
            task_id=task_id,
            source_location=(
                "base_supply" if packaging else "primary_bin" if recovery else "source_bin"
            ),
            door_joint_id=model.joint("room_door_hinge").id if packaging else None,
            active_payload_name="primary_part" if recovery else "payload",
        )
        world.reset()
        return world

    def reset(self) -> None:
        self.reset_count += 1
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
            home = list(panda_home)
            if self.task_id == "three_robot_packaging_delivery":
                # Turn both elbows toward their own half of the shared bench.
                # With the courier home pose, the opposed hands overlap at the
                # bench center before either BT has started.
                home[0] = 0.8 if prefix == "franka_a_" else -0.8
            for index, value in enumerate(home, 1):
                self._set_qpos(f"{prefix}joint{index}", value)
                self.data.ctrl[self.model.actuator(f"{prefix}actuator{index}").id] = value
            self._set_qpos(f"{prefix}finger_joint1", 0.04)
            self._set_qpos(f"{prefix}finger_joint2", 0.04)
            self.data.ctrl[self.model.actuator(f"{prefix}actuator8").id] = 255.0

        poses = (
            PACKAGING_STATION_POSES
            if self.task_id == "three_robot_packaging_delivery"
            else RECOVERY_STATION_POSES
            if self.task_id == RECOVERY_TASK_ID
            else STATION_POSES
        )
        if self.task_id == RECOVERY_TASK_ID:
            self._set_free_body_pose("primary_part", poses["primary_bin"])
            self._set_free_body_pose("spare_part", poses["backup_bin"])
            self.active_payload_name = "primary_part"
        else:
            self._set_free_body_pose("payload", poses[self.source_location])
        if "package_lid" in self.object_body_ids:
            self._set_free_body_pose("package_lid", poses["lid_supply"])
        if self.door_joint_id is not None:
            door_qadr = int(self.model.jnt_qposadr[self.door_joint_id])
            self.data.qpos[door_qadr] = 0.0

        self.data.qvel[:] = 0
        self.data.ctrl[:12] = 0
        mujoco.mj_forward(self.model, self.data)
        self.qpos_writes_after_reset = 0

    @property
    def payload_position(self) -> np.ndarray:
        return self.object_position(self.active_payload_name)

    def object_position(self, name: str) -> np.ndarray:
        return self.data.xpos[self.object_body_ids[name]].copy()

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

    def dock_position(self, name: str) -> np.ndarray:
        poses = PACKAGING_DOCK_POSES if self.task_id == "three_robot_packaging_delivery" else DOCK_POSES
        return poses[name].copy()

    def equality_active(self, owner: str) -> bool:
        return bool(self.data.eq_active[self.grip_equalities[owner]])

    def robot_holding_any(self, robot: str) -> bool:
        keys = [key for key in self.grip_equalities if key == robot or key.startswith(f"{robot}:")]
        return any(self.equality_active(key) for key in keys)

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
        if ":" in owner:
            candidate = owner.rpartition(":")[2]
            if candidate in self.object_body_ids and candidate != "package_lid":
                self.active_payload_name = candidate

    def deactivate_weld(self, owner: str) -> None:
        self.data.eq_active[self.grip_equalities[owner]] = 0

    def set_cradle_holding_friction(self, location: str, *, enabled: bool) -> None:
        supported = {
            "source_cradle",
            "destination_cradle",
            "packing_station",
            "delivery_station",
        }
        if location not in supported:
            return
        geom_id = self.model.geom(f"{location}_pad").id
        self.model.geom_friction[geom_id] = [4.0, 0.08, 0.015] if enabled else [1.2, 0.02, 0.005]
        for object_name in self.object_body_ids:
            if object_name == "package_lid":
                continue
            geom_id = self.model.geom(f"{object_name}_geom").id
            self.model.geom_friction[geom_id] = (
                [3.0, 0.06, 0.012] if enabled else [1.4, 0.03, 0.008]
            )

    def object_contact_with(self, object_name: str, body_prefix: str) -> bool:
        object_body_id = self.object_body_ids[object_name]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if object_body_id not in {body1, body2}:
                continue
            other = body2 if body1 == object_body_id else body1
            if self.model.body(other).name.startswith(body_prefix):
                return True
        return False

    def payload_contact_with(self, body_prefix: str) -> bool:
        return self.object_contact_with("payload", body_prefix)

    def object_contact_with_z1_finger_pad(self, object_name: str, side: str) -> bool:
        pad_geoms = self.z1_finger_pad_geoms[side]
        object_body_id = self.object_body_ids[object_name]
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if object_body_id not in {body1, body2}:
                continue
            other_geom = int(contact.geom2 if body1 == object_body_id else contact.geom1)
            if other_geom in pad_geoms:
                return True
        return False

    def payload_contact_with_z1_finger_pad(self, side: str) -> bool:
        return self.object_contact_with_z1_finger_pad(self.active_payload_name, side)

    def apply_object_force(self, object_name: str, force: np.ndarray) -> None:
        self.data.xfrc_applied[self.object_body_ids[object_name], :3] = force

    def clear_object_force(self, object_name: str) -> None:
        self.data.xfrc_applied[self.object_body_ids[object_name], :] = 0

    @property
    def door_angle(self) -> float:
        if self.door_joint_id is None:
            return 0.0
        qadr = int(self.model.jnt_qposadr[self.door_joint_id])
        return float(self.data.qpos[qadr])

    def door_open(self) -> bool:
        return self.door_joint_id is not None and self.door_angle > 0.70

    def door_closed(self) -> bool:
        return self.door_joint_id is not None and abs(self.door_angle) < 0.12

    def finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def _set_free_body_pose(self, body_name: str, position: np.ndarray) -> None:
        body_id = self.object_body_ids[body_name]
        joint_id = int(self.model.body(body_id).jntadr[0])
        qadr = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[qadr : qadr + 7] = [*position, 1, 0, 0, 0]

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


def _add_weld(spec: mujoco.MjSpec, name: str, body1: str, body2: str) -> None:
    spec.add_equality(
        name=name,
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        active=0,
        name1=body1,
        name2=body2,
        solref=[0.005, 1.0],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
    )


def _configure_contacts_and_gravity(model: mujoco.MjModel) -> None:
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
            model.geom_friction[geom_id] = [1.6, 0.02, 0.01]
    for body_id in range(model.nbody):
        body_name = model.body(body_id).name
        if body_name.startswith(("franka_a_", "franka_b_", "go2_z1_")):
            model.body_gravcomp[body_id] = 1.0


def _add_courier_scene(spec: mujoco.MjSpec) -> None:
    _add_workbench(
        spec,
        "source",
        center_x=0.00,
        half_x=0.65,
        mounts={"panda": PANDA_MOUNT_POSES["franka_a"][:2]},
    )
    _add_workbench(
        spec,
        "destination",
        center_x=3.00,
        half_x=0.65,
        mounts={"panda": PANDA_MOUNT_POSES["franka_b"][:2]},
    )
    colors = {
        "source_bin": [0.75, 0.45, 0.10, 1.0],
        "source_cradle": [0.15, 0.48, 0.42, 1.0],
        "destination_cradle": [0.16, 0.62, 0.30, 1.0],
        "target_fixture": [0.68, 0.16, 0.18, 1.0],
    }
    _add_station_pads(spec, STATION_POSES, STATION_PAD_HALF_EXTENTS, colors)
    _add_payload(spec, STATION_POSES["source_bin"])


def _add_recovery_scene(spec: mujoco.MjSpec) -> None:
    _add_workbench(
        spec,
        "source",
        center_x=0.00,
        half_x=0.72,
        mounts={"panda": PANDA_MOUNT_POSES["franka_a"][:2]},
    )
    _add_workbench(
        spec,
        "destination",
        center_x=3.00,
        half_x=0.65,
        mounts={"panda": PANDA_MOUNT_POSES["franka_b"][:2]},
    )
    colors = {
        "primary_bin": [0.78, 0.42, 0.12, 1.0],
        "backup_bin": [0.18, 0.48, 0.82, 1.0],
        "source_cradle": [0.15, 0.48, 0.42, 1.0],
        "destination_cradle": [0.16, 0.62, 0.30, 1.0],
        "target_fixture": [0.68, 0.16, 0.18, 1.0],
    }
    _add_station_pads(
        spec,
        RECOVERY_STATION_POSES,
        RECOVERY_STATION_PAD_HALF_EXTENTS,
        colors,
    )
    _add_payload(
        spec,
        RECOVERY_STATION_POSES["primary_bin"],
        name="primary_part",
        color=[0.95, 0.52, 0.10, 1.0],
    )
    _add_payload(
        spec,
        RECOVERY_STATION_POSES["backup_bin"],
        name="spare_part",
        color=[0.12, 0.56, 0.95, 1.0],
    )


def _add_packaging_scene(spec: mujoco.MjSpec) -> None:
    _add_workbench(
        spec,
        "packing",
        center_x=0.00,
        half_x=0.78,
        mounts={
            "franka_a": PACKAGING_PANDA_MOUNT_POSES["franka_a"][:2],
            "franka_b": PACKAGING_PANDA_MOUNT_POSES["franka_b"][:2],
        },
    )
    _add_delivery_pedestal(spec)
    colors = {
        "base_supply": [0.16, 0.40, 0.76, 1.0],
        "lid_supply": [0.88, 0.64, 0.10, 1.0],
        "packing_station": [0.16, 0.62, 0.30, 1.0],
        "delivery_station": [0.52, 0.22, 0.72, 1.0],
    }
    public_poses = {
        name: pose for name, pose in PACKAGING_STATION_POSES.items() if name != "lid_seal_target"
    }
    _add_station_pads(spec, public_poses, PACKAGING_STATION_PAD_HALF_EXTENTS, colors)
    marker = spec.worldbody.add_body(
        name="lid_seal_target_marker", pos=PACKAGING_STATION_POSES["lid_seal_target"]
    )
    marker.add_site(name="station_lid_seal_target", size=[0.006], rgba=[1.0, 0.9, 0.1, 0.0])
    _add_payload(spec, PACKAGING_STATION_POSES["base_supply"], color=[0.16, 0.42, 0.78, 1.0])
    lid = spec.worldbody.add_body(name="package_lid", pos=PACKAGING_STATION_POSES["lid_supply"])
    lid.add_freejoint(name="package_lid_freejoint")
    lid.add_geom(
        name="package_lid_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=PACKAGE_LID_HALF_SIZE,
        mass=0.035,
        rgba=[0.95, 0.72, 0.12, 1.0],
        friction=[1.4, 0.03, 0.008],
        condim=6,
    )
    _add_room_and_door(spec)


def _add_station_pads(
    spec: mujoco.MjSpec,
    poses: dict[str, np.ndarray],
    extents: dict[str, np.ndarray],
    colors: dict[str, list[float]],
) -> None:
    for name, position in poses.items():
        body = spec.worldbody.add_body(name=f"{name}_body", pos=position)
        object_half_height = PACKAGE_LID_HALF_SIZE[2] if name == "lid_supply" else PAYLOAD_HALF_SIZE
        body.add_geom(
            name=f"{name}_pad",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, 0.0, -(float(object_half_height) + float(extents[name][2]))],
            size=extents[name],
            rgba=colors[name],
            friction=[1.2, 0.02, 0.005],
        )
        body.add_site(name=f"station_{name}", size=[0.006], rgba=[1.0, 0.9, 0.1, 0.0])


def _add_payload(
    spec: mujoco.MjSpec,
    position: np.ndarray,
    *,
    name: str = "payload",
    color: list[float] | None = None,
) -> None:
    payload = spec.worldbody.add_body(name=name, pos=position)
    payload.add_freejoint(name=f"{name}_freejoint")
    payload.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[PAYLOAD_HALF_SIZE] * 3,
        mass=0.12,
        rgba=color or [0.95, 0.80, 0.12, 1.0],
        friction=[1.4, 0.03, 0.008],
        condim=6,
    )


def _add_room_and_door(spec: mujoco.MjSpec) -> None:
    wall = spec.worldbody.add_body(name="room_partition", pos=[0.0, 0.0, 0.0])
    wall_color = [0.64, 0.68, 0.72, 1.0]
    wall.add_geom(
        name="room_wall_lower",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.50, -1.50, 0.75],
        size=[0.055, 1.50, 0.75],
        rgba=wall_color,
    )
    wall.add_geom(
        name="room_wall_upper",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.50, 2.01, 0.75],
        size=[0.055, 0.99, 0.75],
        rgba=wall_color,
    )
    wall.add_geom(
        name="room_wall_header",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[1.50, 0.50, 1.39],
        size=[0.055, 0.50, 0.11],
        rgba=wall_color,
    )

    # The frame leaves real clearance around the dynamic panel.  Earlier
    # dimensions overlapped both the jamb and header, effectively pinning the
    # hinge even though its joint was free.
    door = spec.worldbody.add_body(name="room_door", pos=[1.50, 0.04, 0.0])
    door.add_joint(
        name="room_door_hinge",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, -1.0],
        range=[0.0, 1.65],
        damping=0.12,
        armature=0.006,
    )
    door.add_geom(
        name="room_door_panel",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.45, 0.60],
        size=[0.025, 0.43, 0.60],
        mass=0.45,
        rgba=[0.32, 0.18, 0.09, 1.0],
        friction=[0.8, 0.02, 0.005],
    )
    door.add_geom(
        name="room_door_push_bar",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[-0.07, 0.20, 0.58, -0.07, 0.66, 0.58],
        size=[0.025],
        mass=0.05,
        rgba=[0.75, 0.15, 0.08, 1.0],
    )


def _add_delivery_pedestal(spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body(name="delivery_pedestal", pos=[3.0, 0.37, 0.0])
    body.add_geom(
        name="delivery_pedestal_top",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, 0.455],
        size=[0.34, 0.30, 0.035],
        rgba=[0.28, 0.30, 0.34, 1.0],
        friction=[1.1, 0.02, 0.005],
    )
    for x in (-0.27, 0.27):
        for y in (-0.23, 0.23):
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[x, y, 0.205],
                size=[0.025, 0.025, 0.205],
                rgba=[0.14, 0.15, 0.16, 1.0],
            )


def _add_workbench(
    spec: mujoco.MjSpec,
    name: str,
    *,
    center_x: float,
    half_x: float,
    mounts: dict[str, np.ndarray],
) -> None:
    """Add a laboratory workbench that physically supports one or two Panda bases."""
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
    for label, panda_mount in mounts.items():
        mount_local = np.asarray(panda_mount, dtype=float) - np.array([center_x, center_y])
        plate_name = f"{name}_panda_mounting_plate" if label == "panda" else f"{name}_{label}_mounting_plate"
        top.add_geom(
            name=plate_name,
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
