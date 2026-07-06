"""Build a MuJoCo scene for a :class:`Scenario` from real menagerie robots.

The scenario's symbolic robots/objects/locations are mapped onto physical models:

* ``go2_z1`` (mobile_manipulator) -> Unitree Go2 base + Unitree Z1 arm mounted on
  its back,
* every ``manipulator`` (franka1/franka2) -> Franka Emika Panda,

all sourced from ``third_party/mujoco_menagerie`` and attached (with a per-robot
name prefix) into one parent model. The robots stand in a ring around a shared
work table posed in a natural "ready" stance; objects become shaped free bodies
with realistic materials placed on the table, and the parts drawer becomes a
cabinet with a sliding drawer so open/close is a real joint motion.

Robots are attached fixed-base (no free joint) so they stand still for
visualization; only objects and the drawer have movable joints driven by skills.
Materials, textures, lighting and shaped parts are tuned so the generated
behavior trees replay on a workcell that reads as a believable assembly scene.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import mujoco

from ..config import PROJECT_ROOT
from ..domain import Scenario

# Menagerie model files, relative to the menagerie checkout.
GO2_XML = "unitree_go2/go2.xml"
Z1_XML = "unitree_z1/z1_gripper.xml"
PANDA_XML = "franka_emika_panda/panda.xml"

# Workcell geometry: robots stand on the floor in a ring facing a central table;
# parts rest on the table top so they never float. Tuned so the table centre sits
# within the arms' reach while keeping the (wide) robots clear of each other.
TABLE_HALF = 0.45          # table top half-extent (x, y)
TABLE_TOP_Z = 0.42         # height of the table *surface* parts rest on
TABLE_TOP_HALF_Z = 0.02    # half-thickness of the table top slab
LOC_RING = 0.26            # radius of the per-location spots on the table top
# Reach-aware mounting: bench arms sit on pedestals at the table edge so their
# gripper actually reaches the parts; the mobile manipulator stands a touch back.
FRANKA_RING = 0.60         # Panda pedestal distance from the table centre
GO2_RING = 0.78            # Go2 (mobile manipulator) distance from the centre
PEDESTAL_TOP = 0.52        # Panda base mounted above the table so it reaches down

# Natural "ready" joint poses so arms fold over the table instead of the bare
# model's splayed zero pose (values are the menagerie ``home`` keyframes).
PANDA_HOME = [0.0, -0.30, 0.0, -2.20, 0.0, 1.90, 0.79]
Z1_HOME = [0.0, 0.90, -1.10, -0.30, 0.0, 0.0]
GO2_STANCE = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}


def default_menagerie_dir() -> Path:
    return PROJECT_ROOT / "third_party" / "mujoco_menagerie"


@dataclass
class ArmHandles:
    """Per-robot kinematic handles for Stage-2 inverse-kinematics grasping."""

    ee_body: str  # end-effector body name (Panda hand / Z1 last link)
    joint_names: list[str] = field(default_factory=list)  # actuated arm hinges, base->tip
    home_qpos: list[float] = field(default_factory=list)  # neutral pose to reach from


@dataclass
class SceneInfo:
    """Lookup tables the skills need to drive the scene at runtime."""

    location_xy: dict[str, tuple[float, float]]
    robot_location: dict[str, str]
    object_bodies: list[str] = field(default_factory=list)
    drawer_object: str | None = None
    drawer_joint: str | None = None
    drawer_actuator: str | None = None
    drawer_open_pos: float = 0.20
    arms: dict[str, ArmHandles] = field(default_factory=dict)
    # Runtime grasp state (object -> robot currently holding it); used by IK skills.
    held: dict[str, str] = field(default_factory=dict)
    surface_z: float = 0.0  # table-top height parts rest at when placed
    object_half_z: dict[str, float] = field(default_factory=dict)  # half-height per part
    # Grasp weld equality names, keyed (robot_id, object) -> equality name; a held
    # part is attached to its gripper by activating the matching weld.
    grasp_welds: dict[tuple[str, str], str] = field(default_factory=dict)
    ee_prefix: dict[str, str] = field(default_factory=dict)  # robot_id -> manip name prefix


def _layout_locations(locations: list[str]) -> dict[str, tuple[float, float]]:
    """Spread the named locations as spots on the table top (a small circle),
    so parts placed at a location land on the table within the arms' reach."""
    coords: dict[str, tuple[float, float]] = {}
    n = max(len(locations), 1)
    for i, loc in enumerate(locations):
        angle = 2.0 * math.pi * i / n
        coords[loc] = (LOC_RING * math.cos(angle), LOC_RING * math.sin(angle))
    return coords


def _atoms(facts: list[str]) -> list[tuple[str, list[str]]]:
    parsed: list[tuple[str, list[str]]] = []
    for fact in facts:
        if "(" not in fact or not fact.endswith(")"):
            continue
        head, _, rest = fact.partition("(")
        args = [a.strip() for a in rest[:-1].split(",") if a.strip()]
        parsed.append((head, args))
    return parsed


def _robot_model_for(robot_id: str, robot_type: str) -> list[str]:
    key = f"{robot_id} {robot_type}".lower()
    if "go2" in key or "z1" in key or "mobile" in key:
        return [GO2_XML, Z1_XML]
    return [PANDA_XML]


# --- visual assets ----------------------------------------------------------

def _add_assets(spec: mujoco.MjSpec) -> None:
    """Register textures + materials for a workshop look (checker floor, wood
    table, brushed-metal and painted parts) and bump render quality."""
    # A soft studio backdrop instead of a flat blue void.
    spec.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        width=256, height=256, rgb1=[0.62, 0.68, 0.78], rgb2=[0.86, 0.89, 0.93],
    )
    # Textured concrete-ish floor.
    spec.add_texture(
        name="floortex", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        width=512, height=512, rgb1=[0.33, 0.35, 0.39], rgb2=[0.40, 0.42, 0.47],
        mark=mujoco.mjtMark.mjMARK_EDGE, markrgb=[0.25, 0.27, 0.30],
    )
    floor = spec.add_material(name="floor", texrepeat=[12, 12], reflectance=0.10, shininess=0.1)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floortex"

    def mat(name, rgba, *, spec_=0.3, shin=0.3, refl=0.0, metallic=0.0, rough=0.6):
        m = spec.add_material(name=name, rgba=rgba, specular=spec_, shininess=shin,
                              reflectance=refl, metallic=metallic, roughness=rough)
        return m

    mat("wood", [0.52, 0.36, 0.22, 1], spec_=0.15, shin=0.25, refl=0.02)
    mat("worktop", [0.30, 0.33, 0.37, 1], spec_=0.4, shin=0.5, refl=0.08)
    mat("steel", [0.66, 0.68, 0.72, 1], spec_=0.9, shin=0.9, refl=0.35, metallic=1.0, rough=0.25)
    mat("dark_steel", [0.28, 0.30, 0.34, 1], spec_=0.8, shin=0.8, refl=0.25, metallic=1.0, rough=0.35)
    mat("brass", [0.80, 0.62, 0.24, 1], spec_=0.9, shin=0.9, refl=0.35, metallic=1.0, rough=0.3)
    mat("cabinet", [0.72, 0.74, 0.78, 1], spec_=0.5, shin=0.5, refl=0.12, metallic=0.6, rough=0.4)
    mat("red_plastic", [0.82, 0.20, 0.18, 1], spec_=0.5, shin=0.6)
    mat("blue_plastic", [0.16, 0.42, 0.80, 1], spec_=0.5, shin=0.6)
    mat("green_plastic", [0.20, 0.62, 0.32, 1], spec_=0.5, shin=0.6)
    mat("yellow_plastic", [0.90, 0.72, 0.14, 1], spec_=0.5, shin=0.6)
    mat("black_rubber", [0.09, 0.09, 0.10, 1], spec_=0.2, shin=0.2, rough=0.9)
    mat("tray_mat", [0.20, 0.24, 0.30, 1], spec_=0.4, shin=0.4, refl=0.05)


def _add_lights(world) -> None:
    """A key light (casts shadow) plus soft fills for even, believable shading."""
    world.add_light(
        pos=[1.2, 1.0, 3.2], dir=[-0.35, -0.3, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.75, 0.75, 0.73], specular=[0.35, 0.35, 0.35], castshadow=True,
    )
    world.add_light(
        pos=[-1.4, -1.0, 2.6], dir=[0.35, 0.28, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.35, 0.37, 0.42], specular=[0.1, 0.1, 0.1], castshadow=False,
    )
    world.add_light(
        pos=[0.0, 1.6, 2.4], dir=[0.0, -0.4, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.28, 0.28, 0.30], specular=[0.0, 0.0, 0.0], castshadow=False,
    )


def build_scene(scenario: Scenario, menagerie_dir: Path | None = None) -> tuple[mujoco.MjModel, SceneInfo]:
    """Compose the scenario into a compiled :class:`mujoco.MjModel`."""
    men = Path(menagerie_dir) if menagerie_dir is not None else default_menagerie_dir()
    if not (men / GO2_XML).exists():
        raise FileNotFoundError(
            f"mujoco_menagerie not found under {men}. Clone it with:\n"
            "  git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie"
        )

    location_xy = _layout_locations(list(scenario.locations))
    robot_location = {
        args[0]: args[1]
        for head, args in _atoms(list(scenario.initial_state))
        if head == "robot_at" and len(args) == 2
    }

    spec = mujoco.MjSpec()
    spec.modelname = scenario.task_id
    # Elliptic cone + higher impratio match the go2 child options, which both
    # silences the attach conflicts and gives steadier contacts for grasped parts.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0
    # Bigger offscreen framebuffer + crisper shadows for high-res captures.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1440
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8

    _add_assets(spec)
    world = spec.worldbody
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[8, 8, 0.1], material="floor",
    )
    _add_lights(world)
    _add_table(world)

    # Attach each robot on a ring around the table, facing inward, fixed-base. The
    # *manipulator* (the Z1 for go2_z1, the Panda for a franka) is the last-attached
    # model, so its prefix is what the IK skills reach with.
    manip_prefix: dict[str, str] = {}
    n_robots = max(len(scenario.robots), 1)
    for r, robot in enumerate(scenario.robots):
        angle = 2.0 * math.pi * r / n_robots + math.pi / 4.0
        yaw = angle + math.pi  # face the table centre
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        quat = [cy, 0.0, 0.0, sy]
        fwd = (math.cos(yaw), math.sin(yaw))  # unit "facing" direction in world xy
        models = _robot_model_for(robot.id, robot.type)
        is_mobile = GO2_XML in models
        ring = GO2_RING if is_mobile else FRANKA_RING
        rx, ry = ring * math.cos(angle), ring * math.sin(angle)
        for offset, rel in enumerate(models):
            child = mujoco.MjSpec.from_file(str(men / rel))
            if rel == GO2_XML:
                # The Go2 ships with a floating trunk; with no locomotion controller
                # it just collapses. Weld it to a fixed base so it stands still.
                _make_fixed_base(child)
                frame = world.add_frame(pos=[rx, ry, 0.0], quat=quat)
            elif rel == Z1_XML:
                # Seat the Z1 on the Go2's back: lift to trunk height and nudge it
                # slightly rearward along the robot's facing axis so it rides the
                # trunk rather than floating in front of the dog.
                frame = world.add_frame(pos=[rx - 0.10 * fwd[0], ry - 0.10 * fwd[1], 0.44], quat=quat)
            else:
                # Bench arm (Panda): stand it on a pedestal at bench height so its
                # gripper reaches onto the table instead of straining up from the floor.
                _add_pedestal(world, f"{robot.id}__pedestal", rx, ry)
                frame = world.add_frame(pos=[rx, ry, PEDESTAL_TOP], quat=quat)
            spec.attach(child, prefix=f"{robot.id}_{offset}__", frame=frame)
        manip_prefix[robot.id] = f"{robot.id}_{len(models) - 1}__"

    info = SceneInfo(location_xy=location_xy, robot_location=robot_location, surface_z=TABLE_TOP_Z)
    _add_objects(spec, scenario, location_xy, info)
    _add_grasp_welds(spec, scenario, manip_prefix, info)
    _add_drawer_actuator(spec, info)

    with warnings.catch_warnings():
        # Attaching heterogeneous menagerie models yields benign option-merge notes.
        warnings.filterwarnings("ignore", message="Attach conflict", category=UserWarning)
        model = spec.compile()

    for robot in scenario.robots:
        info.ee_prefix[robot.id] = manip_prefix[robot.id]
        _pose_manipulator(model, manip_prefix[robot.id])
        handles = _arm_handles(model, manip_prefix[robot.id])
        if handles is not None:
            info.arms[robot.id] = handles
        _stand_quadruped(model, robot.id)
    return model, info


def _add_table(world) -> None:
    """A solid work table: a slab top on four legs, sized so its surface sits at
    ``TABLE_TOP_Z`` and its centre is within every arm's reach."""
    top_z = TABLE_TOP_Z - TABLE_TOP_HALF_Z
    top = world.add_body(name="worktable", pos=[0.0, 0.0, top_z])
    top.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[TABLE_HALF, TABLE_HALF, TABLE_TOP_HALF_Z], material="worktop",
    )
    # Wooden apron just under the top for a workbench look.
    top.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -TABLE_TOP_HALF_Z - 0.03],
        size=[TABLE_HALF - 0.03, TABLE_HALF - 0.03, 0.03], material="wood",
    )
    leg_h = (TABLE_TOP_Z - 2 * TABLE_TOP_HALF_Z - 0.06) / 2.0
    inset = TABLE_HALF - 0.08
    for sx in (-1, 1):
        for sy in (-1, 1):
            world.add_body(name=f"tableleg_{sx}_{sy}", pos=[sx * inset, sy * inset, leg_h]).add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.035, leg_h], material="dark_steel",
            )


def _add_pedestal(world, name: str, x: float, y: float) -> None:
    """A rigid pedestal that raises a bench arm to table height."""
    col_h = PEDESTAL_TOP / 2.0
    world.add_body(name=name, pos=[x, y, col_h]).add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.085, 0.085, col_h], material="dark_steel",
    )
    world.add_body(name=f"{name}_foot", pos=[x, y, 0.012]).add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.14, 0.14, 0.012], material="cabinet",
    )


def _ee_name_from_spec(spec: mujoco.MjSpec, prefix: str) -> str | None:
    """Best end-effector body name for a manip prefix, read off the *spec* before
    compile (mirrors :func:`_arm_handles`): last non-finger/gripper body."""
    names = [b.name for b in spec.bodies if b.name and b.name.startswith(prefix)]
    if not names:
        return None
    return next((b for b in reversed(names) if "finger" not in b and "gripper" not in b.lower()), names[-1])


def _add_grasp_welds(
    spec: mujoco.MjSpec,
    scenario: Scenario,
    manip_prefix: dict[str, str],
    info: SceneInfo,
) -> None:
    """Pre-declare an (inactive) weld equality between every arm's end-effector and
    every manipulable part. At grasp time the backend activates the matching weld,
    rigidly attaching the part to the gripper so it is carried and collides with
    the world instead of teleporting."""
    for robot in scenario.robots:
        ee = _ee_name_from_spec(spec, manip_prefix[robot.id])
        if ee is None:
            continue
        for obj in info.object_bodies:
            eq = spec.add_equality()
            eq.type = mujoco.mjtEq.mjEQ_WELD
            eq.objtype = mujoco.mjtObj.mjOBJ_BODY
            eq.name1 = ee
            eq.name2 = obj
            eq.active = False
            eq.name = f"grasp__{robot.id}__{obj}"
            eq.solref = [0.02, 1.0]
            info.grasp_welds[(robot.id, obj)] = eq.name


def _make_fixed_base(child: mujoco.MjSpec) -> None:
    """Remove a model's free (floating) base joint and any now-mismatched keyframes
    so the body is rigidly attached at its frame instead of falling under gravity."""
    for joint in list(child.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            child.delete(joint)
    for key in list(child.keys):
        child.delete(key)


def _stand_quadruped(model: mujoco.MjModel, robot_id: str) -> None:
    """Pose any Go2 legs in a natural stance (the bare robot model defaults to a
    splayed zero pose). Writes ``qpos0`` so a fresh MjData starts standing."""
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith(f"{robot_id}_0__") or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        for key, angle in GO2_STANCE.items():
            if key in name:
                model.qpos0[model.jnt_qposadr[j]] = angle
                break


def _pose_manipulator(model: mujoco.MjModel, prefix: str) -> None:
    """Write a natural ready pose into ``qpos0`` for the arm under ``prefix`` so a
    fresh MjData starts folded over the table (Panda: 7 hinges, Z1: 6 hinges)."""
    hinges: list[int] = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith(prefix) or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if "finger" in name or "gripper" in name.lower():
            continue
        hinges.append(j)
    home = {7: PANDA_HOME, 6: Z1_HOME}.get(len(hinges))
    if home is None:
        return
    for j, angle in zip(hinges, home, strict=False):
        model.qpos0[model.jnt_qposadr[j]] = angle


def _arm_handles(model: mujoco.MjModel, prefix: str) -> ArmHandles | None:
    """Discover a manipulator's end-effector body and actuated arm hinges by
    name prefix (skips gripper/finger joints), so IK can reach with that arm."""
    bodies = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]
    bodies = [b for b in bodies if b and b.startswith(prefix)]
    if not bodies:
        return None
    ee = next((b for b in reversed(bodies) if "finger" not in b and "gripper" not in b.lower()), bodies[-1])

    joints: list[str] = []
    home: list[float] = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or not name.startswith(prefix):
            continue
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or "finger" in name or "gripper" in name.lower():
            continue
        joints.append(name)
        home.append(float(model.qpos0[model.jnt_qposadr[j]]))
    if not joints:
        return None
    return ArmHandles(ee_body=ee, joint_names=joints, home_qpos=home)


def _object_initial_location(scenario: Scenario, obj: str) -> str | None:
    """Find a location L from any initial fact ``pred(obj, L)``."""
    for _, args in _atoms(list(scenario.initial_state)):
        if len(args) == 2 and args[0] == obj and args[1] in scenario.locations:
            return args[1]
    return None


def _add_objects(
    spec: mujoco.MjSpec,
    scenario: Scenario,
    location_xy: dict[str, tuple[float, float]],
    info: SceneInfo,
) -> None:
    world = spec.worldbody
    # The drawer is whatever object appears in a drawer_closed/open fact.
    drawer = next(
        (args[0] for head, args in _atoms(list(scenario.initial_state))
         if head in ("drawer_closed", "drawer_open") and args),
        None,
    )

    part_materials = [
        "red_plastic", "blue_plastic", "green_plastic",
        "yellow_plastic", "brass", "steel",
    ]

    top = info.surface_z
    cab_xy = location_xy.get(drawer, (0.0, 0.30)) if drawer else (0.0, 0.30)

    # Every movable part is laid out on a shallow arc across the *front* of the
    # bench (well clear of the cabinet at the back), evenly spaced so nothing
    # overlaps and every part rests flat and stays in view. Tools staged in the
    # drawer thus read as parts set out ready for the job.
    movable = [o for o in scenario.objects if o != drawer]
    n_mov = len(movable)

    def _bench_xy(idx: int) -> tuple[float, float]:
        t = idx / max(n_mov - 1, 1)          # 0..1 left -> right
        ax = -0.33 + 0.66 * t                # x across the bench front, well spaced
        ay = -0.14 + 0.05 * math.sin(math.pi * t)  # gentle bow toward the viewer
        return ax, ay

    for i, obj in enumerate(scenario.objects):
        if obj == drawer:
            _add_cabinet(world, obj, cab_xy, top, info)
            continue
        material = part_materials[i % len(part_materials)]
        x, y = _bench_xy(movable.index(obj))
        # Manipulable part: a free body shaped to resemble the named object.
        half_z = _add_part_geoms(world, obj, material, x, y, top)
        info.object_bodies.append(obj)
        info.object_half_z[obj] = half_z


def _add_cabinet(world, name: str, xy: tuple[float, float], top: float, info: SceneInfo) -> None:
    """A small parts cabinet on the table: a static carcass plus a sliding drawer
    (the named body) with a front panel and handle, so open/close is a real joint
    motion. The drawer body keeps ``name`` so object detection still finds it."""
    cx, cy = xy
    cy += 0.02  # nudge to the back so loose parts sit in front of it
    W, D, H = 0.13, 0.16, 0.11  # half-extents of the carcass
    base_z = top + H
    carcass = world.add_body(name=f"{name}_cabinet", pos=[cx, cy, base_z])
    wall = 0.012
    # shell: back + two sides + top + bottom (open front)
    carcass.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, D - wall, 0],
                     size=[W, wall, H], material="cabinet")
    carcass.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-W + wall, 0, 0],
                     size=[wall, D, H], material="cabinet")
    carcass.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[W - wall, 0, 0],
                     size=[wall, D, H], material="cabinet")
    carcass.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, H - wall],
                     size=[W, D, wall], material="cabinet")
    carcass.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -H + wall],
                     size=[W, D, wall], material="cabinet")

    drawer = world.add_body(name=name, pos=[cx, cy, base_z])
    joint = drawer.add_joint(
        name=f"{name}__slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=[0, -1, 0], range=[0.0, info.drawer_open_pos],
    )
    # drawer box (floor + sides) that rides out on the slide, plus a front face + handle
    dw, dd, dh = W - 2 * wall, D - wall, H - wall
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -dh + wall],
                    size=[dw, dd, wall], material="dark_steel")
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-dw + wall, 0, 0],
                    size=[wall, dd, dh], material="dark_steel")
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[dw - wall, 0, 0],
                    size=[wall, dd, dh], material="dark_steel")
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -dd, 0],
                    size=[W, wall, H], material="dark_steel")  # front face (contrasts carcass)
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, -dd - 0.006, 0.04],
                    size=[W - 0.02, 0.004, 0.02], material="red_plastic")  # label strip
    drawer.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, -dd - 0.03, -0.03],
                    quat=[0.7071, 0.7071, 0, 0], size=[0.007, 0.055], material="steel")  # bar handle
    info.drawer_object = name
    info.drawer_joint = joint.name


def _add_part_geoms(world, name: str, material: str, x: float, y: float, top: float) -> float:
    """Build a free body whose primitive geoms evoke the named part (gear=toothed
    disc with hub + spokes, shaft/screwdriver=rod, tray/plate=walled slab, ...).
    Returns its half-height so it can be rested on the table surface."""
    n = name.lower()

    if "gear" in n and not any(w in n for w in ("base", "box", "tray", "plate")):
        half_z = 0.02
        body = world.add_body(name=name, pos=[x, y, top + half_z + 0.02])
        body.add_freejoint()
        r = 0.055
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, half_z], material="brass")
        for k in range(12):  # teeth around the rim
            a = 2.0 * math.pi * k / 12.0
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.008, half_z],
                          pos=[(r + 0.008) * math.cos(a), (r + 0.008) * math.sin(a), 0.0],
                          quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)], material="brass")
        for k in range(4):  # lightening holes read as a machined gear
            a = math.pi / 4 + 2.0 * math.pi * k / 4.0
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.011, half_z + 0.001],
                          pos=[0.032 * math.cos(a), 0.032 * math.sin(a), 0.0], material="dark_steel")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, half_z + 0.004],
                      material="dark_steel")  # hub
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.009, half_z + 0.006],
                      material="steel")  # bore collar
        return half_z

    if "shaft" in n:
        # A precision shaft lying flat on the bench (stable on its side).
        half_z = 0.012  # rest height = rod radius
        qy = [0.7071, 0.7071, 0.0, 0.0]  # lay the cylinder axis along +y (front-back)
        body = world.add_body(name=name, pos=[x, y, top + half_z + 0.01])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.011, 0.08],
                      quat=qy, material="steel")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.016, 0.006],
                      pos=[0, -0.074, 0], quat=qy, material="dark_steel")  # collar/shoulder
        return half_z

    if "driver" in n or "screw" in n or "wrench" in n:  # screwdriver / torque_driver
        # A screwdriver resting flat (long axis front-back), handle at one end,
        # shank + tip at the other.
        half_z = 0.02  # rest height = handle radius
        qy = [0.7071, 0.7071, 0.0, 0.0]  # lay everything along +y
        body = world.add_body(name=name, pos=[x, y, top + half_z + 0.01])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, 0.045],
                      pos=[0, 0.045, 0], quat=qy, material="red_plastic")  # ergonomic handle
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.021, 0.01],
                      pos=[0, 0.002, 0], quat=qy, material="black_rubber")  # bolster collar
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.006, 0.05],
                      pos=[0, -0.05, 0], quat=qy, material="steel")  # steel shank
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.006, 0.002],
                      pos=[0, -0.1, 0], material="dark_steel")  # flat tip
        return half_z

    if "cable" in n:
        half_z = 0.014
        body = world.add_body(name=name, pos=[x, y, top + half_z + 0.02])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0.07], material="black_rubber")
        return half_z

    if any(w in n for w in ("tray", "plate")):  # tray / plate: walled slab
        half_z = 0.02
        body = world.add_body(name=name, pos=[x, y, top + half_z + 0.02])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, -half_z + 0.006],
                      size=[0.075, 0.055, 0.006], material="tray_mat")  # floor
        for px, py, sxx, syy in [(0, 0.055, 0.075, 0.006), (0, -0.055, 0.075, 0.006),
                                 (0.075, 0, 0.006, 0.055), (-0.075, 0, 0.006, 0.055)]:
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[px, py, 0],
                          size=[sxx, syy, half_z], material="tray_mat")  # rim walls
        return half_z

    if "base" in n or "box" in n:  # gearbase / gearbox: a machined block with a boss
        half_z = 0.03
        body = world.add_body(name=name, pos=[x, y, top + half_z + 0.02])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.045, half_z], material="dark_steel")
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.02, half_z + 0.01],
                      pos=[0, 0, 0], material="steel")  # central bearing boss
        for sx in (-1, 1):
            for sy in (-1, 1):  # mounting bolt heads
                body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.006, half_z + 0.003],
                              pos=[sx * 0.045, sy * 0.032, 0], material="brass")
        return half_z

    # default block (bracket / module / target / sensor / ...)
    half_z = 0.03
    body = world.add_body(name=name, pos=[x, y, top + half_z + 0.02])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.035, half_z], material=material)
    return half_z


def _add_drawer_actuator(spec: mujoco.MjSpec, info: SceneInfo) -> None:
    """Give the drawer slide a position servo so the mobile manipulator can pull it
    open/closed under control instead of the drawer snapping by itself."""
    if not info.drawer_joint:
        return
    a = spec.add_actuator()
    a.name = f"{info.drawer_object}__act"
    a.target = info.drawer_joint
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.gainprm = [600.0] + [0.0] * 9
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.biasprm = [0.0, -600.0, -60.0] + [0.0] * 7
    a.ctrlrange = [0.0, info.drawer_open_pos]
    a.forcerange = [-80.0, 80.0]
    info.drawer_actuator = a.name
