"""MuJoCo execution backend.

Runs a validated plan through the *same* symbolic tick engine (so synchronization,
deadlock and goal semantics are identical to the symbolic backend), but hooks
every executed action to a physical skill that drives a real MuJoCo scene built
from menagerie robots. With ``render=True`` you watch the generated behavior
trees play out, one synchronized round per global tick, on the actual robot
models. Headless (``render=False``) it just steps physics so the run stays
testable in CI.

Predicate-level success/goal come from the tick engine; physics adds a faithful
embodiment and a per-action physics trace. See :mod:`mujoco_skills` for the
Stage-1 (scripted-settle) vs Stage-2 (``mink`` IK grasp) fidelity ladder.

The backend does not yet read success/failure *back* from physics: goal success
is the symbolic tick's. The tick engine already exposes an ``action_oracle`` seam
(see :mod:`llm_mr_bt_planner.recovery`) so a physics failure oracle - checking
whether an action's target predicate actually holds in the scene - can drive the
execution-time retry/reassign ladder; wiring that end to end is tracked in
``docs/roadmap.md``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mujoco

from ..domain import Scenario
from ..plan import Plan
from ..simulation import simulate
from .base import ExecutionResult
from .mujoco_scene import SceneInfo, build_scene
from .mujoco_skills import resolve_skill


class MujocoExecutionBackend:
    name = "mujoco"

    def __init__(
        self,
        max_ticks: int = 80,
        menagerie_dir: str | Path | None = None,
        mjcf: str | Path | None = None,
        render: bool = False,
        settle_steps: int = 60,
        realtime: float = 0.6,
        fidelity: str = "settle",
        hold_open: bool = True,
        start_delay: float = 5.0,
    ) -> None:
        self.max_ticks = max_ticks
        self.menagerie_dir = Path(menagerie_dir) if menagerie_dir else None
        self.mjcf = Path(mjcf) if mjcf else None
        self.render = render
        self.settle_steps = settle_steps
        self.realtime = realtime  # seconds to pause per action when rendering
        self.hold_open = hold_open  # keep the viewer up after the replay (render only)
        self.start_delay = start_delay  # seconds to hold the opening shot before replay
        if fidelity not in ("settle", "ik"):
            raise ValueError("fidelity must be 'settle' (Stage 1) or 'ik' (Stage 2).")
        self.fidelity = fidelity

    def execute(self, plan: Plan, scenario: Scenario) -> ExecutionResult:
        model, info = self._build(scenario)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self._hold_pose(model, data)  # command actuators to the ready pose up front
        self._settle(model, data, None, info, steps=80)  # let objects fall onto the floor

        viewer = None
        if self.render:
            from mujoco import viewer as mj_viewer  # lazy: headless runs need no GL

            viewer = mj_viewer.launch_passive(model, data)
            self._frame_camera(viewer, model)
            self._countdown(viewer, model, data)

        # 'ik' fidelity (Stage 2): drive the arms with mink differential IK and
        # carry each grasped part kinematically (see mujoco_physics.ArmController).
        # 'settle' (Stage 1): the lighter scripted-teleport + gravity-settle path.
        controller = None
        if self.fidelity == "ik":
            try:
                from .mujoco_physics import ArmController
            except ModuleNotFoundError as exc:  # mink / qp solver not installed
                raise ModuleNotFoundError(
                    f"The 'ik' fidelity needs mink for inverse kinematics ({exc.name} is missing). "
                    'Install it with:  pip install mink "qpsolvers[daqp]"\n'
                    "Or run with --physics settle to use the lighter kinematic motion instead."
                ) from exc

            def _sync() -> None:
                if viewer is not None:
                    viewer.sync()
                    time.sleep(0.005)

            controller = ArmController(model, data, info, step_cb=_sync if viewer is not None else None)

        physics_trace: list[dict[str, Any]] = []

        def on_action(event: dict[str, Any]) -> None:
            if controller is not None:
                label = controller.execute(event)
            elif viewer is not None:
                q0 = data.qpos.copy()
                skill = resolve_skill(event.get("name"))
                skill(model, data, event, info)
                self._animate(model, data, viewer, info, q0, data.qpos.copy())
                label = skill.__name__
            else:
                skill = resolve_skill(event.get("name"))
                skill(model, data, event, info)
                self._hold_pose(model, data)  # hold the arm where the skill left it
                self._settle(model, data, None, info, steps=self.settle_steps)
                label = skill.__name__
            physics_trace.append(
                {
                    "tick": event.get("tick"),
                    "robot": event.get("robot"),
                    "action": event.get("action"),
                    "skill": label,
                    "sim_time": round(float(data.time), 3),
                }
            )

        try:
            report = simulate(plan, scenario, max_ticks=self.max_ticks, on_action=on_action)
            self._hold_pose(model, data)
            self._settle(model, data, viewer, info, steps=200)  # let everything come to rest
            if viewer is not None and self.hold_open:
                # Keep the window up after the replay so the final scene is inspectable;
                # exits when the user closes the window.
                print("[mujoco] Replay done - close the viewer window to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
        finally:
            if viewer is not None:
                viewer.close()

        return ExecutionResult(
            backend=self.name,
            success=report.success,
            goal_success=report.goal_success,
            final_state=report.final_state,
            trace=report.trace,
            errors=report.errors,
            details={
                "max_ticks": self.max_ticks,
                "fidelity": self.fidelity,
                "model": {"nbody": int(model.nbody), "njnt": int(model.njnt), "nu": int(model.nu)},
                "robots": [r.id for r in scenario.robots],
                "object_bodies": info.object_bodies,
                "physics_actions": len(physics_trace),
                "physics_trace": physics_trace,
                "sim_time": round(float(data.time), 3),
            },
        )

    # -- helpers -------------------------------------------------------------

    def _hold_pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Command every joint actuator to hold its current angle, so the posed
        arms and quadruped legs stay put while physics settles the parts instead
        of sagging toward a zero-control pose."""
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            jid = int(model.actuator_trnid[a, 0])
            data.ctrl[a] = float(data.qpos[model.jnt_qposadr[jid]])

    def _build(self, scenario: Scenario) -> tuple[mujoco.MjModel, SceneInfo]:
        if self.mjcf is not None:
            model = mujoco.MjModel.from_xml_path(str(self.mjcf))
            return model, _scene_info_from_model(model, scenario)
        return build_scene(scenario, self.menagerie_dir)

    def _frame_camera(self, viewer, model: mujoco.MjModel) -> None:
        """Pull the camera back so the whole workcell is in view from the start."""
        cam = viewer.cam
        cam.lookat[:] = [0.0, 0.0, 0.42]
        cam.distance = 2.6
        cam.azimuth = 125.0
        cam.elevation = -24.0
        viewer.sync()

    def _countdown(self, viewer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Hold the opening shot so the user can orbit/zoom before the replay runs."""
        if self.start_delay <= 0:
            return
        whole = int(self.start_delay)
        for remaining in range(whole, 0, -1):
            print(f"[mujoco] Replay starts in {remaining}s  (drag to orbit, scroll to zoom)...")
            for _ in range(int(1.0 / 0.02)):
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(0.02)

    def _settle(self, model, data: mujoco.MjData, viewer, info: SceneInfo, steps: int) -> None:
        # Step physics so free objects settle under gravity. Any grasped part (in
        # 'ik' fidelity) is re-snapped to the gripper each control step by the
        # ArmController - it is NOT held by a weld. The scene declares grasp welds
        # but leaves them inactive; activating them is tracked in docs/roadmap.md.
        for i in range(steps):
            mujoco.mj_step(model, data)
            if viewer is not None and i % 4 == 0:  # refresh while settling, not just at the end
                viewer.sync()
                time.sleep(0.004)
        if viewer is not None:
            viewer.sync()

    def _animate(self, model, data: mujoco.MjData, viewer, info: SceneInfo, q0, q1) -> None:
        """Kinematically sweep qpos from q0 to q1 over several frames so the motion
        the skill produced renders smoothly, then briefly settle physics."""
        frames = max(int(self.realtime / 0.02), 12)
        for f in range(1, frames + 1):
            if not viewer.is_running():
                return
            t = f / frames
            data.qpos[:] = (1.0 - t) * q0 + t * q1
            self._hold_pose(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.02)
        # Let grasped/placed parts respond to contact + gravity for a moment.
        self._settle(model, data, viewer, info, steps=max(self.settle_steps // 2, 10))


def _scene_info_from_model(model: mujoco.MjModel, scenario: Scenario) -> SceneInfo:
    """Best-effort SceneInfo for an externally supplied MJCF: detect object
    bodies and a drawer slide joint by name so skills still work."""
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    }
    object_bodies = [o for o in scenario.objects if o in names]
    drawer_joint = next(
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
         for j in range(model.njnt)
         if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("__slide")),
        None,
    )
    drawer_object = drawer_joint.replace("__slide", "") if drawer_joint else None
    robot_location = {
        f.split("(")[1].rstrip(")").split(",")[0].strip(): f.split(",")[1].rstrip(")").strip()
        for f in scenario.initial_state
        if f.startswith("robot_at(") and "," in f
    }
    return SceneInfo(
        location_xy={loc: (0.0, 0.0) for loc in scenario.locations},
        robot_location=robot_location,
        object_bodies=object_bodies,
        drawer_object=drawer_object,
        drawer_joint=drawer_joint,
    )
