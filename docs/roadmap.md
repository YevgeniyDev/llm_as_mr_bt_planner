# Roadmap — future implementations

This is the forward-looking plan: stages that are designed-for but not yet built. Each item names the seam
it plugs into so the next contributor can start from the existing code rather than a blank page. Items are
roughly ordered by how much they unlock relative to effort.

## MuJoCo physics failure oracle (wire the recovery ladder to physics)

The execution-time recovery ladder (`recovery.py`) is complete and driven today by an injected/stochastic
oracle. The remaining rung is a **real physics oracle**: after a physical skill runs, check whether the
action's intended predicate actually holds in the MuJoCo scene (object at target, part carried, drawer
open, …) and return `Status.SUCCESS`/`FAILURE` accordingly.

- Seam: `simulation.simulate(action_oracle=…)` already exists and `SymbolicExecutionBackend` already
  accepts `recovery=`. Add a `recovery=` parameter to `MujocoExecutionBackend` (mirroring the symbolic one)
  and a `make_physics_oracle(model, data, info)` in the MuJoCo backend that grounds the symbolic predicate
  to a geometric check.
- Ordering caveat: the oracle is invoked *before* `on_action`, so the physics path must run the skill
  *inside* the oracle (and pass `on_action=None`) to avoid executing the skill twice.
- Re-running whole episodes for retry/reassign re-animates physics from scratch; for rendered runs, consider
  a cheaper "resume from the failing tick" path or accept the re-animation cost headless.

## Contact/friction-based grasping (activate the dormant scaffolding)

Grasping is a **kinematic snap** today: `ArmController` re-snaps each held part to the gripper every control
step. The scene already **declares but leaves inactive** the pieces needed for real grasping:

- `mujoco_scene._add_grasp_welds` pre-declares an `eq_active = False` weld between each end-effector and each
  part (stored in `SceneInfo.grasp_welds`) — activate the matching weld on grasp, deactivate on release.
- `mujoco_scene._add_drawer_actuator` adds a position servo (`SceneInfo.drawer_actuator`) whose `ctrl` is
  never driven — drive it instead of ramping the drawer `qpos` kinematically.
- Then close the gripper against the part with real contacts (friction) instead of snapping.

## Retire the legacy IK module

`execution/mujoco_ik.py` (hand-rolled damped-least-squares IK) is superseded by the `mink`
`ArmController` and is no longer imported at runtime. Remove or archive it once nothing references it, and
drop the corresponding note from `docs/architecture.md`.

## Recovery in the experiment runner + stitched viz trace

- `experiments/run_experiment` measures planning-time metrics only. Add an option to run the execution-time
  recovery ladder per trial (with a stochastic oracle) and report recovery rate / mean episodes as new
  metric columns.
- `RecoveryResult.report` is the *final* episode's trace; the recovery `log` is the ladder timeline. For the
  HTML visualization, optionally stitch a cumulative cross-episode trace so a run with retries/reassignments
  reads as one timeline.

## Real-robot execution (ROS)

`RosExecutionBackend.execute` still raises with wiring guidance. Standing it up needs: an action/skill
server per capability (the leaf `name` selects it, `parameters` are the goal), a blackboard of symbolic
predicates updated by perception/condition monitors, and loading the exported BehaviorTree.CPP XML
(`export_behaviortree_cpp_xml`) into py_trees_ros / BehaviorTree.CPP. The recovery ladder's reassignment
would then operate on live task allocation rather than a re-simulated plan.

## Authored-in-Markdown core prompt (optional)

The load-bearing `_RULES`/`_METHOD` deliberately stay in `prompts.py` (always-on, reproducible). If a future
need arises to author the core guidance in Markdown too, add a specially-named always-selected
`skills/_method.md` that `build_prompt` prefers when present and falls back to the constant otherwise — but
keep the constant as the default so a missing/edited file can't silently weaken the pure-mode baseline.
