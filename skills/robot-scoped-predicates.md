---
name: robot-scoped-predicates
description: Predicates bound to a specific robot must be produced by that robot
tags: manipulation, sync
applies_to: "*"
---
A predicate whose first argument is a robot - e.g. `holding(franka2, gear)` - is *robot-scoped*: it can
only be created by that same robot's own action.

- `holding(franka2, gear)` is added by `franka2`'s own `pick_gear(gear)`. Another robot placing the gear
  nearby does NOT make `franka2` hold it.
- Therefore, if an action assigned to `franka2` requires `holding(franka2, gear)`, `franka2`'s behavior
  tree must contain the pick action that produces it, earlier in the same tree.
- Do not add a synchronization Condition for a robot-scoped predicate of robot R into R's own tree - R
  produces it itself; sync Conditions are only for predicates produced by a *different* robot.

Worked pattern (substitute the scenario's real names): goal needs `mounted(gear)`; `mount_gear(gear)`
requires `holding(franka2, gear)`; that is added only by `franka2`'s `pick_gear(gear)`. So `franka2`'s
tree is `... -> pick_gear(gear) -> mount_gear(gear) ...`, both assigned to `franka2`.
