---
name: inter-robot-synchronization
description: When and how to add synchronization Conditions between robots
tags: sync
applies_to: "*"
---
Use `synchronization` only for a genuine *cross-robot* wait: robot B needs a predicate that a *different*
robot A produces.

- Add an entry `{condition: "<predicate(args)>", producer: "A", consumer: "B"}`, and place a matching
  `Condition` node in B's tree immediately before the action that consumes the predicate.
- The producer A's tree must actually run an action whose `effects.add` creates that exact predicate,
  with exactly matching arguments (a nearby or generic argument will not satisfy it).
- Before a robot waits on a downstream condition, it must first produce anything downstream robots need
  from it - otherwise the two robots wait on each other and deadlock.
- Do NOT add synchronization for same-robot ordering or for predicates already true in `initial_state`.

Example: `franka2` needs `gearbase_stable(gearbase)` before `mount_gear`; `franka1` produces it via
`stabilize_gearbase`. Add `{condition: "gearbase_stable(gearbase)", producer: "franka1", consumer:
"franka2"}` and put that Condition before `mount_gear` in `franka2`'s tree.
