---
name: back-chaining-from-goals
description: Derive the action set by regressing goals to their producers
tags: planning, method
applies_to: "*"
---
Work backwards from `goal_state`, not forwards from the instruction.

1. For each goal predicate not already in `initial_state`, find the capability whose `effects.add`
   creates it, add that action, and assign it to a robot that owns the capability.
2. For every action you add, take each of its `preconditions` in turn. A precondition is satisfied only
   if it is (a) already in `initial_state`, (b) added earlier by the *same* robot's tree, or
   (c) produced by *another* robot and waited on with a synchronization Condition placed before it.
3. If a precondition is none of these, add the action that produces it and recurse on *its*
   preconditions. Stop only when every chain bottoms out at `initial_state`.

The single most common failure is omitting an intermediate producer action, so re-walk every action's
preconditions once more before answering.
