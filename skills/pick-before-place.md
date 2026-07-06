---
name: pick-before-place
description: Grasp/pick actions must precede the place/use action that consumes the hold
tags: manipulation
applies_to: pick_gear, pick_tool, pick_tray, place_tray, place_tool, return_tool, mount_gear, fasten_screw
---
Manipulation capabilities come in producer/consumer pairs. The action that *uses* a held object has a
`holding(robot, object)` precondition that is only created by that robot's own *pick/grasp* action.

- Put the pick/grasp before the place/mount/fasten/return action in the same robot's Sequence.
- A tool used and then returned (`pick_tool -> ... -> return_tool`) still needs the `pick_tool` first;
  do not assume the tool is already held.
- Match the object argument exactly between the pick and the consuming action (`pick_gear(gear)` then
  `mount_gear(gear)`, not a different part).
- If two robots both `pick_tool`, each needs its own pick action and its own `holding(robot, tool)`;
  these are distinct tasks, not shared.
