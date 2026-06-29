"""Prompts for the baseline generators.

These deliberately *omit* the proposed method's two key assists - the back-chaining
``_METHOD`` block and the synchronization rules - so each baseline is faithful to its
source paper's level of capability. They reuse :func:`prompts._scenario_context` so
the scenario is presented identically; only the instructions differ.
"""

from __future__ import annotations

import json
from typing import Any

from ..domain import Scenario
from ..prompts import _SCHEMA, _scenario_context

# --------------------------------------------------------------------------- #
# Flat baseline (LLM-MARS-style): one generic call, BTs per robot, no synchronization
# machinery, no back-chaining method, no self-correction.
# --------------------------------------------------------------------------- #

_FLAT_RULES = """Rules:
1. Use only the robot, object, location, action, and predicate names supplied above.
2. Output one behavior tree per robot; each is a Sequence of that robot's Action nodes.
3. Assign each action to a robot that has that capability.
4. BT node 'name' is the bare action/predicate name; parameters go in the parameters array.
5. Build task_graph and assignments so every BT action is one task assigned to its robot.
"""


def build_flat_prompt(scenario: Scenario) -> str:
    """Single-shot, naive multi-robot BT generation (no method, no sync guidance)."""
    return "\n".join(
        [
            "Generate behavior trees for a multi-robot team to carry out the task below.\n"
            "Return ONLY valid JSON. Do not use markdown or explanatory text.\n",
            _scenario_context(scenario, include_hints=False),
            _SCHEMA,
            _FLAT_RULES,
            'Leave "synchronization" as an empty list.\n',
        ]
    )


# --------------------------------------------------------------------------- #
# Hierarchical baseline (LLM-as-BT-Planner-style): decompose -> per-robot BTs, with a
# recursive self-correction loop, but robots are planned INDEPENDENTLY - no cross-robot
# Condition / synchronization guidance, so inter-robot waits cannot be expressed.
# --------------------------------------------------------------------------- #

_HIER_DECOMPOSE_SCHEMA = """Required output schema (decomposition only - NO behavior trees yet):
{
  "action_plan": {
    "robot_id": [
      {"action": "action_name", "parameters": ["arg"]}
    ]
  }
}
"""


def build_hier_decompose_prompt(scenario: Scenario) -> str:
    """Stage 1: decompose the instruction+goal into an ordered per-robot action list."""
    return "\n".join(
        [
            "Decompose the task below into the sequence of actions each robot performs.\n"
            "Work robot by robot: for each robot, list the ordered actions it should execute to\n"
            "achieve its share of the goal. Output only the per-robot action lists - no behavior\n"
            "trees yet. Return ONLY valid JSON, no markdown.\n",
            _scenario_context(scenario, include_hints=False),
            _HIER_DECOMPOSE_SCHEMA,
            "Rules:\n"
            "1. Use only the action and object/location names from the capability library.\n"
            "2. Each action's parameters must match its capability exactly.\n"
            "3. Order each robot's own actions so its preconditions are met by the time it acts.\n",
        ]
    )


_HIER_BT_RULES = """Encoding rules:
1. Each robot's behavior_tree is a Sequence containing exactly its actions from the
   decomposition, in the same order, with identical parameters.
2. Build task_graph and assignments so every action is one task assigned to its robot.
3. You may add a Condition node before an action to check a precondition that this SAME
   robot produced earlier; do not coordinate across robots.
4. BT node 'name' is the bare action/predicate name; parameters go in the parameters array.
5. Leave "synchronization" as an empty list - plan each robot's tree independently.
"""


def build_hier_bt_prompt(scenario: Scenario, action_plan: dict[str, Any]) -> str:
    """Stage 2: encode the per-robot decomposition into independent behavior trees."""
    return "\n".join(
        [
            "Encode the per-robot action decomposition below as one behavior tree per robot.\n"
            "Return ONLY valid JSON, no markdown.\n",
            _scenario_context(scenario, include_hints=False),
            f"Decomposition (use these exact actions, per robot, in this order):\n"
            f"{json.dumps({'action_plan': action_plan}, indent=2)}\n",
            _SCHEMA,
            _HIER_BT_RULES,
        ]
    )


def build_hier_correction_prompt(
    scenario: Scenario,
    validation_errors: list[dict[str, str]],
    simulation_summary: dict[str, Any],
    previous_plan: dict[str, Any],
) -> str:
    """Recursive self-correction: feed typed errors + sim summary back to regenerate.

    Faithful to the source method's iterative/recursive generation, but still WITHOUT
    cross-robot synchronization - so same-robot ordering/producer errors can be fixed
    while inter-robot dependencies remain unaddressable.
    """
    return (
        "The previous behavior-tree plan failed validation or simulation. Return a COMPLETE\n"
        "corrected plan as JSON. Keep what was already correct; fix only what the errors require.\n"
        "Plan each robot independently; keep \"synchronization\" empty.\n\n"
        f"{_scenario_context(scenario, include_hints=False)}\n"
        f"{_SCHEMA}\n"
        f"{_HIER_BT_RULES}\n"
        f"Previous plan that failed:\n{json.dumps(previous_plan, indent=2)}\n\n"
        f"Validator errors:\n{json.dumps(validation_errors, indent=2)}\n\n"
        f"Simulation result:\n{json.dumps(simulation_summary, indent=2)}\n\n"
        "Return only the complete corrected JSON object."
    )
