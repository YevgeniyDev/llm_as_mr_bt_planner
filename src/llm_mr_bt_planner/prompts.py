"""Prompt construction and JSON extraction for direct LLM BT generation."""

from __future__ import annotations

import json
import re
from typing import Any

from .domain import Scenario
from .simulation import SimulationReport

SYSTEM_PROMPT = (
    "You are a multi-robot Behavior Tree planner. Produce one complete synchronized Behavior Tree "
    "for every declared robot as strict JSON. You—not a downstream compiler—own the control flow, "
    "conditions, waits, resource operations, action ordering, node IDs, and task IDs."
)

_OUTPUT_SCHEMA = """Required output schema (schema_version must be \"2.0\"):
{
  "schema_version": "2.0",
  "mission_id": "the exact scenario task_id",
  "behavior_trees": {
    "robot_id": {
      "id": "globally_unique_node_id",
      "type": "Sequence",
      "source": "llm",
      "children": [
        {
          "id": "globally_unique_wait_id",
          "type": "WaitFor",
          "name": "predicate_name",
          "parameters": ["grounded", "arguments"],
          "timeout_ticks": 80,
          "source": "llm"
        },
        {
          "id": "globally_unique_resource_id",
          "type": "AcquireResource",
          "name": "declared_resource_id",
          "parameters": [],
          "timeout_ticks": 40,
          "source": "llm"
        },
        {
          "id": "globally_unique_action_node_id",
          "type": "Action",
          "task_id": "globally_unique_task_id",
          "name": "declared_capability_name",
          "parameters": ["grounded", "arguments"],
          "source": "llm"
        },
        {
          "id": "globally_unique_release_id",
          "type": "ReleaseResource",
          "name": "declared_resource_id",
          "parameters": [],
          "source": "llm"
        }
      ]
    }
  }
}

Allowed composite nodes:
- Sequence, ReactiveSequence, and Fallback: id, type, source, children.
- Parallel and ParallelAll: id, type, source, children, success_threshold.

Allowed leaf nodes:
- Action: id, type, source, task_id, name, parameters.
- Condition: id, type, source, name, parameters.
- WaitFor: id, type, source, name, parameters, timeout_ticks.
- AcquireResource: id, type, source, name, parameters, timeout_ticks.
- ReleaseResource: id, type, source, name, parameters.
"""

_RULES = """Planning and Behavior Tree rules:
1. Return a complete behavior_trees entry for every scenario robot and no unknown robot.
2. Use only the allowed node types, declared identifiers, predicates, resources, and robot capabilities.
3. Every node must have a globally unique id and source \"llm\". Every Action also needs a globally unique task_id.
4. The capability library is authoritative for Action parameter types, preconditions, effects, resources,
   duration, and timeout. Do not copy those contracts into Action nodes or invent overrides.
5. You own all BT structure. No downstream component will insert, reorder, or repair nodes.
6. Put each Action only in the tree of a robot that declares that capability.
7. Work backwards from every goal and include every producer Action needed to reach it.
8. Before an Action that consumes a predicate asynchronously produced by another robot, include an exact
   WaitFor for that grounded predicate. WaitFor returns RUNNING until true and must have a finite timeout.
9. Condition returns FAILURE immediately when false; it is a branch/guard, not cross-robot synchronization.
10. Explicitly AcquireResource before every Action requiring a declared exclusive resource and ReleaseResource
    after the protected actions. Do not hold a resource while waiting for another robot.
11. A Sequence succeeds in order; Fallback selects the first successful child; Parallel/ParallelAll execute
    children concurrently and require a valid success_threshold.
12. Ensure every possible successful completion releases acquired resources and reaches every declared goal.
13. Do not invent physical observations, successful outcomes, capabilities, recovery behavior, or motor commands.
14. Return only the complete JSON object, without markdown or explanation.
"""


def build_prompt(scenario: Scenario) -> str:
    """Build the initial direct full-BT generation request."""
    return "\n".join(
        [
            "Construct the complete synchronized multi-robot Behavior Trees for this mission.",
            _scenario_context(scenario),
            _OUTPUT_SCHEMA,
            _RULES,
        ]
    )


def build_correction_prompt(
    scenario: Scenario,
    validation_errors: list[dict[str, str]],
    simulation: SimulationReport,
    previous_plan: dict[str, Any],
) -> str:
    """Ask the model to replace its complete BT using typed diagnostics."""
    diagnostics = {
        "validation_errors": validation_errors,
        "simulation": {
            "success": simulation.success,
            "goal_success": simulation.goal_success,
            "errors": simulation.errors,
            "final_state": simulation.final_state,
            "trace_tail": simulation.trace[-8:],
        },
    }
    return (
        f"{build_prompt(scenario)}\n\n"
        "Your previous complete Behavior Tree failed deterministic validation or contract simulation. "
        "Return a complete replacement behavior_trees object. Correct the BT itself: its composites, guards, "
        "WaitFor synchronization, explicit resource operations, action ordering, parameters, IDs, and timeouts. "
        "No downstream compiler will add or repair any of those nodes.\n\n"
        f"Previous LLM-generated BT candidate:\n{json.dumps(previous_plan, indent=2)}\n\n"
        f"Typed diagnostics:\n{json.dumps(diagnostics, indent=2)}\n"
    )


def _scenario_context(scenario: Scenario) -> str:
    task = {
        "task_id": scenario.task_id,
        "instruction": scenario.instruction,
        "initial_state": list(scenario.initial_state),
        "goal_state": list(scenario.goal_state),
        "entities": [{"id": entity.id, "type": entity.type} for entity in scenario.entities],
        "resources": [{"id": resource.id, "capacity": resource.capacity} for resource in scenario.resources],
    }
    robots = [
        {
            "id": robot.id,
            "type": robot.type,
            "capabilities": [
                {
                    "name": cap.name,
                    "parameters": list(cap.parameters),
                    "parameter_types": list(cap.parameter_types),
                    "resources": list(cap.resources),
                    "action_type": cap.action_type,
                    "duration_ticks": cap.duration_ticks,
                    "timeout_ticks": cap.timeout_ticks,
                    "preconditions": list(cap.preconditions),
                    "effects": {"add": list(cap.effects.add), "delete": list(cap.effects.delete)},
                }
                for cap in robot.capabilities
            ],
        }
        for robot in scenario.robots
    ]
    return f"Scenario:\n{json.dumps(task, indent=2)}\n\nRobot capability library:\n{json.dumps(robots, indent=2)}"


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a provider response."""
    candidate = _first_json_object(_strip_fence(text.strip()))
    if candidate is None:
        raise ValueError("Could not find a JSON object in the LLM response.")
    document = json.loads(candidate)
    if not isinstance(document, dict):
        raise ValueError("The LLM response JSON root must be an object.")
    return document


def _strip_fence(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
