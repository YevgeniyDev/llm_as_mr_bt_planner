"""Hierarchical baseline (LLM-as-BT-Planner-style): decompose then per-robot BTs.

Faithful to the source method's three ingredients - (1) hierarchical decomposition of
the instruction into per-robot action lists, (2) BT encoding, and (3) a recursive
self-correction loop driven by execution feedback. The crucial difference from the
proposed method: robots are planned INDEPENDENTLY, with no cross-robot Condition or
synchronization guidance. So the loop can repair same-robot ordering/producer mistakes,
but inter-robot dependencies surface as unsupported preconditions or deadlocks it cannot
express a fix for. ``correction_rounds`` counts the loop iterations.
"""

from __future__ import annotations

import time
from typing import Any

from ..domain import Scenario
from ..llm.base import LLMClient
from ..planner import PlannerResult
from ..prompts import SYSTEM_PROMPT, _compact_simulation, extract_json
from .base import generate_best, make_result
from .prompts import (
    build_hier_bt_prompt,
    build_hier_correction_prompt,
    build_hier_decompose_prompt,
)


def run_hier(
    scenario: Scenario,
    client: LLMClient,
    *,
    max_corrections: int = 4,
    max_ticks: int = 80,
    samples: int = 1,
    **kwargs: Any,
) -> PlannerResult:
    start = time.monotonic()

    # Stage 1: decompose the task into an ordered per-robot action list.
    raw = extract_json(client.complete(SYSTEM_PROMPT, build_hier_decompose_prompt(scenario)))
    action_plan = raw.get("action_plan", raw) if isinstance(raw, dict) else {}

    # Stage 2: encode the decomposition into independent per-robot behavior trees.
    plan, validation, simulation = generate_best(
        client, build_hier_bt_prompt(scenario, action_plan), scenario, samples, max_ticks
    )

    # Stage 3: recursive self-correction on validator + simulator feedback.
    rounds = 0
    while rounds < max_corrections and not (validation.valid and simulation.success):
        rounds += 1
        prompt = build_hier_correction_prompt(
            scenario, validation.to_dicts(), _compact_simulation(simulation), plan.to_dict()
        )
        plan, validation, simulation = generate_best(client, prompt, scenario, samples, max_ticks)

    return make_result(
        scenario, client, plan, validation, simulation,
        rounds=rounds, wall_seconds=time.monotonic() - start,
    )
