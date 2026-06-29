"""Flat baseline (LLM-MARS-style): single-shot multi-robot BT generation.

One generic LLM call asks for one behavior tree per robot. There is no back-chaining
method, no synchronization machinery, and no self-correction loop - so this exposes the
producer-omission and missing-synchronization failure modes that the proposed method's
verifier loop is designed to catch. ``correction_rounds`` is always 0.
"""

from __future__ import annotations

import time
from typing import Any

from ..domain import Scenario
from ..llm.base import LLMClient
from ..planner import PlannerResult
from .base import generate_best, make_result
from .prompts import build_flat_prompt


def run_flat(
    scenario: Scenario,
    client: LLMClient,
    *,
    max_corrections: int = 4,  # noqa: ARG001 - no loop; accepted for a uniform signature
    max_ticks: int = 80,
    samples: int = 1,
    **kwargs: Any,
) -> PlannerResult:
    start = time.monotonic()
    plan, validation, simulation = generate_best(
        client, build_flat_prompt(scenario), scenario, samples, max_ticks
    )
    return make_result(
        scenario, client, plan, validation, simulation,
        rounds=0, wall_seconds=time.monotonic() - start,
    )
