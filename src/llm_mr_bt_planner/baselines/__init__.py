"""Baseline plan generators for comparison against the proposed method.

Each baseline produces a plan that is scored by the *same* validator + simulator as the
proposed method, so all methods populate identical metric columns on the same scenarios.
Select one via the ``BASELINES`` registry (exposed as ``--method`` on the CLI).
"""

from __future__ import annotations

from typing import Callable

from ..planner import PlannerResult, run_planner
from .flat_llm import run_flat
from .hier_llm import run_hier
from .mrbtp_adapter import run_mrbtp

# "proposed" is the full method (run_planner); the rest are external baselines.
BASELINES: dict[str, Callable[..., PlannerResult]] = {
    "proposed": run_planner,
    "flat": run_flat,
    "hier": run_hier,
    "mrbtp": run_mrbtp,
}

METHOD_LABELS = {
    "proposed": "Proposed",
    "flat": "LLM-MARS-style (flat)",
    "hier": "LLM-as-BT-Planner-style",
    "mrbtp": "MRBTP",
}


def get_runner(method: str) -> Callable[..., PlannerResult]:
    if method not in BASELINES:
        raise KeyError(f"Unknown method '{method}'. Choices: {', '.join(BASELINES)}.")
    return BASELINES[method]


__all__ = ["BASELINES", "METHOD_LABELS", "get_runner", "run_flat", "run_hier", "run_mrbtp"]
