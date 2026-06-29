"""LLM-as-MR-BT-Planner: LLM-guided synchronized multi-robot Behavior Trees.

Public API re-exports the pieces most code needs. The pipeline is:

    scenario = load_scenario("data/scenario.json")
    client = get_client("openai")             # or "anthropic"
    result = run_planner(scenario, client)

Execution backends (symbolic now, ROS scaffold for real robots) and a
reproducible experiment runner live in :mod:`llm_mr_bt_planner.execution` and
:mod:`llm_mr_bt_planner.experiments`.
"""

from __future__ import annotations

__version__ = "0.2.0"

from .domain import Scenario, load_scenario, parse_scenario
from .llm import AnthropicClient, OpenAIClient, get_client
from .plan import Plan, parse_plan
from .planner import PlannerResult, run_planner
from .simulation import SimulationReport, simulate
from .validation import ValidationReport, validate_plan
from .viz import bt_to_mermaid, plan_to_html

__all__ = [
    "__version__",
    "Scenario",
    "load_scenario",
    "parse_scenario",
    "Plan",
    "parse_plan",
    "PlannerResult",
    "run_planner",
    "SimulationReport",
    "simulate",
    "ValidationReport",
    "validate_plan",
    "get_client",
    "OpenAIClient",
    "AnthropicClient",
    "bt_to_mermaid",
    "plan_to_html",
]
