"""Generate and verify synchronized multi-robot Behavior Trees.

The package contains a standalone symbolic contract planner plus an optional
MuJoCo executor for the first courier scenario. ROS 2, perception, and hardware
execution remain outside this release.
"""

from __future__ import annotations

__version__ = "0.5.0"

from .domain import Scenario, load_scenario, parse_scenario
from .llm import AnthropicClient, OpenAIClient, get_client
from .plan import Plan, parse_plan
from .planner import PlannerResult, run_planner
from .service import PipelineOutcome, PlannerService
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
    "PlannerService",
    "PipelineOutcome",
    "ValidationReport",
    "validate_plan",
    "get_client",
    "OpenAIClient",
    "AnthropicClient",
    "bt_to_mermaid",
    "plan_to_html",
]
