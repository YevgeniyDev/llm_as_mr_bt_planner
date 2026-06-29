"""Symbolic execution backend - wraps the in-process BT simulator."""

from __future__ import annotations

from ..domain import Scenario
from ..plan import Plan
from ..simulation import simulate
from .base import ExecutionResult


class SymbolicExecutionBackend:
    name = "symbolic"

    def __init__(self, max_ticks: int = 80) -> None:
        self.max_ticks = max_ticks

    def execute(self, plan: Plan, scenario: Scenario) -> ExecutionResult:
        report = simulate(plan, scenario, max_ticks=self.max_ticks)
        return ExecutionResult(
            backend=self.name,
            success=report.success,
            goal_success=report.goal_success,
            final_state=report.final_state,
            trace=report.trace,
            errors=report.errors,
            details={"max_ticks": self.max_ticks},
        )
