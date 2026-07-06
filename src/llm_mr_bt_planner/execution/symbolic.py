"""Symbolic execution backend - wraps the in-process BT simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..domain import Scenario
from ..plan import Plan
from ..simulation import simulate
from .base import ExecutionResult

if TYPE_CHECKING:
    from ..recovery import RecoveryController


class SymbolicExecutionBackend:
    name = "symbolic"

    def __init__(self, max_ticks: int = 80, recovery: RecoveryController | None = None) -> None:
        self.max_ticks = max_ticks
        self.recovery = recovery

    def execute(self, plan: Plan, scenario: Scenario) -> ExecutionResult:
        if self.recovery is not None:
            return self._execute_with_recovery(plan, scenario)
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

    def _execute_with_recovery(self, plan: Plan, scenario: Scenario) -> ExecutionResult:
        assert self.recovery is not None
        result = self.recovery.run(plan, scenario)
        report = result.report
        return ExecutionResult(
            backend=self.name,
            success=result.success,
            goal_success=result.goal_success,
            final_state=report.final_state if report else [],
            trace=report.trace if report else [],
            errors=report.errors if report else [],
            details={
                "max_ticks": self.max_ticks,
                "recovery": result.to_dict(),
                "failures": report.failures if report else [],
            },
        )
