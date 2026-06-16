"""Execution-backend abstraction.

A backend takes a validated plan and *runs* it against some world. The symbolic
backend runs the in-process simulator; a future ROS backend would dispatch the
same behavior trees to real robot skill servers. Keeping this behind one
interface is what lets the rest of the pipeline stay unchanged when the
execution substrate is upgraded for real-robot testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..domain import Scenario
from ..plan import Plan


@dataclass
class ExecutionResult:
    backend: str
    success: bool
    goal_success: bool
    final_state: list[str]
    trace: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "success": self.success,
            "goal_success": self.goal_success,
            "final_state": self.final_state,
            "trace": self.trace,
            "errors": self.errors,
            "details": self.details,
        }


@runtime_checkable
class ExecutionBackend(Protocol):
    name: str

    def execute(self, plan: Plan, scenario: Scenario) -> ExecutionResult:
        ...
