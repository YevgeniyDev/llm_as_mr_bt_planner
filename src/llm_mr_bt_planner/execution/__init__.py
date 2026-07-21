"""Execution backends for a validated plan.

* ``symbolic`` - the in-process tick simulator (default), optionally wrapped with
  the execution-time recovery ladder (see :mod:`llm_mr_bt_planner.recovery`);
* ``mujoco`` - replays the same trees on real menagerie robots in physics
  (``settle``/``ik`` fidelity; needs the optional ``mujoco`` extra), constructed
  lazily via :func:`get_backend` so the package never hard-depends on it;
* ``ros`` - a documented BehaviorTree.CPP/ROS scaffold for real-robot dispatch.
"""

from __future__ import annotations

from .anomalies import (
    ExecutionIncident,
    IncidentMitigationPolicy,
    IncidentType,
    MitigationAction,
    MitigationDecision,
    apply_incident_to_state,
)
from .base import ExecutionBackend, ExecutionResult
from .ros import RosExecutionBackend, export_behaviortree_cpp_xml
from .symbolic import SymbolicExecutionBackend

__all__ = [
    "ExecutionBackend",
    "ExecutionResult",
    "ExecutionIncident",
    "IncidentMitigationPolicy",
    "IncidentType",
    "MitigationAction",
    "MitigationDecision",
    "apply_incident_to_state",
    "SymbolicExecutionBackend",
    "RosExecutionBackend",
    "export_behaviortree_cpp_xml",
    "get_backend",
]

_BACKENDS = {
    "symbolic": SymbolicExecutionBackend,
    "ros": RosExecutionBackend,
}


def get_backend(name: str, **kwargs) -> ExecutionBackend:
    key = name.lower()
    if key == "mujoco":
        # Imported lazily so the package never hard-depends on the optional 'mujoco' extra.
        from .mujoco_backend import MujocoExecutionBackend

        return MujocoExecutionBackend(**kwargs)
    if key not in _BACKENDS:
        choices = ", ".join(sorted([*_BACKENDS, "mujoco"]))
        raise ValueError(f"Unknown execution backend '{name}'. Choose from: {choices}.")
    return _BACKENDS[key](**kwargs)
