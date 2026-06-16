"""Execution backends: symbolic now, ROS-ready scaffold for real robots."""

from __future__ import annotations

from .base import ExecutionBackend, ExecutionResult
from .ros import RosExecutionBackend, export_behaviortree_cpp_xml
from .symbolic import SymbolicExecutionBackend

__all__ = [
    "ExecutionBackend",
    "ExecutionResult",
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
    if key not in _BACKENDS:
        raise ValueError(f"Unknown execution backend '{name}'. Choose from: {', '.join(sorted(_BACKENDS))}.")
    return _BACKENDS[key](**kwargs)
