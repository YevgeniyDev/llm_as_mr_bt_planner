"""Physical MuJoCo execution, intentionally isolated from LLM generation.

Importing :mod:`llm_mr_bt_planner` or launching its UI does not import MuJoCo.
The simulator is entered only through ``lmrbtp mujoco``.
"""

from __future__ import annotations

__all__: list[str] = []
