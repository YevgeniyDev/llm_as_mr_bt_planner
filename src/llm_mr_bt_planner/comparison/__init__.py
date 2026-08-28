"""Paper-baseline reproductions and non-semantic comparison adapters.

Each baseline keeps its native planner output beside any canonical form used by
the shared validator.  A canonical adapter may change representation, but it
must never add, delete, reorder, or otherwise repair behavior.
"""

from .betr_xp import BETR_XP_METHOD_ID
from .llm_as_bt import LLM_AS_BT_METHOD_ID
from .llm_bt import LLM_BT_METHOD_ID
from .llm_hbt import LLM_HBT_METHOD_ID
from .mrbtp import MRBTP_METHOD_ID

__all__ = [
    "BETR_XP_METHOD_ID",
    "LLM_AS_BT_METHOD_ID",
    "LLM_BT_METHOD_ID",
    "LLM_HBT_METHOD_ID",
    "MRBTP_METHOD_ID",
]
