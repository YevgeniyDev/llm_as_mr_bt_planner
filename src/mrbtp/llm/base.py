"""Provider-agnostic LLM client interface.

A client takes a system prompt plus a user prompt and returns the model's raw
text. JSON extraction lives in the planner, not here, so a client only has to
deal with transport.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when an LLM request fails. The message is API-key redacted."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface every provider implements."""

    name: str
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the model's text response to ``user`` under ``system``."""
        ...


def redact_secrets(message: str) -> str:
    """Strip newlines and obvious API keys from an error string before surfacing it."""
    message = message.replace("\r", " ").replace("\n", " ")
    message = re.sub(r"sk-proj-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    message = re.sub(r"sk-ant-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted_api_key]", message)
    return message[:600]
