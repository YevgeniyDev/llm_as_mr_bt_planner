"""LLM provider clients and a small factory."""

from __future__ import annotations

from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMError, redact_secrets
from .openai_client import OpenAIClient

__all__ = [
    "LLMClient",
    "LLMError",
    "redact_secrets",
    "OpenAIClient",
    "AnthropicClient",
    "get_client",
]

_PROVIDERS = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def get_client(provider: str, model: str | None = None, **kwargs) -> LLMClient:
    """Construct a client for ``provider`` ('openai' or 'anthropic')."""
    key = provider.lower()
    if key not in _PROVIDERS:
        raise ValueError(f"Unknown LLM provider '{provider}'. Choose from: {', '.join(sorted(_PROVIDERS))}.")
    return _PROVIDERS[key](model=model, **kwargs)
