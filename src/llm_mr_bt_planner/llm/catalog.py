"""Curated current text-model catalog for multi-robot BT generation."""

from __future__ import annotations

from typing import Final

DEFAULT_MODELS: Final[dict[str, str]] = {
    "openai": "gpt-5.6-sol",
    "anthropic": "claude-opus-5",
}

# These are current, generally available text models suitable for structured
# planning. Specialized, deprecated, legacy, and restricted-access models are
# intentionally excluded from the UI.
MODEL_OPTIONS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "openai": (
        ("GPT-5.6 Sol — highest planning quality", "gpt-5.6-sol"),
        ("GPT-5.6 Terra — balanced quality and cost", "gpt-5.6-terra"),
        ("GPT-5.6 Luna — fastest and lowest cost", "gpt-5.6-luna"),
    ),
    "anthropic": (
        ("Claude Opus 5 — recommended for complex planning", "claude-opus-5"),
        ("Claude Fable 5 — highest capability, slower", "claude-fable-5"),
        ("Claude Sonnet 5 — balanced speed and quality", "claude-sonnet-5"),
        ("Claude Haiku 4.5 — fastest and lowest cost", "claude-haiku-4-5-20251001"),
    ),
}

MODEL_CATALOG_REVIEWED: Final = "2026-08-11"


def default_model(provider: str) -> str:
    """Return the application's current default model for a provider."""
    try:
        return DEFAULT_MODELS[provider.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported provider: {provider!r}.") from error


def model_choices(provider: str) -> list[tuple[str, str]]:
    """Return Gradio label/value pairs with the real default model selected."""
    normalized = provider.strip().lower()
    try:
        options = MODEL_OPTIONS[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported provider: {provider!r}.") from error
    default = default_model(normalized)
    alternatives = [option for option in options if option[1] != default]
    return [(f"Provider default — {default}", default), *alternatives]


def is_catalog_model(provider: str, model: str) -> bool:
    """Return whether an explicit model is selectable for the provider."""
    normalized_provider = provider.strip().lower()
    normalized_model = model.strip()
    return any(value == normalized_model for _, value in MODEL_OPTIONS.get(normalized_provider, ()))
