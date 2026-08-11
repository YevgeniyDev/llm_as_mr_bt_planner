"""OpenAI (and OpenAI-compatible) chat-completions client.

Uses only the standard library so the project stays dependency-free. The base
URL and timeout are configurable so any OpenAI-compatible endpoint works.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .base import LLMError, redact_secrets
from .catalog import default_model

DEFAULT_MODEL = default_model("openai")

# Bounded retry for transient errors (rate limits / 5xx). Stays dependency-free.
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", "6"))
_RETRY_HINT = re.compile(r"try again in ([0-9.]+)s")


class OpenAIClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.name = "openai"
        self.model: str = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self._base_url = resolved_base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
        requested_temperature = (
            temperature if temperature is not None else float(os.environ.get("OPENAI_TEMPERATURE", "0"))
        )
        self._temperature = (
            None if _requires_default_temperature(self.model) else requested_temperature
        )
        self.temperature = self._temperature
        self.seed = seed

    def set_seed(self, seed: int | None) -> None:
        """Set the best-effort provider seed used by the next request."""
        self.seed = seed

    def complete(self, system: str, user: str) -> str:
        if not self._api_key:
            raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add a key.")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self._temperature is not None:
            payload["temperature"] = self._temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        request = urllib.request.Request(
            self._completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        body = _send(request, self._timeout)
        result = json.loads(body)
        return result["choices"][0]["message"]["content"]

    def _completions_url(self) -> str:
        explicit = os.environ.get("OPENAI_API_URL")
        if explicit:
            return explicit
        if self._base_url.endswith("/chat/completions"):
            return self._base_url
        return f"{self._base_url}/chat/completions"


def _send(request: urllib.request.Request, timeout: float) -> str:
    for attempt in range(MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if error.code in RETRY_STATUSES and attempt < MAX_RETRIES:
                time.sleep(_retry_after(error, detail, attempt))
                continue
            raise LLMError(
                f"OpenAI API request failed (HTTP {error.code}): "
                f"{redact_secrets(_api_error_detail(detail))}"
            ) from error
        except urllib.error.URLError as error:
            raise LLMError(f"LLM request failed: {redact_secrets(str(error.reason))}") from error
    raise LLMError("LLM request failed: exhausted retries")  # pragma: no cover


def _retry_after(error: urllib.error.HTTPError, detail: str, attempt: int) -> float:
    """Seconds to wait before retrying: honor Retry-After header or the API's
    'try again in Xs' hint, else exponential backoff. Capped to avoid stalls."""
    header = error.headers.get("Retry-After") if error.headers else None
    if header:
        try:
            return min(float(header), 30.0)
        except ValueError:
            pass
    match = _RETRY_HINT.search(detail)
    if match:
        return min(float(match.group(1)) + 0.5, 30.0)
    return min(2.0 ** attempt, 30.0)


def _requires_default_temperature(model: str) -> bool:
    """GPT-5-family models reject custom temperature at default reasoning settings."""
    return model.lower().startswith("gpt-5")


def _api_error_detail(detail: str) -> str:
    """Extract a readable provider message instead of exposing raw JSON in the UI."""
    try:
        document = json.loads(detail)
    except (json.JSONDecodeError, TypeError):
        return detail
    error = document.get("error") if isinstance(document, dict) else None
    if not isinstance(error, dict):
        return detail
    message = error.get("message")
    parameter = error.get("param")
    if not isinstance(message, str) or not message.strip():
        return detail
    if isinstance(parameter, str) and parameter and parameter not in message:
        return f"{message.strip()} (parameter: {parameter})"
    return message.strip()
