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

from .base import LLMError, redact_secrets

DEFAULT_MODEL = "gpt-4o"

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
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self._timeout = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
        self._temperature = temperature if temperature is not None else float(os.environ.get("OPENAI_TEMPERATURE", "0"))
        self.temperature = self._temperature
        self.seed = seed

    def set_seed(self, seed: int | None) -> None:
        """Set the best-effort provider seed used by the next request."""
        self.seed = seed

    def complete(self, system: str, user: str) -> str:
        if not self._api_key:
            raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add a key.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self._temperature,
            "response_format": {"type": "json_object"},
        }
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
            raise LLMError(f"LLM request failed: HTTP {error.code}: {redact_secrets(detail)}") from error
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
