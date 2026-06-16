"""OpenAI (and OpenAI-compatible) chat-completions client.

Uses only the standard library so the project stays dependency-free. The base
URL and timeout are configurable so any OpenAI-compatible endpoint works.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import LLMError, redact_secrets

DEFAULT_MODEL = "gpt-4o"


class OpenAIClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        self.name = "openai"
        self.model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self._timeout = timeout if timeout is not None else float(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
        self._temperature = temperature if temperature is not None else float(os.environ.get("OPENAI_TEMPERATURE", "0"))

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
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise LLMError(f"LLM request failed: HTTP {error.code}: {redact_secrets(detail)}") from error
    except urllib.error.URLError as error:
        raise LLMError(f"LLM request failed: {redact_secrets(str(error.reason))}") from error
