"""Anthropic Messages API client (stdlib only).

Endpoint, headers, and request/response shape follow the Messages API:
``POST https://api.anthropic.com/v1/messages`` with ``x-api-key`` +
``anthropic-version: 2023-06-01``; the response carries a ``content`` array of
blocks. The default model is ``claude-opus-4-8``; JSON is requested via the
prompt (the plan schema is mildly recursive, so structured-output schemas do not
apply cleanly here).

Note: on Opus 4.7/4.8 ``temperature`` is not a valid parameter, so it is omitted.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .base import LLMError, redact_secrets

DEFAULT_MODEL = "claude-opus-4-8"
API_VERSION = "2023-06-01"


class AnthropicClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.name = "anthropic"
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        self._timeout = timeout if timeout is not None else float(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "120"))
        self._max_tokens = max_tokens or int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))

    def complete(self, system: str, user: str) -> str:
        if not self._api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a key.")

        payload = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        request = urllib.request.Request(
            f"{self._base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )
        body = _send(request, self._timeout)
        result = json.loads(body)
        if result.get("stop_reason") == "refusal":
            raise LLMError("Anthropic API refused the request (stop_reason=refusal).")
        return _extract_text(result)


def _extract_text(result: dict) -> str:
    blocks = result.get("content", [])
    texts = [block.get("text", "") for block in blocks if block.get("type") == "text"]
    if not texts:
        raise LLMError("Anthropic response contained no text block.")
    return "".join(texts)


def _send(request: urllib.request.Request, timeout: float) -> str:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise LLMError(f"LLM request failed: HTTP {error.code}: {redact_secrets(detail)}") from error
    except urllib.error.URLError as error:
        raise LLMError(f"LLM request failed: {redact_secrets(str(error.reason))}") from error
