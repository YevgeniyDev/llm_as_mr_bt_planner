"""Explicit API-key persistence using the operating system credential store."""

from __future__ import annotations

SERVICE_NAME = "llm-mr-bt-planner"


class SecretStore:
    """Keys are never written to project files, logs, manifests, or artifacts."""

    def save(self, provider: str, api_key: str) -> None:
        if not api_key.strip():
            raise ValueError("Cannot save an empty API key.")
        keyring = _keyring()
        keyring.set_password(SERVICE_NAME, _account(provider), api_key.strip())

    def load(self, provider: str) -> str | None:
        return _keyring().get_password(SERVICE_NAME, _account(provider))

    def delete(self, provider: str) -> None:
        keyring = _keyring()
        try:
            keyring.delete_password(SERVICE_NAME, _account(provider))
        except keyring.errors.PasswordDeleteError:
            return


def _account(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in {"openai", "anthropic"}:
        raise ValueError(f"Unsupported credential provider '{provider}'.")
    return f"{normalized}-api-key"


def _keyring():
    try:
        import keyring
    except ImportError as error:
        raise RuntimeError("Saving keys requires the optional UI dependency 'keyring'.") from error
    return keyring
