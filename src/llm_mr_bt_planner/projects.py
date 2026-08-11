"""Small local project store for reusable scenario documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import save_json
from .domain import parse_scenario, scenario_to_dict


class ProjectStore:
    def __init__(self, root: str | Path = "projects") -> None:
        self.root = Path(root)

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.stem for path in self.root.glob("*.json") if path.is_file())

    def save(
        self,
        name: str,
        scenario_document: dict[str, Any],
        settings: dict[str, Any] | None = None,
    ) -> Path:
        project = _safe_name(name)
        scenario = parse_scenario(scenario_document, strict=True)
        path = self.root / f"{project}.json"
        safe_settings = {
            key: value
            for key, value in (settings or {}).items()
            if key in {"provider", "model", "max_corrections", "max_ticks"}
        }
        save_json(
            path,
            {
                "project_version": "1.0",
                "scenario": scenario_to_dict(scenario),
                "settings": safe_settings,
            },
        )
        return path

    def load(self, name: str) -> dict[str, Any]:
        path = self.root / f"{_safe_name(name)}.json"
        if not path.exists():
            raise FileNotFoundError(f"Saved project '{name}' does not exist.")
        document = json.loads(path.read_text(encoding="utf-8"))
        scenario = document.get("scenario") if isinstance(document, dict) else None
        if not isinstance(scenario, dict):
            raise ValueError(f"Saved project '{name}' has no scenario object.")
        parse_scenario(scenario, strict=True)
        settings = document.get("settings", {})
        if not isinstance(settings, dict):
            settings = {}
        return {"project_version": "1.0", "scenario": scenario, "settings": settings}


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-")
    if not safe:
        raise ValueError("Project name must contain at least one letter or number.")
    return safe
