"""Markdown-authored planning *skills* injected into the LLM prompt.

This is the "skills for the LLM / markdown file / prompt engineering" layer:
reusable planning guidance authored as Markdown files rather than hardcoded in
:mod:`llm_mr_bt_planner.prompts`. A skill is a ``.md`` file with a small
frontmatter block::

    ---
    name: robot-scoped-predicates
    description: How holding(R, x) style predicates are produced
    tags: manipulation, sync
    applies_to: pick_gear, mount_gear
    ---
    <free-text guidance / worked example for the model>

The loader parses the frontmatter with the standard library only (no PyYAML),
selects the skills relevant to a scenario, and renders them into a prompt
section. Skills are *additive and optional* - off by default - so they never
silently change the pure-mode baseline the experiments depend on. The core
planning rules/method stay in :mod:`llm_mr_bt_planner.prompts`; this layer only
adds opt-in guidance on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .domain import Scenario

DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    applies_to: tuple[str, ...] = ()
    body: str = ""
    path: str = ""


# --------------------------------------------------------------------------- #
# Frontmatter parsing (stdlib only - a tiny YAML subset, not PyYAML)
# --------------------------------------------------------------------------- #


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``---``-delimited frontmatter block from the body.

    Supports just what skill metadata needs: ``key: value`` lines, where a value
    becomes a list if it is wrapped in ``[ ... ]`` or contains commas, and a
    scalar (quotes stripped) otherwise. Returns ``(metadata, body)``; when there
    is no frontmatter (or it is unterminated) returns ``({}, text)``.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, Any] = {}
    body_start: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            body_start = index + 1
            break
        line = lines[index]
        if not line.strip() or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        meta[key.strip()] = _parse_value(raw.strip())
    if body_start is None:
        return {}, text
    return meta, "\n".join(lines[body_start:]).strip()


def _parse_value(raw: str) -> Any:
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw.strip().strip('"').strip("'")


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if value in (None, ""):
        return ()
    return (str(value).strip(),)


# --------------------------------------------------------------------------- #
# Loading & selection
# --------------------------------------------------------------------------- #


def load_skills(skills_dir: str | Path = DEFAULT_SKILLS_DIR) -> list[Skill]:
    """Load every ``*.md`` skill in ``skills_dir``, sorted by name.

    Malformed or unnamed files are skipped silently (this never raises and, by
    design, never emits a ``DeprecationWarning`` - the test config turns those
    into errors).
    """
    directory = Path(skills_dir)
    if not directory.is_dir():
        return []
    skills: list[Skill] = []
    for path in sorted(directory.glob("*.md")):
        try:
            meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        name = str(meta.get("name") or "").strip()
        if not name:
            continue  # a skill must at least be named to be usable
        skills.append(
            Skill(
                name=name,
                description=str(meta.get("description", "")).strip(),
                tags=_as_tuple(meta.get("tags")),
                applies_to=_as_tuple(meta.get("applies_to")),
                body=body.strip(),
                path=str(path),
            )
        )
    return skills


def _scenario_vocab(scenario: Scenario) -> set[str]:
    """Names a skill's ``applies_to`` may match: capabilities + robot ids/types."""
    vocab: set[str] = set()
    for robot in scenario.robots:
        vocab.add(robot.id)
        vocab.add(robot.type)
        vocab |= robot.capability_names
    return vocab


def select_skills(
    skills: list[Skill],
    scenario: Scenario,
    *,
    tags: set[str] | None = None,
) -> list[Skill]:
    """Keep the skills relevant to ``scenario``.

    A skill is relevant when it is *general* (``applies_to`` contains ``"*"``, or
    both ``applies_to`` and ``tags`` are empty), or its ``applies_to`` intersects
    the scenario vocabulary (capability/robot names), or its ``tags`` intersect
    the requested ``tags``. General skills are listed first, then the rest by the
    load order (already alphabetical by name).
    """
    vocab = _scenario_vocab(scenario)
    wanted_tags = tags or set()
    general: list[Skill] = []
    specific: list[Skill] = []
    for skill in skills:
        if "*" in skill.applies_to or (not skill.applies_to and not skill.tags):
            general.append(skill)
        elif (set(skill.applies_to) & vocab) or (set(skill.tags) & wanted_tags):
            specific.append(skill)
    return general + specific


def render_skills_section(skills: list[Skill]) -> str:
    """Render selected skills as a prompt block (``""`` when there are none)."""
    if not skills:
        return ""
    parts = ["Authored planning skills (reusable guidance; the scenario data remains authoritative):"]
    for skill in skills:
        header = f"## {skill.name}"
        if skill.description:
            header += f" - {skill.description}"
        parts.append(f"{header}\n{skill.body}".rstrip())
    return "\n\n".join(parts) + "\n"


def skills_section_for(
    scenario: Scenario,
    skills_dir: str | Path = DEFAULT_SKILLS_DIR,
    *,
    tags: set[str] | None = None,
) -> str:
    """Convenience: load + select + render in one call (``""`` if no skills)."""
    return render_skills_section(select_skills(load_skills(skills_dir), scenario, tags=tags))
