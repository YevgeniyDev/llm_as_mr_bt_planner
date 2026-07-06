"""Markdown-authored planning skills: parsing, selection, rendering, injection."""

from __future__ import annotations

import warnings

from llm_mr_bt_planner.domain import parse_scenario
from llm_mr_bt_planner.prompts import build_prompt
from llm_mr_bt_planner.skills import (
    DEFAULT_SKILLS_DIR,
    Skill,
    load_skills,
    parse_frontmatter,
    render_skills_section,
    select_skills,
    skills_section_for,
)


def _scenario():
    return parse_scenario({
        "task_id": "t", "instruction": "x",
        "initial_state": [], "goal_state": ["done()"],
        "objects": [], "locations": [],
        "robots": [
            {"id": "arm", "type": "manipulator", "capabilities": [
                {"name": "pick_gear", "parameters": ["g"], "preconditions": [],
                 "effects": {"add": ["holding(arm, g)"], "delete": []}}]},
        ],
    })


# --- frontmatter parsing (stdlib only) ---------------------------------------


def test_parse_frontmatter_stdlib():
    text = (
        "---\n"
        "name: demo\n"
        "description: a demo\n"
        "tags: a, b\n"
        "applies_to: [pick_gear, mount_gear]\n"
        "---\n"
        "Body line 1\n"
        "Body line 2\n"
    )
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "demo"
    assert meta["description"] == "a demo"
    assert meta["tags"] == ["a", "b"]                 # comma list
    assert meta["applies_to"] == ["pick_gear", "mount_gear"]  # [ ... ] list
    assert body == "Body line 1\nBody line 2"


def test_parse_frontmatter_absent():
    meta, body = parse_frontmatter("no frontmatter here\njust text")
    assert meta == {}
    assert body == "no frontmatter here\njust text"


# --- loading -----------------------------------------------------------------


def test_load_skips_malformed_without_warning(tmp_path):
    (tmp_path / "good.md").write_text(
        "---\nname: good\ndescription: d\napplies_to: \"*\"\n---\nbody\n", encoding="utf-8")
    (tmp_path / "no_frontmatter.md").write_text("just notes, no frontmatter\n", encoding="utf-8")
    (tmp_path / "unnamed.md").write_text("---\ndescription: missing name\n---\nbody\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("---\nname: ignored\n---\n", encoding="utf-8")  # not .md
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any DeprecationWarning would fail here
        skills = load_skills(tmp_path)
    assert [skill.name for skill in skills] == ["good"]


def test_load_missing_dir_is_empty(tmp_path):
    assert load_skills(tmp_path / "does_not_exist") == []


# --- selection ---------------------------------------------------------------


def test_select_by_capability_tag_and_general():
    scenario = _scenario()  # capability: pick_gear
    skills = [
        Skill(name="gen", applies_to=("*",)),
        Skill(name="cap", applies_to=("pick_gear",)),
        Skill(name="tagged", tags=("sync",)),
        Skill(name="irrelevant", applies_to=("no_such_cap",), tags=("other",)),
    ]
    selected = select_skills(skills, scenario, tags={"sync"})
    picked = {skill.name for skill in selected}
    assert {"gen", "cap", "tagged"} <= picked
    assert "irrelevant" not in picked
    assert selected[0].name == "gen"  # general skills first


# --- rendering & prompt injection --------------------------------------------


def test_render_is_empty_when_no_skills():
    assert render_skills_section([]) == ""


def test_render_includes_name_and_body():
    section = render_skills_section([Skill(name="s1", description="d", body="guidance")])
    assert "## s1 - d" in section
    assert "guidance" in section


def test_prompt_unchanged_without_skills():
    scenario = _scenario()
    assert build_prompt(scenario) == build_prompt(scenario, skills_section="")
    assert "Authored planning skills" not in build_prompt(scenario)


def test_prompt_places_skills_between_context_and_schema():
    scenario = _scenario()
    section = render_skills_section([Skill(name="x", body="body")])
    prompt = build_prompt(scenario, skills_section=section)
    assert prompt.index("Robot capability library") < prompt.index("Authored planning skills")
    assert prompt.index("Authored planning skills") < prompt.index("Required output schema")


# --- the shipped seed skills -------------------------------------------------


def test_seed_skills_load_and_select():
    skills = load_skills(DEFAULT_SKILLS_DIR)
    assert len(skills) >= 4
    section = skills_section_for(_scenario(), DEFAULT_SKILLS_DIR)
    assert "back-chaining-from-goals" in section  # a general skill is always relevant
    assert "pick-before-place" in section          # matches the pick_gear capability
