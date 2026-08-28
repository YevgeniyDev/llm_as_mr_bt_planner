from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest

from llm_mr_bt_planner.bt import iter_nodes
from llm_mr_bt_planner.cli import main as cli_main
from llm_mr_bt_planner.comparison.mrbtp import run_mrbtp
from llm_mr_bt_planner.comparison.mrbtp_native import (
    MRBTPNativeError,
    native_forest_document,
    plan_mrbtp,
    validate_native_construction,
)
from llm_mr_bt_planner.comparison.mrbtp_source import (
    REQUIRED_SOURCE_FILES,
    prepare_official_source,
    verify_prepared_source,
)
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.validation import validate_plan

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "examples" / "three_robot_component_installation.json"


def test_fifo_cross_tree_expansion_finds_the_expected_heterogeneous_witness():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    construction = plan_mrbtp(scenario)

    assert construction.solved
    assert validate_native_construction(scenario, construction) == []
    assert [edge.action.name for edge in construction.witness] == [
        "pick_source_part",
        "place_source_cradle",
        "pick_source_cradle",
        "navigate_destination",
        "place_destination_cradle",
        "stow_arm_destination",
        "pick_destination_cradle",
        "install_target",
    ]
    assert {edge.robot for edge in construction.witness} == {
        "franka_a",
        "unitree_go2_z1",
        "franka_b",
    }
    assert any(edge.operation == "cross_tree_expand" for edge in construction.expanded_edges)
    assert construction.witness[0].premise <= set(scenario.initial_state)
    assert construction.witness[-1].target == frozenset(scenario.goal_state)


def test_native_forest_retains_every_expanded_task_action_branch():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    construction = plan_mrbtp(scenario)
    forest = native_forest_document(construction)

    action_nodes = sum(_count_native_actions(tree) for tree in forest.values())
    assert action_nodes == len(construction.expanded_edges)
    assert all(edge.action.robot == edge.robot for edge in construction.expanded_edges)
    assert all(
        len(_part_locations(edge.premise)) <= 1
        for edge in construction.expanded_edges
    )


def test_common_reactive_observation_passes_nominal_protocol(tmp_path):
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    result = run_mrbtp(scenario, tmp_path / "runs", max_ticks=300)

    assert result.plan_generation_success and result.static_validity
    assert result.symbolic_goal_success and result.accepted_plan is not None
    metrics = json.loads(result.metrics.read_text(encoding="utf-8"))
    assert metrics["model_calls"] == 0
    assert metrics["input_tokens"] == 0 and metrics["monetary_cost"] == 0.0
    assert metrics["cross_tree_edge_count"] > 0
    plan = parse_plan(json.loads(result.accepted_plan.read_text(encoding="utf-8")))
    assert all(
        node.source == "planner"
        for tree in plan.behavior_trees.values()
        for node in iter_nodes(tree)
    )
    trace = json.loads(result.simulation_trace.read_text(encoding="utf-8"))
    executed = [event["name"] for event in trace["trace"] if event["event"] == "action"]
    assert executed == [
        "pick_source_part",
        "place_source_cradle",
        "pick_source_cradle",
        "navigate_destination",
        "place_destination_cradle",
        "stow_arm_destination",
        "pick_destination_cradle",
        "install_target",
    ]
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["method"]["llm_subtree_plugin_enabled"] is False
    assert manifest["fidelity"]["semantic_task_action_rewrites"] == []
    assert manifest["prepared_source"]["verified_before_run"] is False


def test_planner_provenance_requires_explicit_reactive_policy_profile():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    construction = plan_mrbtp(scenario)
    document = {
        "schema_version": "2.0",
        "mission_id": scenario.task_id,
        "behavior_trees": {
            robot: tree.to_dict() for robot, tree in construction.trees.items()
        },
    }
    plan = parse_plan(document)
    default = validate_plan(plan, scenario)
    assert any(error.type == "invalid_provenance" for error in default.errors)

    adapted = validate_plan(
        plan,
        scenario,
        allowed_sources=frozenset({"planner"}),
        validation_profile="reactive_policy",
    )
    assert adapted.valid, adapted.to_dicts()


def test_unsolvable_and_expansion_bound_are_reported():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    impossible = replace(scenario, goal_state=("unreachable_goal()",))
    construction = plan_mrbtp(impossible)
    assert not construction.solved
    assert validate_native_construction(impossible, construction)[0]["type"] == "unsolvable"
    with pytest.raises(MRBTPNativeError, match="safety bound"):
        plan_mrbtp(scenario, max_expansions=1)


def test_pinned_official_source_fixture_records_mit_and_verifies_hashes(
    tmp_path,
    monkeypatch,
):
    fixture = tmp_path / "mrbtp.tar.gz"
    with tarfile.open(fixture, "w:gz") as bundle:
        for relative in REQUIRED_SOURCE_FILES:
            content = (
                "MIT License\nPermission is hereby granted\n"
                if relative == "LICENSE"
                else f"fixture {relative}\n"
            )
            _add_tar_text(bundle, f"MRBTP-fixture/{relative}", content)
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()

    def downloader(_url: str, target: Path) -> None:
        target.write_bytes(fixture.read_bytes())

    prepared = prepare_official_source(
        tmp_path / "prepared",
        downloader=downloader,
        expected_sha256=digest,
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["software_license"] == "MIT"
    assert manifest["commit"] == "3d6bd240aa2903245b2335711a97ee394f174313"
    assert manifest["archive_sha256"] == digest
    monkeypatch.setattr(
        "llm_mr_bt_planner.comparison.mrbtp_source.MRBTP_ARCHIVE_SHA256",
        digest,
    )
    assert verify_prepared_source(prepared.directory)["file_count"] == len(
        REQUIRED_SOURCE_FILES
    )


def test_cli_verifies_source_and_runs_complete_nominal_protocol(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source_manifest.json").write_text("{}", encoding="utf-8")
    verified: list[Path] = []

    def fake_verify(path):
        verified.append(Path(path))
        return {"software_license": "MIT"}

    monkeypatch.setattr(
        "llm_mr_bt_planner.comparison.mrbtp_source.verify_prepared_source",
        fake_verify,
    )
    output = tmp_path / "runs"
    assert cli_main(
        [
            "compare",
            "mrbtp",
            "run",
            "--scenario",
            str(SCENARIO_PATH),
            "--source",
            str(source),
            "--output",
            str(output),
            "--max-ticks",
            "300",
        ]
    ) == 0
    assert verified == [source]
    accepted = list(output.glob("*/accepted_plan.json"))
    assert len(accepted) == 1


def _count_native_actions(node: dict) -> int:
    return int(node.get("node") == "Action") + sum(
        _count_native_actions(child) for child in node.get("children", [])
    )


def _part_locations(condition: frozenset[str]) -> list[str]:
    return [literal for literal in condition if literal.startswith("at(primary_part,")]


def _add_tar_text(bundle: tarfile.TarFile, name: str, content: str) -> None:
    payload = content.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    bundle.addfile(info, BytesIO(payload))
