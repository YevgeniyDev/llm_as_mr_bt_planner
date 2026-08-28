from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_mr_bt_planner.cli import main as cli_main
from llm_mr_bt_planner.comparison.llm_bt import (
    REASONING_SYSTEM_PROMPT,
    ReplayReasoner,
    build_reasoning_prompt,
    run_llm_bt,
    run_llm_bt_recovery,
)
from llm_mr_bt_planner.comparison.llm_bt_native import (
    AliasEntry,
    build_alias_catalog,
    semantic_map_xml,
)
from llm_mr_bt_planner.comparison.llm_bt_parser import (
    LLMBTParserError,
    ReplayKeywordParser,
    parse_predictions,
)
from llm_mr_bt_planner.comparison.llm_bt_source import (
    PARSER_DIRECTORY,
    PARSER_MODEL_RELATIVE,
    SOURCE_FILES,
    prepare_official_source,
)
from llm_mr_bt_planner.domain import Scenario, load_scenario
from llm_mr_bt_planner.recovery import build_runtime_recovery_scenario

ROOT = Path(__file__).resolve().parents[1]
COURIER_PATH = ROOT / "examples" / "three_robot_courier.json"
RECOVERY_PATH = ROOT / "examples" / "three_robot_component_installation.json"


def _entry(
    catalog: list[AliasEntry],
    predicate: str,
    action: str,
) -> AliasEntry:
    return next(
        item
        for item in catalog
        if item.predicate.replace(" ", "") == predicate.replace(" ", "")
        and item.action == action
    )


def _replay_bundle(
    scenario: Scenario,
    goals: list[tuple[str, str]],
) -> tuple[ReplayReasoner, ReplayKeywordParser, list[AliasEntry]]:
    catalog = build_alias_catalog(scenario)
    selected = [_entry(catalog, predicate, action) for predicate, action in goals]
    predictions: list[dict[str, str]] = []
    for item in selected:
        target_prefix, target_suffix = item.target.split("_", 1)
        destination_prefix, destination_suffix = item.destination.split("_", 1)
        predictions.extend(
            [
                {"entity": "B-Action", "word": "Move"},
                {"entity": "B-Target", "word": target_prefix},
                {"entity": "I-Target", "word": target_suffix},
                {"entity": "B-Destination", "word": destination_prefix},
                {"entity": "I-Destination", "word": destination_suffix},
            ]
        )
    response = "\n".join(f"{index}. {item.phrase}" for index, item in enumerate(selected, 1))
    return ReplayReasoner(response), ReplayKeywordParser(predictions), selected


def _component_goals() -> list[tuple[str, str]]:
    return [
        ("docked(unitree_go2_z1,destination_dock)", "navigate_destination"),
        ("arm_stowed(unitree_go2_z1)", "stow_arm_destination"),
        ("gripper_empty(unitree_go2_z1)", "place_destination_cradle"),
        ("gripper_empty(franka_a)", "place_source_cradle"),
        ("installed_component(target_fixture)", "install_target"),
        ("gripper_empty(franka_b)", "install_target"),
    ]


def _courier_goals() -> list[tuple[str, str]]:
    return [
        ("docked(unitree_go2_z1,destination_dock)", "navigate_destination"),
        ("arm_stowed(unitree_go2_z1)", "stow_arm_destination"),
        ("gripper_empty(unitree_go2_z1)", "place_destination_cradle"),
        ("gripper_empty(franka_a)", "place_source_cradle"),
        ("installed(payload,target_fixture)", "install_target"),
        ("gripper_empty(franka_b)", "install_target"),
    ]


def _measured_facts() -> tuple[str, ...]:
    return (
        "system_ready()",
        "robot_ready(franka_a)",
        "robot_ready(unitree_go2_z1)",
        "robot_ready(franka_b)",
        "usable(primary_part)",
        "at(primary_part,source_floor)",
        "gripper_empty(franka_a)",
        "gripper_empty(unitree_go2_z1)",
        "gripper_empty(franka_b)",
        "arm_stowed(unitree_go2_z1)",
        "base_stationary(unitree_go2_z1)",
        "docked(unitree_go2_z1,source_dock)",
    )


def _failure_observation() -> dict[str, object]:
    return {
        "classification": "dropped_to_floor",
        "object": "primary_part",
        "object_usable": True,
        "recovery_location": "source_floor",
        "position_m": [-0.04, 0.38, 0.02],
    }


def test_semantic_map_and_nominal_prompt_expose_no_future_failure():
    scenario = load_scenario(RECOVERY_PATH, strict=True)
    semantic_map = semantic_map_xml(scenario)
    catalog = build_alias_catalog(scenario)
    prompt = build_reasoning_prompt(scenario, semantic_map, catalog)

    assert semantic_map.startswith('<semantic_map task_id="three_robot_component_installation">')
    assert '<fact predicate="at(primary_part, primary_bin)"' in semantic_map
    assert "Move object 1 to position 11." in prompt
    assert "postcondition=" in prompt and "ATL action=" in prompt
    assert "dropped_to_floor" not in prompt
    assert "source_floor" not in prompt
    assert "recover_fallen_part" not in prompt
    assert "failure" not in REASONING_SYSTEM_PROMPT.lower()


def test_released_parser_state_machine_joins_wordpieces_and_rejects_drift():
    moves = parse_predictions(
        [
            {"entity": "O", "word": "1"},
            {"entity": "B-Action", "word": "Move"},
            {"entity": "B-Target", "word": "object"},
            {"entity": "I-Target", "word": "##7"},
            {"entity": "B-Destination", "word": "position"},
            {"entity": "I-Destination", "word": "17"},
            {"entity": "B-Location", "word": "table"},
            {"entity": "I-Location", "word": "top"},
        ]
    )
    assert moves[0].to_dict() == {
        "action": "move",
        "target": "object_7",
        "destination": "position_17",
        "location": "table_top",
    }

    with pytest.raises(LLMBTParserError, match="no action-template mapping"):
        parse_predictions([{"entity": "B-Action", "word": "pick"}])
    with pytest.raises(LLMBTParserError, match="before a B-Action"):
        parse_predictions([{"entity": "B-Target", "word": "object"}])
    with pytest.raises(LLMBTParserError, match="missing a target or destination"):
        parse_predictions([{"entity": "B-Action", "word": "move"}])


def test_nominal_replay_expands_atl_and_passes_common_protocol(tmp_path):
    scenario = load_scenario(COURIER_PATH, strict=True)
    reasoner, parser, selected = _replay_bundle(scenario, _courier_goals())
    result = run_llm_bt(
        scenario,
        reasoner,
        parser,
        tmp_path / "runs",
        max_ticks=100,
        invocation=["lmrbtp", "compare", "llm-bt", "run"],
    )

    assert result.plan_generation_success and result.static_validity
    assert result.symbolic_goal_success and result.accepted_plan is not None
    metrics = json.loads(result.metrics.read_text(encoding="utf-8"))
    assert metrics["model_calls"] == 0
    assert metrics["parser_inference_count"] == 0
    assert metrics["archived_response_count"] == 1
    trace = json.loads(
        (result.directory / "native" / "expansion_trace.json").read_text(encoding="utf-8")
    )
    assert trace["unresolved"] == []
    assert any(item["event"] == "partition_external_goal" for item in trace["events"])
    canonical = json.loads(result.canonical_plan.read_text(encoding="utf-8"))
    serialized = json.dumps(canonical)
    assert '"type": "WaitFor"' in serialized
    assert '"type": "AcquireResource"' in serialized
    assert [item.predicate.replace(" ", "") for item in selected] == [
        predicate.replace(" ", "") for predicate, _ in _courier_goals()
    ]
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["fidelity"]["validator_feedback_to_model"] is False
    assert manifest["fidelity"]["semantic_rewrites"] == []


def test_recovery_reuses_nominal_goals_and_retrieves_same_fallen_object(tmp_path):
    nominal_scenario = load_scenario(RECOVERY_PATH, strict=True)
    reasoner, parser, _selected = _replay_bundle(nominal_scenario, _component_goals())
    nominal = run_llm_bt(nominal_scenario, reasoner, parser, tmp_path / "nominal")
    snapshot = {
        "measured_initial_state": list(_measured_facts()),
        "failure_observation": _failure_observation(),
    }
    runtime_scenario = build_runtime_recovery_scenario(
        nominal_scenario,
        measured_initial_state=_measured_facts(),
        failure_observation=_failure_observation(),
    )
    recovery = run_llm_bt_recovery(
        runtime_scenario,
        nominal.directory,
        snapshot,
        tmp_path / "recovery",
        max_ticks=100,
    )

    assert recovery.plan_generation_success and recovery.static_validity
    assert recovery.symbolic_goal_success and recovery.accepted_plan is not None
    metrics = json.loads(recovery.metrics.read_text(encoding="utf-8"))
    assert metrics["model_calls"] == 0
    assert metrics["parser_inference_count"] == 0
    assert metrics["archived_response_count"] == 0
    trace = json.loads(
        (recovery.directory / "native" / "expansion_trace.json").read_text(encoding="utf-8")
    )
    actions = [
        action["name"]
        for event in trace["events"]
        for action in event.get("producer_templates", [])
    ]
    assert "recover_fallen_part" in actions
    assert "pick_source_part" not in actions
    serialized = recovery.canonical_plan.read_text(encoding="utf-8")
    assert "primary_part" in serialized and "source_floor" in serialized
    assert "spare_part" not in serialized
    manifest = json.loads(recovery.manifest.read_text(encoding="utf-8"))
    assert manifest["recovery"] == {
        "llm_recalled_after_failure": False,
        "parser_recalled_after_failure": False,
        "nominal_manifest_sha256": hashlib.sha256(
            nominal.manifest.read_bytes()
        ).hexdigest(),
        "same_parsed_goals_reused": True,
        "runtime_atl_capability_count": sum(
            len(robot.capabilities) for robot in runtime_scenario.robots
        ),
    }


def test_nominal_and_recovery_cli_replay(tmp_path):
    scenario = load_scenario(RECOVERY_PATH, strict=True)
    reasoner, parser, _selected = _replay_bundle(scenario, _component_goals())
    replay = {
        "reasoning_response": reasoner.generate("", "").text,
        "ner_predictions": parser.parse("fixture").predictions,
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    nominal_root = tmp_path / "cli-nominal"
    assert cli_main(
        [
            "compare",
            "llm-bt",
            "run",
            "--scenario",
            str(RECOVERY_PATH),
            "--responses",
            str(replay_path),
            "--output",
            str(nominal_root),
            "--max-ticks",
            "100",
        ]
    ) == 0
    nominal_run = next(nominal_root.iterdir())
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "measured_initial_state": list(_measured_facts()),
                "failure_observation": _failure_observation(),
            }
        ),
        encoding="utf-8",
    )
    recovery_root = tmp_path / "cli-recovery"
    assert cli_main(
        [
            "compare",
            "llm-bt",
            "recover",
            "--scenario",
            str(RECOVERY_PATH),
            "--nominal-run",
            str(nominal_run),
            "--failure-snapshot",
            str(snapshot_path),
            "--output",
            str(recovery_root),
            "--max-ticks",
            "100",
        ]
    ) == 0
    assert len(list(recovery_root.glob("*/accepted_plan.json"))) == 1


def test_pinned_source_fixture_records_license_boundary_and_model_hash(tmp_path):
    config = {
        "architectures": ["DistilBertForTokenClassification"],
        "id2label": {
            "0": "O",
            "1": "B-Action",
            "2": "B-Target",
            "3": "I-Target",
            "4": "B-Destination",
            "5": "I-Destination",
            "6": "B-Location",
            "7": "I-Location",
        },
    }
    payloads = {
        relative: (
            json.dumps(config).encode("utf-8")
            if relative == f"{PARSER_DIRECTORY}/config.json"
            else b"MIT License\n"
            if relative == "LLMBT/BTsUpdate/core/LICENSE"
            else f"fixture {relative}\n".encode()
        )
        for relative in SOURCE_FILES
    }
    model = b"small released-model fixture"
    payloads[PARSER_MODEL_RELATIVE] = model
    expected = {
        relative: hashlib.sha256(content).hexdigest()
        for relative, content in payloads.items()
        if relative != PARSER_MODEL_RELATIVE
    }
    source_root = tmp_path / "prepared" / "source"

    def downloader(_url: str, target: Path) -> None:
        relative = target.relative_to(source_root).as_posix()
        target.write_bytes(payloads[relative])

    prepared = prepare_official_source(
        tmp_path / "prepared",
        downloader=downloader,
        expected_files=expected,
        expected_model_sha256=hashlib.sha256(model).hexdigest(),
        expected_model_bytes=len(model),
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    attribution = (prepared.directory / "ATTRIBUTION.md").read_text(encoding="utf-8")
    assert prepared.file_count == len(SOURCE_FILES)
    assert prepared.parser_model_included
    assert manifest["project_license"] is None
    assert manifest["parser"]["license"] is None
    assert manifest["parser"]["checkpoint_sha256"] == hashlib.sha256(model).hexdigest()
    assert "must not be interpreted as a license for the complete" in attribution
