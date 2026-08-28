from __future__ import annotations

import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from llm_mr_bt_planner.cli import main as cli_main
from llm_mr_bt_planner.comparison.llm_bt_native import ground_action_templates
from llm_mr_bt_planner.comparison.llm_hbt import (
    ProviderGenerator,
    ReplayGenerator,
    run_llm_hbt,
    run_llm_hbt_recovery,
)
from llm_mr_bt_planner.comparison.llm_hbt_native import (
    LLMHBTNativeError,
    parse_action_response,
    parse_assignment_response,
    parse_initialization_response,
)
from llm_mr_bt_planner.comparison.llm_hbt_prompts import (
    build_assignment_prompt,
    build_initialization_prompt,
)
from llm_mr_bt_planner.comparison.llm_hbt_source import (
    prepare_official_source,
    verify_prepared_source,
)
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.recovery import build_runtime_recovery_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "examples" / "three_robot_component_installation.json"


def _conditions() -> list[str]:
    return [
        "docked(unitree_go2_z1,destination_dock)",
        "at(primary_part,destination_cradle)",
        "arm_stowed(unitree_go2_z1)",
        "installed_component(target_fixture)",
        "gripper_empty(franka_a)",
        "gripper_empty(unitree_go2_z1)",
        "gripper_empty(franka_b)",
    ]


def _entry(stage: str, response: dict, **context: object) -> dict:
    return {"stage": stage, "response": response, **context}


def _pair(
    condition: str,
    requester: str | None,
    robot: str,
    mode: str,
    action: str,
    track: str,
) -> list[dict]:
    return [
        _entry(
            "assign",
            {"robot": robot, "mode": mode, "task": f"establish {condition}"},
            condition=condition,
            requester=requester,
            track=track,
        ),
        _entry(
            "select_action",
            {"action": action},
            condition=condition,
            robot=robot,
            track=track,
        ),
    ]


def _nominal_responses() -> list[dict]:
    responses = [
        _entry(
            "initialize",
            {"conditions": _conditions()},
            track="nominal",
        )
    ]
    responses += _pair(
        "docked(unitree_go2_z1,destination_dock)",
        "unitree_go2_z1",
        "unitree_go2_z1",
        "local",
        "navigate_destination(primary_part)",
        "nominal",
    )
    responses += _pair(
        "holding(unitree_go2_z1,primary_part)",
        "unitree_go2_z1",
        "unitree_go2_z1",
        "local",
        "pick_source_cradle(primary_part)",
        "nominal",
    )
    responses += _pair(
        "at(primary_part,source_cradle)",
        "unitree_go2_z1",
        "franka_a",
        "delegated",
        "place_source_cradle(primary_part)",
        "nominal",
    )
    responses += _pair(
        "holding(franka_a,primary_part)",
        "franka_a",
        "franka_a",
        "local",
        "pick_source_part(primary_part,primary_bin)",
        "nominal",
    )
    responses += _pair(
        "at(primary_part,destination_cradle)",
        None,
        "unitree_go2_z1",
        "local",
        "place_destination_cradle(primary_part)",
        "nominal",
    )
    responses += _pair(
        "arm_stowed(unitree_go2_z1)",
        "unitree_go2_z1",
        "unitree_go2_z1",
        "local",
        "stow_arm_destination(primary_part)",
        "nominal",
    )
    responses += _pair(
        "installed_component(target_fixture)",
        None,
        "franka_b",
        "local",
        "install_target(primary_part,target_fixture)",
        "nominal",
    )
    responses += _pair(
        "holding(franka_b,primary_part)",
        "franka_b",
        "franka_b",
        "local",
        "pick_destination_cradle(primary_part)",
        "nominal",
    )
    return responses


def _recovery_responses() -> list[dict]:
    responses: list[dict] = []
    responses += _pair(
        "docked(unitree_go2_z1,destination_dock)",
        "unitree_go2_z1",
        "unitree_go2_z1",
        "local",
        "navigate_destination(primary_part)",
        "recovery",
    )
    responses += _pair(
        "holding(unitree_go2_z1,primary_part)",
        "unitree_go2_z1",
        "unitree_go2_z1",
        "local",
        "recover_fallen_part(primary_part,source_floor)",
        "recovery",
    )
    responses += _pair(
        "at(primary_part,destination_cradle)",
        None,
        "unitree_go2_z1",
        "local",
        "place_destination_cradle(primary_part)",
        "recovery",
    )
    responses += _pair(
        "arm_stowed(unitree_go2_z1)",
        "unitree_go2_z1",
        "unitree_go2_z1",
        "local",
        "stow_arm_destination(primary_part)",
        "recovery",
    )
    responses += _pair(
        "installed_component(target_fixture)",
        None,
        "franka_b",
        "local",
        "install_target(primary_part,target_fixture)",
        "recovery",
    )
    responses += _pair(
        "holding(franka_b,primary_part)",
        "franka_b",
        "franka_b",
        "local",
        "pick_destination_cradle(primary_part)",
        "recovery",
    )
    return responses


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


def _snapshot() -> dict[str, object]:
    return {
        "measured_initial_state": list(_measured_facts()),
        "failure_observation": _failure_observation(),
    }


def test_nominal_prompts_are_fault_blind_and_recovery_prompt_exposes_snapshot():
    nominal = load_scenario(SCENARIO_PATH, strict=True)
    initialization = build_initialization_prompt(nominal, list(nominal.goal_state))
    assignment = build_assignment_prompt(
        nominal,
        ground_action_templates(nominal),
        failed_condition="installed_component(target_fixture)",
        requester=None,
        observed_state=set(nominal.initial_state),
        failure_observation=None,
    )
    assert "source_floor" not in initialization + assignment
    assert "recover_fallen_part" not in initialization + assignment

    runtime = build_runtime_recovery_scenario(
        nominal,
        measured_initial_state=_measured_facts(),
        failure_observation=_failure_observation(),
    )
    recovery = build_assignment_prompt(
        runtime,
        ground_action_templates(runtime),
        failed_condition="holding(unitree_go2_z1,primary_part)",
        requester="unitree_go2_z1",
        observed_state=set(runtime.initial_state),
        failure_observation=_failure_observation(),
    )
    assert "source_floor" in recovery and "recover_fallen_part" in recovery


def test_strict_native_parsers_reject_hallucination_and_bad_assignment():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    with pytest.raises(LLMHBTNativeError, match="outside the supplied library"):
        parse_initialization_response('{"conditions":["teleported(part)"]}', scenario)
    with pytest.raises(LLMHBTNativeError, match="ownership requires"):
        parse_assignment_response(
            '{"robot":"franka_a","mode":"local","task":"help"}',
            scenario,
            requester="franka_b",
        )
    actions = ground_action_templates(scenario)
    with pytest.raises(LLMHBTNativeError, match="does not establish"):
        parse_action_response(
            '{"action":"navigate_destination(primary_part)"}',
            actions,
            robot="unitree_go2_z1",
            failed_condition="holding(unitree_go2_z1,primary_part)",
        )


def test_nominal_replay_builds_delegated_extensions_and_passes_protocol(tmp_path):
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    result = run_llm_hbt(
        scenario,
        ReplayGenerator(_nominal_responses()),
        tmp_path / "nominal",
        max_ticks=120,
    )

    assert result.plan_generation_success and result.static_validity
    assert result.symbolic_goal_success and result.accepted_plan is not None
    metrics = json.loads(result.metrics.read_text(encoding="utf-8"))
    assert metrics["model_calls"] == 0
    assert metrics["archived_response_count"] == len(_nominal_responses())
    trace = json.loads(
        (result.directory / "native" / "update_trace.json").read_text(encoding="utf-8")
    )
    operations = [event.get("operation") for event in trace["events"]]
    assert "delegated_root_insertion_and_requester_monitor" in operations
    native = json.loads(
        (result.directory / "native" / "native_forest.json").read_text(encoding="utf-8")
    )
    assert any(
        child.get("node") == "Selector"
        for tree in native.values()
        for child in tree["children"]
    )
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["method"]["official_executable_code_found"] is False
    assert manifest["fidelity"]["semantic_rewrites"] == []
    assert manifest["fidelity"]["validator_feedback_to_model"] is False


def test_post_failure_llm_inserts_same_object_floor_recovery(tmp_path):
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    nominal = run_llm_hbt(
        scenario,
        ReplayGenerator(_nominal_responses()),
        tmp_path / "nominal",
        max_ticks=120,
    )
    runtime = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=_measured_facts(),
        failure_observation=_failure_observation(),
    )
    recovery = run_llm_hbt_recovery(
        runtime,
        nominal.directory,
        _snapshot(),
        ReplayGenerator(_recovery_responses()),
        tmp_path / "recovery",
        max_ticks=120,
    )

    assert recovery.plan_generation_success and recovery.static_validity
    assert recovery.symbolic_goal_success and recovery.accepted_plan is not None
    trace = (recovery.directory / "simulation_trace.json").read_text(encoding="utf-8")
    assert "recover_fallen_part" in trace
    assert "primary_part" in trace and "spare_part" not in trace
    update = json.loads(
        (recovery.directory / "native" / "online_update.json").read_text(encoding="utf-8")
    )
    selected = [item["action"]["name"] for item in update["operations"]]
    assert "recover_fallen_part" in selected
    manifest = json.loads(recovery.manifest.read_text(encoding="utf-8"))
    assert manifest["recovery"]["failure_detected_before_recovery_calls"] is True
    assert manifest["recovery"]["same_object_recovery"] is True
    assert manifest["recovery"]["real_llm_recalled_after_failure"] is False


def test_wrong_recovery_action_is_not_repaired(tmp_path):
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    nominal = run_llm_hbt(
        scenario,
        ReplayGenerator(_nominal_responses()),
        tmp_path / "nominal",
    )
    runtime = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=_measured_facts(),
        failure_observation=_failure_observation(),
    )
    responses = _recovery_responses()
    responses[3]["response"] = {"action": "navigate_destination(primary_part)"}
    failed = run_llm_hbt_recovery(
        runtime,
        nominal.directory,
        _snapshot(),
        ReplayGenerator(responses),
        tmp_path / "failed",
    )
    assert not failed.plan_generation_success
    assert failed.accepted_plan is None
    report = json.loads(failed.validation_report.read_text(encoding="utf-8"))
    assert "does not establish" in report["native_generation_errors"][0]["message"]


def test_cli_runs_nominal_then_post_failure_recovery_replay(tmp_path):
    nominal_replay = tmp_path / "nominal-responses.json"
    nominal_replay.write_text(
        json.dumps({"responses": _nominal_responses()}),
        encoding="utf-8",
    )
    nominal_root = tmp_path / "cli-nominal"
    assert cli_main(
        [
            "compare",
            "llm-hbt",
            "run",
            "--scenario",
            str(SCENARIO_PATH),
            "--responses",
            str(nominal_replay),
            "--output",
            str(nominal_root),
            "--max-ticks",
            "120",
        ]
    ) == 0
    nominal_run = next(nominal_root.iterdir())

    snapshot_path = tmp_path / "failure-snapshot.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    recovery_replay = tmp_path / "recovery-responses.json"
    recovery_replay.write_text(
        json.dumps({"responses": _recovery_responses()}),
        encoding="utf-8",
    )
    recovery_root = tmp_path / "cli-recovery"
    assert cli_main(
        [
            "compare",
            "llm-hbt",
            "recover",
            "--scenario",
            str(SCENARIO_PATH),
            "--nominal-run",
            str(nominal_run),
            "--failure-snapshot",
            str(snapshot_path),
            "--responses",
            str(recovery_replay),
            "--output",
            str(recovery_root),
            "--max-ticks",
            "120",
        ]
    ) == 0
    accepted = list(recovery_root.glob("*/accepted_plan.json"))
    assert len(accepted) == 1
    assert "recover_fallen_part" in accepted[0].read_text(encoding="utf-8")


def test_pinned_project_page_and_arxiv_source_fixture(tmp_path, monkeypatch):
    project = tmp_path / "project.tar.gz"
    with tarfile.open(project, "w:gz") as bundle:
        _add_tar_text(bundle, "LLM-HBT-fixture/README.md", "# LLM-HBT\n")
        _add_tar_text(
            bundle,
            "LLM-HBT-fixture/index.html",
            "<p>Code Repository: Coming Soon</p>",
        )
    paper = tmp_path / "paper.tar"
    with tarfile.open(paper, "w") as bundle:
        _add_tar_text(bundle, "00README.json", "{}")
        _add_tar_text(
            bundle,
            "bare_jrnl_new_sample4.tex",
            "Automatic Design of Behavior Trees for Heterogeneous Multirobots",
        )
    project_hash = hashlib.sha256(project.read_bytes()).hexdigest()
    paper_hash = hashlib.sha256(paper.read_bytes()).hexdigest()

    def downloader(url: str, target: Path) -> None:
        fixture = project if "codeload" in url else paper
        target.write_bytes(fixture.read_bytes())

    prepared = prepare_official_source(
        tmp_path / "prepared",
        downloader=downloader,
        project_expected_sha256=project_hash,
        paper_expected_sha256=paper_hash,
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["official_executable_code_found"] is False
    assert manifest["code_availability_statement"] == "Coming Soon"
    monkeypatch.setattr(
        "llm_mr_bt_planner.comparison.llm_hbt_source.PROJECT_ARCHIVE_SHA256",
        project_hash,
    )
    monkeypatch.setattr(
        "llm_mr_bt_planner.comparison.llm_hbt_source.ARXIV_SOURCE_SHA256",
        paper_hash,
    )
    assert verify_prepared_source(prepared.directory)["file_count"] == 4


def test_provider_uses_json_mode_and_records_unreported_reproduction_choice(monkeypatch):
    captured: dict = {}

    def fake_send(request, timeout):  # noqa: ARG001
        captured.update(json.loads(request.data))
        return json.dumps(
            {
                "id": "chatcmpl-llmhbt",
                "model": "gpt-4o-2024-08-06",
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
                "choices": [{"message": {"content": '{"conditions":["system_ready()"]}'}}],
            }
        )

    monkeypatch.setattr("llm_mr_bt_planner.llm.openai_client._send", fake_send)
    generator = ProviderGenerator("openai", "gpt-4o-2024-08-06", "test-only", seed=42)
    result = generator.generate("initialize", "system", "user", {})

    assert "system_ready" in result.text
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42
    assert result.metadata["total_tokens"] == 13


def _add_tar_text(bundle: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    bundle.addfile(info, BytesIO(payload))
