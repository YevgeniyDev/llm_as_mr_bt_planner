from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from llm_mr_bt_planner.cli import main as cli_main
from llm_mr_bt_planner.comparison.betr_xp import (
    PAPER_TEMPERATURE,
    PAPER_TOP_P,
    ProviderCaller,
    ReplayCaller,
    build_goal_prompt,
    run_betr_xp,
    run_betr_xp_recovery,
)
from llm_mr_bt_planner.comparison.betr_xp_native import (
    BetrXPNativeError,
    build_condition_schemas,
    build_entity_aliases,
    encode_predicate,
    parse_goal_response,
    plan_formula,
)
from llm_mr_bt_planner.comparison.betr_xp_source import (
    REQUIRED_SOURCE_FILES,
    prepare_official_source,
    verify_prepared_source,
)
from llm_mr_bt_planner.domain import Scenario, load_scenario
from llm_mr_bt_planner.recovery import build_runtime_recovery_scenario

ROOT = Path(__file__).resolve().parents[1]
COURIER_PATH = ROOT / "examples" / "three_robot_courier.json"
RECOVERY_PATH = ROOT / "examples" / "three_robot_component_installation.json"


def _goal_response(scenario: Scenario) -> str:
    schemas = build_condition_schemas(scenario)
    entities = build_entity_aliases(scenario)
    goals = [encode_predicate(predicate, schemas, entities) for predicate in scenario.goal_state]
    return "Goal: " + " & ".join(reversed(goals))


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


def test_nominal_prompt_is_strict_and_fault_blind():
    scenario = load_scenario(RECOVERY_PATH, strict=True)
    prompt = build_goal_prompt(
        scenario,
        build_condition_schemas(scenario),
        build_entity_aliases(scenario),
    )

    assert "Conditions:" in prompt and "Objects:" in prompt
    assert "Scene information:" in prompt and "Examples:" in prompt
    assert "Required final conditions for the common evaluation protocol:" in prompt
    assert "installed_component(target_fixture)" in prompt
    assert "Goal: one well-formed formula" in prompt
    assert "source_floor" not in prompt
    assert "dropped_to_floor" not in prompt
    assert "recover_fallen_part" not in prompt


def test_goal_formula_parser_supports_logic_and_rejects_semantic_drift():
    scenario = load_scenario(COURIER_PATH, strict=True)
    schemas = build_condition_schemas(scenario)
    entities = build_entity_aliases(scenario)
    first, second, third = [
        encode_predicate(predicate, schemas, entities)
        for predicate in scenario.goal_state[:3]
    ]
    formula = parse_goal_response(
        f"Goal: {first} | ~({second} & {third})",
        schemas,
        entities,
    )

    assert len(formula.alternatives) == 3
    assert formula.alternatives[0][0].negated is False
    assert formula.alternatives[1][0].negated is True
    with pytest.raises(BetrXPNativeError, match="negative-goal"):
        plan_formula(scenario, formula)
    with pytest.raises(BetrXPNativeError, match="Unknown formal condition"):
        parse_goal_response("Goal: Hallucinated_Payload", schemas, entities)
    with pytest.raises(BetrXPNativeError, match="expects"):
        parse_goal_response(f"Goal: {first}_Payload", schemas, entities)


def test_nominal_replay_generates_reactive_policy_and_passes_protocol(tmp_path):
    scenario = load_scenario(COURIER_PATH, strict=True)
    result = run_betr_xp(
        scenario,
        ReplayCaller({"goal_response": _goal_response(scenario)}),
        tmp_path / "runs",
        max_ticks=100,
    )

    assert result.plan_generation_success and result.static_validity
    assert result.symbolic_goal_success and result.accepted_plan is not None
    metrics = json.loads(result.metrics.read_text(encoding="utf-8"))
    assert metrics["model_calls"] == 0
    assert metrics["archived_response_count"] == 1
    native = json.loads(
        (result.directory / "native" / "native_policy.bt.json").read_text(encoding="utf-8")
    )
    assert all(tokens[0] == "s(" for tokens in native.values())
    canonical = result.canonical_plan.read_text(encoding="utf-8")
    assert '"id": "betrxp.' in canonical
    assert '"type": "Fallback"' in canonical
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["fidelity"]["semantic_rewrites"] == []
    assert manifest["fidelity"]["validator_feedback_to_model"] is False
    assert manifest["method"]["implementation"].endswith("compatibility runner")


def test_recovery_llm_parameter_updates_policy_and_recovers_same_object(tmp_path):
    nominal_scenario = load_scenario(RECOVERY_PATH, strict=True)
    nominal = run_betr_xp(
        nominal_scenario,
        ReplayCaller({"goal_response": _goal_response(nominal_scenario)}),
        tmp_path / "nominal",
        max_ticks=100,
    )
    runtime_scenario = build_runtime_recovery_scenario(
        nominal_scenario,
        measured_initial_state=_measured_facts(),
        failure_observation=_failure_observation(),
    )
    recovery = run_betr_xp_recovery(
        runtime_scenario,
        nominal.directory,
        _snapshot(),
        ReplayCaller(
            {
                "recovery_response": (
                    "Reasoning: The measured intact part is on the source floor.\n"
                    "Parameter value: source_floor"
                )
            }
        ),
        tmp_path / "recovery",
        max_ticks=100,
    )

    assert recovery.plan_generation_success and recovery.static_validity
    assert recovery.symbolic_goal_success and recovery.accepted_plan is not None
    metrics = json.loads(recovery.metrics.read_text(encoding="utf-8"))
    assert metrics["model_calls"] == 0
    assert metrics["archived_response_count"] == 1
    update = json.loads(
        (recovery.directory / "native" / "parameter_update.json").read_text(encoding="utf-8")
    )
    assert update["resolved_parameter"] == {
        "name": "location",
        "before": "source_cradle",
        "after": "source_floor",
    }
    assert update["common_skill_binding"]["name"] == "recover_fallen_part"
    assert update["permanent_policy_update"] is True
    trace = (recovery.directory / "simulation_trace.json").read_text(encoding="utf-8")
    assert "recover_fallen_part" in trace
    assert "primary_part" in trace and "spare_part" not in trace
    manifest = json.loads(recovery.manifest.read_text(encoding="utf-8"))
    assert manifest["recovery"]["failure_resolver_invoked_after_failure"] is True
    assert manifest["recovery"]["real_llm_recalled_after_failure"] is False
    assert manifest["recovery"]["same_object_recovery"] is True


def test_incorrect_recovery_parameter_remains_a_failed_trial(tmp_path):
    scenario = load_scenario(RECOVERY_PATH, strict=True)
    nominal = run_betr_xp(
        scenario,
        ReplayCaller({"goal_response": _goal_response(scenario)}),
        tmp_path / "nominal",
    )
    runtime_scenario = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=_measured_facts(),
        failure_observation=_failure_observation(),
    )
    failed = run_betr_xp_recovery(
        runtime_scenario,
        nominal.directory,
        _snapshot(),
        ReplayCaller(
            {"recovery_response": "Reasoning: Try the old cradle.\nParameter value: source_cradle"}
        ),
        tmp_path / "failed",
    )

    assert not failed.plan_generation_success
    assert not failed.static_validity
    assert not failed.symbolic_goal_success
    report = json.loads(failed.validation_report.read_text(encoding="utf-8"))
    assert "measured object location" in report["native_generation_errors"][0]["message"]


def test_nominal_and_recovery_cli_replay(tmp_path):
    scenario = load_scenario(RECOVERY_PATH, strict=True)
    goal_replay = tmp_path / "goal.json"
    goal_replay.write_text(
        json.dumps({"goal_response": _goal_response(scenario)}),
        encoding="utf-8",
    )
    nominal_root = tmp_path / "cli-nominal"
    assert cli_main(
        [
            "compare",
            "betr-xp-llm",
            "run",
            "--scenario",
            str(RECOVERY_PATH),
            "--responses",
            str(goal_replay),
            "--output",
            str(nominal_root),
            "--max-ticks",
            "100",
        ]
    ) == 0
    nominal_run = next(nominal_root.iterdir())
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(_snapshot()), encoding="utf-8")
    recovery_replay = tmp_path / "recovery.json"
    recovery_replay.write_text(
        json.dumps(
            {
                "recovery_response": (
                    "Reasoning: Use the measured location.\nParameter value: source_floor"
                )
            }
        ),
        encoding="utf-8",
    )
    recovery_root = tmp_path / "cli-recovery"
    assert cli_main(
        [
            "compare",
            "betr-xp-llm",
            "recover",
            "--scenario",
            str(RECOVERY_PATH),
            "--nominal-run",
            str(nominal_run),
            "--failure-snapshot",
            str(snapshot_path),
            "--responses",
            str(recovery_replay),
            "--output",
            str(recovery_root),
            "--max-ticks",
            "100",
        ]
    ) == 0
    assert len(list(recovery_root.glob("*/accepted_plan.json"))) == 1


def test_pinned_source_fixture_records_license_and_verifies_hashes(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture.zip"
    with zipfile.ZipFile(fixture, "w") as bundle:
        for relative in REQUIRED_SOURCE_FILES:
            content = (
                "Redistribution and use in source and binary forms are permitted.\n"
                if relative == "LICENSE"
                else f"fixture {relative}\n"
            )
            bundle.writestr(f"BETR-XP-LLM-fixture/{relative}", content)
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()

    def downloader(_url: str, target: Path) -> None:
        target.write_bytes(fixture.read_bytes())

    prepared = prepare_official_source(
        tmp_path / "prepared",
        downloader=downloader,
        expected_sha256=digest,
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["software_license"] == "BSD-3-Clause"
    assert manifest["commit"] == "bf83bda4b8921eea7fe0b8756daacb7da9fb6133"
    assert manifest["archive_sha256"] == digest
    monkeypatch.setattr(
        "llm_mr_bt_planner.comparison.betr_xp_source.SOURCE_ARCHIVE_SHA256",
        digest,
    )
    assert verify_prepared_source(prepared.directory)["file_count"] == len(REQUIRED_SOURCE_FILES)


def test_provider_uses_paper_sampling_and_one_user_message(monkeypatch):
    captured: dict = {}

    def fake_send(request, timeout):  # noqa: ARG001
        captured.update(json.loads(request.data))
        return json.dumps(
            {
                "id": "chatcmpl-betrxp",
                "model": "gpt-4-1106-preview",
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                "choices": [{"message": {"content": "Goal: SystemReady"}}],
            }
        )

    monkeypatch.setattr("llm_mr_bt_planner.llm.openai_client._send", fake_send)
    caller = ProviderCaller("gpt-4-1106-preview", "test-only", seed=None)
    response = caller.generate("formalize this task", stage="goal_response")

    assert response.text == "Goal: SystemReady"
    assert captured["messages"] == [{"role": "user", "content": "formalize this task"}]
    assert captured["temperature"] == PAPER_TEMPERATURE
    assert captured["top_p"] == PAPER_TOP_P
    assert "response_format" not in captured
    assert response.metadata["total_tokens"] == 16
