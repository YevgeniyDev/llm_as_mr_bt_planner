from __future__ import annotations

import json
from pathlib import Path

from llm_mr_bt_planner.artifacts import load_plan_file
from llm_mr_bt_planner.cli import _build_parser
from llm_mr_bt_planner.config import PROJECT_ROOT
from llm_mr_bt_planner.domain import load_scenario
from llm_mr_bt_planner.llm import openai_client
from llm_mr_bt_planner.mujoco_sim.faults import load_fault_spec
from llm_mr_bt_planner.recovery import (
    OpenAIResponsesRecoveryClient,
    OracleRecoveryClient,
    plan_diff,
    plan_recovery,
    recovery_plan_json_schema,
)

SCENARIO_PATH = PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.json"
NOMINAL_PATH = PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.bt.json"
FAULT_PATH = PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.fault.json"
ORACLE_PATH = (
    PROJECT_ROOT / "examples" / "three_robot_spare_part_recovery.expected_recovery.bt.json"
)


def _measured_facts() -> tuple[str, ...]:
    return (
        "system_ready()",
        "robot_ready(franka_a)",
        "robot_ready(unitree_go2_z1)",
        "robot_ready(franka_b)",
        "usable(spare_part)",
        "at(spare_part,backup_bin)",
        "gripper_empty(franka_a)",
        "gripper_empty(unitree_go2_z1)",
        "gripper_empty(franka_b)",
        "arm_stowed(unitree_go2_z1)",
        "base_stationary(unitree_go2_z1)",
        "docked(unitree_go2_z1,source_dock)",
    )


def test_bundled_oracle_is_a_valid_continuation_from_the_failure_snapshot():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    nominal = load_plan_file(NOMINAL_PATH)

    result = plan_recovery(
        OracleRecoveryClient(ORACLE_PATH),
        scenario,
        measured_initial_state=_measured_facts(),
        failure_observation={"classification": "dropped_to_floor"},
        nominal_plan=nominal,
    )

    assert result.provider == "deterministic_oracle"
    assert result.validation.valid
    assert result.simulation.success
    assert len(result.attempts) == 1
    assert "usable(primary_part)" not in result.runtime_scenario.initial_state
    assert "at(primary_part, primary_bin)" not in result.runtime_scenario.initial_state
    assert "installed_component(target_fixture)" in result.simulation.final_state


def test_recovery_planner_retries_a_rejected_candidate_then_accepts_replacement():
    valid = load_plan_file(ORACLE_PATH).to_dict()

    class CorrectingClient:
        provider = "test"
        model = "test-model"
        reasoning_effort = "high"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _system: str, user: str):
            self.calls += 1
            if self.calls == 1:
                return {
                    "schema_version": "2.0",
                    "mission_id": "three_robot_spare_part_recovery",
                    "behavior_trees": {},
                }, {"attempt": 1}
            assert "previous recovery candidate was rejected" in user
            return valid, {"attempt": 2}

    client = CorrectingClient()
    progress: list[tuple[str, float]] = []
    result = plan_recovery(
        client,
        load_scenario(SCENARIO_PATH, strict=True),
        measured_initial_state=_measured_facts(),
        failure_observation={"classification": "dropped_to_floor"},
        nominal_plan=load_plan_file(NOMINAL_PATH),
        max_corrections=1,
        progress=lambda message, fraction: progress.append((message, fraction)),
    )

    assert client.calls == 2
    assert [attempt["accepted"] for attempt in result.attempts] == [False, True]
    assert any("provider response received" in message for message, _ in progress)
    assert any("rejected" in message for message, _ in progress)
    assert progress[-1] == ("Validated continuation BT is ready for MuJoCo", 1.0)


def test_openai_recovery_uses_responses_api_structured_output_and_sol(monkeypatch):
    plan_document = load_plan_file(ORACLE_PATH).to_dict()
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "id": "resp_test",
                    "model": "gpt-5.6-sol",
                    "status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json.dumps(plan_document)}
                            ],
                        }
                    ],
                }
            ).encode("utf-8")

    def urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(openai_client.urllib.request, "urlopen", urlopen)
    client = OpenAIResponsesRecoveryClient(api_key="test-only")
    document, provenance = client.complete("system", "user")

    payload = captured["payload"]
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["strict"] is True
    assert payload["store"] is False
    assert "messages" not in payload
    assert document == plan_document
    assert provenance["response_id"] == "resp_test"


def test_recovery_schema_requires_all_three_robot_trees():
    schema = recovery_plan_json_schema()
    behavior_trees = schema["properties"]["behavior_trees"]

    assert behavior_trees["required"] == ["franka_a", "unitree_go2_z1", "franka_b"]
    assert behavior_trees["additionalProperties"] is False
    assert schema["properties"]["mission_id"]["const"] == (
        "three_robot_spare_part_recovery"
    )


def test_plan_diff_exposes_primary_to_spare_adaptation():
    diff = plan_diff(load_plan_file(NOMINAL_PATH), load_plan_file(ORACLE_PATH))

    assert "-            \"primary_part\"" in diff
    assert "+            \"spare_part\"" in diff
    assert "backup_bin" in diff


def test_fault_contract_and_recovery_cli_defaults_are_explicit():
    fault = load_fault_spec(FAULT_PATH)
    args = _build_parser().parse_args(["recovery-experiment"])

    assert fault.fault_type == "drop_object"
    assert fault.fault_id == "drop_primary_after_handoff_placement"
    assert fault.trigger.object == "primary_part"
    assert fault.trigger.action == "place_source_cradle"
    assert fault.trigger.event == "action_success"
    assert fault.trigger.location == "source_cradle"
    assert fault.trigger.before_robot == "unitree_go2_z1"
    assert fault.trigger.before_action == "pick_source_cradle"
    assert fault.force_newtons == (0.0, 4.0, 0.0)
    assert args.planner == "openai"
    assert args.model == "gpt-5.6-sol"
    assert args.reasoning_effort == "high"
    assert args.no_video is False
    assert Path(args.oracle_bt) == ORACLE_PATH


def test_adaptive_demo_defaults_to_one_real_llm_video_workflow():
    args = _build_parser().parse_args(["adaptive-demo"])

    assert Path(args.scenario) == SCENARIO_PATH
    assert Path(args.fault) == FAULT_PATH
    assert args.model == "gpt-5.6-sol"
    assert args.reasoning_effort == "high"
    assert args.generation_max_corrections == 4
    assert args.recovery_max_corrections == 2
    assert args.no_video is False
    assert args.headless is False
    assert args.realtime_factor == 1.0
    assert not hasattr(args, "bt")
    assert not hasattr(args, "planner")
