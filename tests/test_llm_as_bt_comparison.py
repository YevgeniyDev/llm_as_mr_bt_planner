from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from llm_mr_bt_planner.cli import main as cli_main
from llm_mr_bt_planner.comparison.llm_as_bt import (
    ProviderGenerator,
    ReplayGenerator,
    run_llm_as_bt_planner,
)
from llm_mr_bt_planner.comparison.llm_as_bt_native import (
    KiosTreeError,
    action_sequence,
    native_forest_to_plan,
    parse_kios_tree,
    simulate_kios_tree,
)
from llm_mr_bt_planner.comparison.llm_as_bt_source import (
    REQUIRED_SOURCE_FILES,
    prepare_official_source,
)
from llm_mr_bt_planner.domain import Capability, Effects, Robot, Scenario, load_scenario
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.simulation import simulate
from llm_mr_bt_planner.validation import validate_plan

ROOT = Path(__file__).resolve().parents[1]
COURIER = ROOT / "examples" / "three_robot_courier.json"


def test_strict_kios_parser_native_simulator_and_nonrepairing_observer():
    scenario = _single_action_scenario()
    document = _unit("done()", "act(r1)")
    tree = parse_kios_tree(document)

    assert action_sequence(tree) == ["act(r1)"]
    native = simulate_kios_tree(tree, scenario, "r1", [])
    assert native.success and "done()" in native.world_state
    plan_document = native_forest_to_plan([("finish", "r1", tree)], scenario, wait_timeout_ticks=20)
    plan = parse_plan(plan_document)
    assert validate_plan(plan, scenario).valid
    assert simulate(plan, scenario).goal_success

    invalid = dict(document)
    invalid["invented"] = True
    with pytest.raises(KiosTreeError, match="unknown field"):
        parse_kios_tree(invalid)


def test_goal_guard_before_its_action_is_valid_behavior_tree_pattern():
    scenario = _single_action_scenario()
    tree = parse_kios_tree(_unit("done()", "act(r1)"))
    plan = parse_plan(
        native_forest_to_plan([("finish", "r1", tree)], scenario, wait_timeout_ticks=20)
    )
    report = validate_plan(plan, scenario)

    assert report.valid, report.to_dicts()
    assert not any(error.type == "condition_before_producer" for error in report.errors)


def test_one_step_courier_replay_passes_common_protocol_and_cli(tmp_path):
    scenario = load_scenario(COURIER, strict=True)
    responses = _courier_one_step_responses()
    result = run_llm_as_bt_planner(
        scenario,
        ReplayGenerator(responses),
        tmp_path / "runs",
        scheme="one-step",
        max_ticks=200,
    )

    assert result.plan_generation_success
    assert result.static_validity
    assert result.symbolic_goal_success
    assert result.accepted_plan is not None
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["method"]["official_executable_code_found"] is True
    assert manifest["method"]["scheme"] == "one-step"
    assert manifest["generator"]["real_model_inference"] is False
    assert manifest["fidelity"]["semantic_rewrites"] == []
    assert json.loads(result.metrics.read_text(encoding="utf-8"))["model_calls"] == 0

    response_file = tmp_path / "responses.json"
    response_file.write_text(json.dumps({"responses": responses}), encoding="utf-8")
    exit_code = cli_main(
        [
            "compare",
            "llm-as-bt-planner",
            "run",
            "--scenario",
            str(COURIER),
            "--responses",
            str(response_file),
            "--output",
            str(tmp_path / "cli"),
            "--max-ticks",
            "200",
        ]
    )
    assert exit_code == 0
    assert len(list((tmp_path / "cli").glob("*/manifest.json"))) == 1


def test_iterative_scheme_retries_with_native_failure_feedback(tmp_path):
    scenario = _single_action_scenario()
    bad = _unit("done()", "act(r1)", [_leaf("precondition", "done()")])
    good = _unit("done()", "act(r1)")
    responses = [
        _response("decompose", _decomposition("r1", "done()")),
        _response("iterative", _envelope(["act(r1)"], bad), attempt=1),
        _response("iterative", _envelope(["act(r1)"], good), attempt=2),
    ]
    result = run_llm_as_bt_planner(
        scenario,
        ReplayGenerator(responses),
        tmp_path,
        scheme="iterative",
        max_iterations=5,
    )

    assert result.symbolic_goal_success
    record = json.loads(
        (result.directory / "native" / "generation_record.json").read_text(encoding="utf-8")
    )["subgoals"][0]
    assert record["attempt_count"] == 2
    assert record["attempts"][0]["execution"]["result"] == "failure"
    second_prompt = sorted((result.directory / "prompts").glob("*iterative.txt"))[1]
    assert "behavior tree returned FAILURE" in second_prompt.read_text(encoding="utf-8")


def test_human_scheme_archives_every_feedback_revision(tmp_path):
    scenario = _single_action_scenario()
    tree = _unit("done()", "act(r1)")
    responses = [
        _response("decompose", _decomposition("r1", "done()")),
        _response(
            "sequential_plan",
            {"explanation": "one action", "task_plan": ["act(r1)"]},
        ),
        _response("human_tree", tree, attempt=1),
        _response("human_refine", tree, attempt=2),
    ]
    result = run_llm_as_bt_planner(
        scenario,
        ReplayGenerator(responses),
        tmp_path,
        scheme="human",
        human_feedback={"goal": ["Keep the explicit target guard."]},
    )

    assert result.symbolic_goal_success
    record = json.loads(
        (result.directory / "native" / "generation_record.json").read_text(encoding="utf-8")
    )["subgoals"][0]
    assert record["feedback_count"] == 1
    assert len(record["human_rounds"]) == 2
    assert record["human_rounds"][1]["feedback"] == "Keep the explicit target guard."


def test_recursive_scheme_runs_makeplan_maketree_predictstate(tmp_path):
    scenario = _single_action_scenario()
    responses = [
        _response("decompose", _decomposition("r1", "done()")),
        _response(
            "make_plan",
            {"explanation": "act establishes done", "task_plan": ["act(r1)"]},
            depth=0,
        ),
        _response("make_tree", _unit("done()", "act(r1)"), depth=0),
        _response(
            "predict_state",
            {"explanation": "apply act", "estimated_world_state": ["done()"]},
            depth=0,
        ),
    ]
    result = run_llm_as_bt_planner(
        scenario,
        ReplayGenerator(responses),
        tmp_path,
        scheme="recursive",
    )

    assert result.plan_generation_success and result.symbolic_goal_success
    calls = json.loads(result.manifest.read_text(encoding="utf-8"))["generator"]["calls"]
    assert [call["stage"] for call in calls] == [
        "decompose",
        "make_plan",
        "make_tree",
        "predict_state",
    ]


def test_openai_provider_uses_deterministic_json_mode_and_archives_usage(monkeypatch):
    captured: dict = {}

    def fake_send(request, timeout):  # noqa: ARG001
        captured.update(json.loads(request.data))
        return json.dumps(
            {
                "id": "chatcmpl-kios",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                "choices": [{"message": {"content": "{}"}}],
            }
        )

    monkeypatch.setattr("llm_mr_bt_planner.llm.openai_client._send", fake_send)
    result = ProviderGenerator("openai", "gpt-4", "test-only", seed=42).generate(
        "decompose", "system", "user", {}
    )

    assert result.text == "{}"
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42
    assert captured["max_tokens"] == 3000
    assert result.metadata["total_tokens"] == 20


def test_pinned_kios_source_fixture_is_hashed_and_safely_extracted(tmp_path):
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name in REQUIRED_SOURCE_FILES:
            content = "MIT License\n" if name == "LICENSE" else f"fixture {name}\n"
            bundle.writestr(f"kios-fixture/{name}", content)
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()

    def downloader(url: str, target: Path) -> None:  # noqa: ARG001
        target.write_bytes(archive.read_bytes())

    prepared = prepare_official_source(
        tmp_path / "prepared",
        downloader=downloader,
        expected_sha256=expected,
    )
    assert prepared.file_count == len(REQUIRED_SOURCE_FILES)
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    assert manifest["archive_sha256"] == expected
    assert manifest["software_license"] == "MIT"


def _courier_one_step_responses() -> list[dict]:
    subgoals = {
        "explanation": "sequential handoff, transfer, stow, and installation",
        "subgoals": [
            {
                "id": "source_handoff",
                "robot": "franka_a",
                "target": "at(payload, source_cradle)",
                "instruction": "place payload in source cradle",
            },
            {
                "id": "mobile_transfer",
                "robot": "unitree_go2_z1",
                "target": "at(payload, destination_cradle)",
                "instruction": "carry payload to destination cradle",
            },
            {
                "id": "stow_mobile_arm",
                "robot": "unitree_go2_z1",
                "target": "arm_stowed(unitree_go2_z1)",
                "instruction": "stow the mobile arm",
            },
            {
                "id": "install_target",
                "robot": "franka_b",
                "target": "installed(payload, target_fixture)",
                "instruction": "install payload",
            },
        ],
    }
    source_pick = _unit(
        "holding(franka_a, payload)",
        "pick_source(franka_a, payload)",
        [
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(franka_a)"),
            _leaf("precondition", "at(payload, source_bin)"),
            _leaf("precondition", "gripper_empty(franka_a)"),
        ],
    )
    source = _unit(
        "at(payload, source_cradle)",
        "place_source_cradle(franka_a, payload)",
        [
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(franka_a)"),
            source_pick,
        ],
    )
    mobile_pick = _unit(
        "holding(unitree_go2_z1, payload)",
        "pick_source_cradle(unitree_go2_z1, payload)",
        [
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(unitree_go2_z1)"),
            _leaf("precondition", "base_stationary(unitree_go2_z1)"),
            _leaf("precondition", "arm_stowed(unitree_go2_z1)"),
            _leaf("precondition", "docked(unitree_go2_z1, source_dock)"),
            _leaf("precondition", "at(payload, source_cradle)"),
            _leaf("precondition", "gripper_empty(unitree_go2_z1)"),
        ],
    )
    navigate = _unit(
        "docked(unitree_go2_z1, destination_dock)",
        "navigate_destination(unitree_go2_z1, payload)",
        [
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(unitree_go2_z1)"),
            _leaf("precondition", "base_stationary(unitree_go2_z1)"),
            _leaf("precondition", "holding(unitree_go2_z1, payload)"),
            _leaf("precondition", "docked(unitree_go2_z1, source_dock)"),
        ],
    )
    transfer = _unit(
        "at(payload, destination_cradle)",
        "place_destination_cradle(unitree_go2_z1, payload)",
        [
            mobile_pick,
            navigate,
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(unitree_go2_z1)"),
            _leaf("precondition", "base_stationary(unitree_go2_z1)"),
            _leaf("precondition", "holding(unitree_go2_z1, payload)"),
        ],
    )
    stow = _unit(
        "arm_stowed(unitree_go2_z1)",
        "stow_arm_destination(unitree_go2_z1, payload)",
        [
            _leaf("precondition", "robot_ready(unitree_go2_z1)"),
            _leaf("precondition", "base_stationary(unitree_go2_z1)"),
            _leaf("precondition", "at(payload, destination_cradle)"),
            _leaf("precondition", "gripper_empty(unitree_go2_z1)"),
        ],
    )
    destination_pick = _unit(
        "holding(franka_b, payload)",
        "pick_destination_cradle(franka_b, payload)",
        [
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(franka_b)"),
            _leaf("precondition", "at(payload, destination_cradle)"),
            _leaf("precondition", "gripper_empty(franka_b)"),
        ],
    )
    install = _unit(
        "installed(payload, target_fixture)",
        "install_target(franka_b, payload, target_fixture)",
        [
            _leaf("precondition", "system_ready()"),
            _leaf("precondition", "robot_ready(franka_b)"),
            destination_pick,
        ],
    )
    return [
        _response("decompose", subgoals),
        _response(
            "one_step",
            _envelope(
                [
                    "pick_source(franka_a, payload)",
                    "place_source_cradle(franka_a, payload)",
                ],
                source,
            ),
        ),
        _response(
            "one_step",
            _envelope(
                [
                    "pick_source_cradle(unitree_go2_z1, payload)",
                    "navigate_destination(unitree_go2_z1, payload)",
                    "place_destination_cradle(unitree_go2_z1, payload)",
                ],
                transfer,
            ),
        ),
        _response(
            "one_step",
            _envelope(["stow_arm_destination(unitree_go2_z1, payload)"], stow),
        ),
        _response(
            "one_step",
            _envelope(
                [
                    "pick_destination_cradle(franka_b, payload)",
                    "install_target(franka_b, payload, target_fixture)",
                ],
                install,
            ),
        ),
    ]


def _response(stage: str, response: dict, **context: object) -> dict:
    return {"stage": stage, "response": response, **context}


def _decomposition(robot: str, target: str) -> dict:
    return {
        "explanation": "one grounded subgoal",
        "subgoals": [
            {"id": "goal", "robot": robot, "target": target, "instruction": "achieve goal"}
        ],
    }


def _envelope(actions: list[str], tree: dict) -> dict:
    return {"thought": "grounded plan", "action_sequence": actions, "behavior_tree": tree}


def _leaf(kind: str, body: str) -> dict:
    return {"summary": body, "name": f"{kind}: {body}"}


def _unit(target: str, action: str, prerequisites: list[dict] | None = None) -> dict:
    return {
        "summary": f"achieve {target}",
        "name": f"selector: achieve {target}",
        "children": [
            _leaf("target", target),
            {
                "summary": f"execute {action}",
                "name": f"sequence: execute {action}",
                "children": [*(prerequisites or []), _leaf("action", action)],
            },
        ],
    }


def _single_action_scenario() -> Scenario:
    return Scenario(
        task_id="single",
        instruction="make done true",
        initial_state=(),
        goal_state=("done()",),
        objects=(),
        locations=(),
        robots=(
            Robot(
                "r1",
                "R1",
                "robot",
                (Capability("act", (), (), Effects(add=("done()",))),),
            ),
        ),
    )
