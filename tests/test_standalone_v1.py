from __future__ import annotations

import json
import threading
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from llm_mr_bt_planner.artifacts import build_bt_artifact, load_plan_file, verify_artifact
from llm_mr_bt_planner.bt import iter_leaves, iter_nodes
from llm_mr_bt_planner.cli import main as cli_main
from llm_mr_bt_planner.domain import ScenarioError, load_scenario, parse_scenario
from llm_mr_bt_planner.llm.anthropic_client import AnthropicClient
from llm_mr_bt_planner.llm.base import LLMError
from llm_mr_bt_planner.llm.catalog import default_model, is_catalog_model, model_choices
from llm_mr_bt_planner.llm.openai_client import OpenAIClient
from llm_mr_bt_planner.plan import parse_plan
from llm_mr_bt_planner.planner import PlanningCancelled
from llm_mr_bt_planner.projects import ProjectStore
from llm_mr_bt_planner.prompts import build_prompt
from llm_mr_bt_planner.service import PlannerService
from llm_mr_bt_planner.simulation import simulate
from llm_mr_bt_planner.validation import validate_plan

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "examples" / "three_robot_courier.json"
BT_PATH = ROOT / "examples" / "three_robot_courier.bt.json"
PACKAGING_SCENARIO_PATH = ROOT / "examples" / "three_robot_packaging_delivery.json"
PACKAGING_BT_PATH = ROOT / "examples" / "three_robot_packaging_delivery.bt.json"


@pytest.fixture
def courier():
    scenario = load_scenario(SCENARIO_PATH, strict=True)
    plan_document = json.loads(BT_PATH.read_text(encoding="utf-8"))
    return scenario, parse_plan(plan_document)


@pytest.fixture
def packaging():
    scenario = load_scenario(PACKAGING_SCENARIO_PATH, strict=True)
    plan_document = json.loads(PACKAGING_BT_PATH.read_text(encoding="utf-8"))
    return scenario, parse_plan(plan_document)


def test_strict_schema_rejects_unknown_fields():
    document = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    document["execute_this"] = "not allowed"
    with pytest.raises(ScenarioError, match="Additional properties"):
        parse_scenario(document, strict=True)


def test_reference_is_a_complete_llm_authored_bt_without_compiler_nodes(courier):
    scenario, plan = courier
    assert set(plan.behavior_trees) == scenario.robot_ids
    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    leaves = [leaf for tree in plan.behavior_trees.values() for leaf in iter_leaves(tree)]
    assert sum(leaf.type == "WaitFor" for leaf in leaves) == 2
    assert sum(leaf.type == "AcquireResource" for leaf in leaves) == 4
    assert sum(leaf.type == "ReleaseResource" for leaf in leaves) == 4
    assert all(node.source == "llm" for node in nodes)
    assert all(leaf.task_id for leaf in leaves if leaf.type == "Action")
    assert parse_plan(plan.to_dict()).to_dict() == plan.to_dict()


def test_prompt_requires_direct_complete_bt_and_forbids_downstream_repair(courier):
    scenario, _ = courier
    prompt = build_prompt(scenario)
    assert '"behavior_trees"' in prompt
    assert '"schema_version": "2.0"' in prompt
    assert "No downstream component will insert, reorder, or repair nodes" in prompt
    assert "action_sequences" not in prompt


def test_reference_contract_validates_and_simulates_with_real_async_and_resource_events(courier):
    scenario, plan = courier
    validation = validate_plan(plan, scenario)
    assert validation.valid, validation.to_dicts()
    report = simulate(plan, scenario, max_ticks=100)
    assert report.success and report.goal_success and report.errors == []
    events = {event["event"] for event in report.trace}
    assert {"action_running", "wait_satisfied", "resource_acquired", "resource_released"} <= events


def test_packaging_reference_coordinates_assembly_door_crossing_and_delivery(packaging):
    scenario, plan = packaging
    validation = validate_plan(plan, scenario)
    assert validation.valid, validation.to_dicts()

    nodes = [node for tree in plan.behavior_trees.values() for node in iter_nodes(tree)]
    assert sum(node.type == "Fallback" for node in nodes) == 1
    assert sum(node.type == "WaitFor" for node in nodes) == 3
    assert sum(node.type == "AcquireResource" for node in nodes) == 6
    assert sum(node.type == "ReleaseResource" for node in nodes) == 6

    report = simulate(plan, scenario, max_ticks=140)
    assert report.success and report.goal_success and report.errors == []
    actions = [event.get("name") for event in report.trace if event.get("event") == "action"]
    assert "pick_loaded_package_base" in actions
    assert "pick_package_lid" in actions
    assert "fit_and_seal_package_lid" in actions
    assert "push_open_door_and_cross" in actions
    assert "cross_already_open_door" not in actions
    assert "place_parcel_at_delivery_station" in actions
    assert set(scenario.goal_state) <= set(report.final_state)


def test_packaging_scenario_runs_the_direct_llm_generation_pipeline(tmp_path):
    class _PackagingReferenceClient:
        name = "test-reference-client"
        model = "committed-packaging-proposal"

        @staticmethod
        def complete(system: str, user: str) -> str:  # noqa: ARG004
            return PACKAGING_BT_PATH.read_text(encoding="utf-8")

    service = PlannerService(tmp_path, client_factory=_factory(_PackagingReferenceClient()))
    outcome = service.generate(
        service.load_json(PACKAGING_SCENARIO_PATH),
        provider="openai",
        api_key="test-only",
        max_corrections=0,
        max_ticks=140,
    )

    assert outcome.scenario.task_id == "three_robot_packaging_delivery"
    assert outcome.validation.valid
    assert outcome.simulation.success
    assert outcome.artifacts.behavior_tree_json is not None
    generated = load_plan_file(outcome.artifacts.behavior_tree_json)
    assert generated.to_dict() == json.loads(PACKAGING_BT_PATH.read_text(encoding="utf-8"))


def test_cancellation_releases_owned_resources(courier):
    scenario, plan = courier
    report = simulate(plan, scenario, max_ticks=100, cancel_at_tick=2)
    assert not report.success
    assert any(error["type"] == "cancelled" for error in report.errors)
    assert not any(error["type"] == "resource_leak" for error in report.errors)
    assert any(event["event"] == "resource_released_on_cancel" for event in report.trace)


def test_artifact_hash_is_stable_and_verifiable(courier):
    scenario, plan = courier
    validation = validate_plan(plan, scenario)
    simulation = simulate(plan, scenario)
    kwargs = dict(
        provider="test-reference-client",
        model="committed-proposal",
        correction_rounds=0,
        validation=validation,
        simulation=simulation,
    )
    first = build_bt_artifact(plan, scenario, **kwargs)
    second = build_bt_artifact(plan, scenario, **kwargs)
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert verify_artifact(first)
    first["plan"]["mission_id"] = "tampered"
    assert not verify_artifact(first)


class _ReferenceClient:
    name = "test-reference-client"
    model = "committed-proposal"

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return BT_PATH.read_text(encoding="utf-8")


class _EmptyClient:
    name = "test-empty-client"
    model = "empty-proposal"

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        return json.dumps(
            {
                "schema_version": "2.0",
                "mission_id": "three_robot_courier",
                "behavior_trees": {
                    robot: {"id": f"{robot}.root", "type": "Sequence", "source": "llm", "children": []}
                    for robot in ("franka_a", "unitree_go2_z1", "franka_b")
                },
            }
        )


class _UnexpectedFieldClient(_ReferenceClient):
    name = "test-unexpected-field-client"

    def complete(self, system: str, user: str) -> str:
        document = json.loads(super().complete(system, user))
        document["behavior_trees"]["franka_a"]["compiler_inserted"] = True
        return json.dumps(document)


def _factory(client):
    return lambda provider, **kwargs: client


def test_shared_service_writes_final_bundle_without_leaking_key(tmp_path):
    service = PlannerService(tmp_path, client_factory=_factory(_ReferenceClient()))
    progress_events: list[tuple[str, float]] = []
    outcome = service.generate(
        service.load_json(SCENARIO_PATH),
        provider="openai",
        api_key="test-secret-never-persist",
        max_corrections=0,
        progress=lambda message, fraction: progress_events.append((message, fraction)),
    )
    assert outcome.artifacts.behavior_tree_json is not None
    assert outcome.artifacts.behavior_tree_xml is not None
    assert all(Path(path).exists() for path in outcome.artifacts.download_paths())
    assert outcome.artifacts.pipeline_log.exists()
    messages = [message for message, _ in progress_events]
    assert any("sending request" in message for message in messages)
    assert any("response received" in message for message in messages)
    assert any("static validation passed" in message for message in messages)
    assert any("contract simulation passed" in message.lower() for message in messages)
    assert any("publication is authorized" in message for message in messages)
    persisted_log = outcome.artifacts.pipeline_log.read_text(encoding="utf-8")
    assert "provider response received" in persisted_log
    assert "test-secret-never-persist" not in persisted_log
    round_tripped = load_plan_file(outcome.artifacts.behavior_tree_json)
    assert round_tripped.to_dict() == outcome.plan.to_dict()
    assert outcome.plan.to_dict() == json.loads(BT_PATH.read_text(encoding="utf-8"))
    artifact = json.loads(outcome.artifacts.behavior_tree_json.read_text(encoding="utf-8"))
    assert artifact["provenance"]["generation_method"] == "direct_llm_behavior_tree"
    assert artifact["provenance"]["semantic_rewrites"] == []
    assert (
        cli_main(
            [
                "validate",
                "--scenario",
                str(SCENARIO_PATH),
                "--bt",
                str(outcome.artifacts.behavior_tree_json),
                "--output",
                str(tmp_path / "independent-validation.json"),
            ]
        )
        == 0
    )
    assert (
        cli_main(
            [
                "simulate",
                "--scenario",
                str(SCENARIO_PATH),
                "--bt",
                str(outcome.artifacts.behavior_tree_json),
                "--output",
                str(tmp_path / "independent-simulation.json"),
            ]
        )
        == 0
    )
    all_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in outcome.artifacts.directory.iterdir()
        if path.is_file()
    )
    assert "test-secret-never-persist" not in all_text


def test_cancellation_after_provider_response_prevents_artifact_publication(tmp_path):
    cancelled = threading.Event()

    class _CancellingClient(_ReferenceClient):
        def complete(self, system: str, user: str) -> str:
            response = super().complete(system, user)
            cancelled.set()
            return response

    service = PlannerService(tmp_path, client_factory=_factory(_CancellingClient()))
    with pytest.raises(PlanningCancelled, match="no final BT"):
        service.generate(
            service.load_json(SCENARIO_PATH),
            provider="openai",
            api_key="test-only",
            cancelled=cancelled.is_set,
        )
    assert not list(tmp_path.rglob("behavior_tree.json"))


def test_failed_pipeline_writes_diagnostics_but_no_final_bt(tmp_path):
    service = PlannerService(tmp_path, client_factory=_factory(_EmptyClient()))
    outcome = service.generate(
        service.load_json(SCENARIO_PATH),
        provider="anthropic",
        api_key="test-only",
        max_corrections=0,
    )
    assert not outcome.validation.valid
    assert outcome.artifacts.behavior_tree_json is None
    assert not (outcome.artifacts.directory / "behavior_tree.json").exists()
    assert outcome.artifacts.validation_report.exists()


def test_unknown_llm_fields_survive_independent_recheck_and_block_publication(tmp_path):
    service = PlannerService(tmp_path, client_factory=_factory(_UnexpectedFieldClient()))
    outcome = service.generate(
        service.load_json(SCENARIO_PATH),
        provider="openai",
        api_key="test-only",
        max_corrections=0,
    )
    assert not outcome.validation.valid
    assert "unknown_node_field" in {error.type for error in outcome.validation.errors}
    assert outcome.artifacts.behavior_tree_json is None
    result = json.loads(outcome.artifacts.result.read_text(encoding="utf-8"))
    assert result["plan"]["behavior_trees"]["franka_a"]["compiler_inserted"] is True


def test_saved_project_keeps_preferences_but_never_accepts_a_key(tmp_path):
    store = ProjectStore(tmp_path)
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    path = store.save(
        "demo",
        scenario,
        {"provider": "anthropic", "model": "model-name", "api_key": "must-not-save"},
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["settings"] == {"provider": "anthropic", "model": "model-name"}
    assert "must-not-save" not in path.read_text(encoding="utf-8")


def test_current_provider_model_catalog_is_explicit_and_uses_current_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert default_model("openai") == "gpt-5.6-sol"
    assert default_model("anthropic") == "claude-opus-5"
    assert OpenAIClient(api_key="test-only").model == "gpt-5.6-sol"
    assert AnthropicClient(api_key="test-only").model == "claude-opus-5"
    assert model_choices("openai")[0] == (
        "Provider default — gpt-5.6-sol",
        "gpt-5.6-sol",
    )
    assert is_catalog_model("openai", "gpt-5.6-terra")
    assert is_catalog_model("anthropic", "claude-sonnet-5")
    assert not is_catalog_model("anthropic", "gpt-5.6-sol")


def test_gpt5_request_omits_unsupported_temperature(monkeypatch):
    from llm_mr_bt_planner.llm import openai_client

    requests: list[dict] = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def read() -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    def urlopen(request, timeout):  # noqa: ARG001
        requests.append(json.loads(request.data))
        return _Response()

    monkeypatch.setattr(openai_client.urllib.request, "urlopen", urlopen)
    client = OpenAIClient(model="gpt-5.6-sol", api_key="test-only", temperature=0)

    assert client.complete("system", "user") == "{}"
    assert client.temperature is None
    assert requests == [
        {
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "response_format": {"type": "json_object"},
        }
    ]


def test_openai_http_error_extracts_readable_message(monkeypatch):
    from llm_mr_bt_planner.llm import openai_client

    detail = json.dumps(
        {
            "error": {
                "message": "Unsupported value: temperature must use the default.",
                "param": "temperature",
                "type": "invalid_request_error",
            }
        }
    ).encode()

    def urlopen(request, timeout):  # noqa: ARG001
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            BytesIO(detail),
        )

    monkeypatch.setattr(openai_client.urllib.request, "urlopen", urlopen)

    with pytest.raises(LLMError) as captured:
        OpenAIClient(model="gpt-5.6-sol", api_key="test-only").complete("system", "user")

    message = str(captured.value)
    assert message == (
        "OpenAI API request failed (HTTP 400): "
        "Unsupported value: temperature must use the default."
    )
    assert '"error"' not in message


def test_gradio_app_builds_without_ros_or_mujoco_imports():
    pytest.importorskip("gradio")
    from llm_mr_bt_planner.ui import build_app

    app = build_app()
    assert app is not None
    config = app.get_config_file()
    assert config["fill_width"] is True
    components = config.get("components", [])
    labels = {component.get("props", {}).get("label") for component in components}
    assert "Live pipeline log" in labels
    assert "Scenario JSON (advanced)" in labels
    assert "Model" in labels
    assert "Model override" not in labels
    assert "Bundled scenario" in labels

    accordions = {
        component.get("props", {}).get("label"): component.get("props", {})
        for component in components
        if component.get("type") == "accordion"
    }
    assert accordions["Advanced: edit scenario JSON"]["open"] is False
    assert accordions["Advanced provider options"]["open"] is False
    assert accordions["Run settings"]["open"] is False
    assert accordions["Saved projects"]["open"] is False

    scenario_editor = next(
        component
        for component in components
        if component.get("props", {}).get("label") == "Scenario JSON (advanced)"
    )
    assert scenario_editor["props"]["lines"] == 12

    model_picker = next(
        component for component in components if component.get("props", {}).get("label") == "Model"
    )
    assert model_picker["type"] == "dropdown"
    assert model_picker["props"]["allow_custom_value"] is False
    assert ("Provider default — gpt-5.6-sol", "gpt-5.6-sol") in model_picker["props"][
        "choices"
    ]

    provider_dependency = next(
        dependency
        for dependency in config.get("dependencies", [])
        if dependency.get("api_name") == "update_provider_options"
    )
    assert provider_dependency["show_progress"] == "hidden"

    run_dependency = next(
        dependency
        for dependency in config.get("dependencies", [])
        if dependency.get("api_name") == "run_pipeline"
    )
    assert run_dependency["show_progress"] == "hidden"


def test_ui_provider_switch_replaces_models_and_clears_previous_key():
    pytest.importorskip("gradio")
    from llm_mr_bt_planner.ui import build_app

    app = build_app()
    switch_function = next(
        block.fn for block in app.fns.values() if block.fn.__name__ == "update_provider_options"
    )
    model_update, api_key, status = switch_function("anthropic")

    assert model_update.value == "claude-opus-5"
    assert ("Provider default — claude-opus-5", "claude-opus-5") in (
        model_update.choices
    )
    assert all(not value.startswith("gpt-") for _, value in model_update.choices)
    assert api_key == ""
    assert "cleared for safety" in status


def test_ui_blocking_error_explains_json_location_and_redacts_keys():
    from llm_mr_bt_planner.ui import _explain_ui_error

    json_error = json.JSONDecodeError("Expecting property name", '{"bad": }', 8)
    message = _explain_ui_error(json_error)
    assert "Cause: Invalid JSON" in message
    assert "line 1, column 9" in message
    assert "Correct the JSON syntax" in message
    assert "sk-proj-secret-value" not in _explain_ui_error("request failed for sk-proj-secret-value")

    empty_text_error = AttributeError("'NoneType' object has no attribute 'strip'")
    empty_text_message = _explain_ui_error(empty_text_error)
    assert "empty optional text value" in empty_text_message
    assert "not an API-key rejection" in empty_text_message
    assert "NoneType" not in empty_text_message


def test_ui_accepts_entered_key_when_optional_model_is_none(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    from llm_mr_bt_planner.ui import build_app

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    received: dict[str, object] = {}

    def factory(provider, **kwargs):
        received.update(provider=provider, **kwargs)
        return _ReferenceClient()

    service = PlannerService(tmp_path, client_factory=factory)
    app = build_app(service=service)
    run_function = next(block.fn for block in app.fns.values() if block.fn.__name__ == "run_pipeline")
    scenario_text = SCENARIO_PATH.read_text(encoding="utf-8")
    instruction = json.loads(scenario_text)["instruction"]

    updates = list(
        run_function(
            scenario_text,
            instruction,
            "openai",
            None,
            "test-only-key",
            False,
            0,
            100,
        )
    )

    assert updates[-1][0].startswith("Pipeline passed")
    assert updates[-1][2]["valid"] is True
    assert updates[-1][3]["success"] is True
    assert updates[-1][4]
    assert received["model"] == "gpt-5.6-sol"


def test_ui_pipeline_failure_updates_outputs_then_raises_explanatory_popup(tmp_path):
    gr = pytest.importorskip("gradio")
    from llm_mr_bt_planner.ui import build_app

    class _FailingClient:
        name = "test-failing-client"
        model = "deliberate-failure"

        def complete(self, system: str, user: str) -> str:  # noqa: ARG002
            raise RuntimeError("deliberate provider failure")

    service = PlannerService(tmp_path, client_factory=_factory(_FailingClient()))
    app = build_app(service=service)
    run_function = next(block.fn for block in app.fns.values() if block.fn.__name__ == "run_pipeline")
    scenario_text = SCENARIO_PATH.read_text(encoding="utf-8")
    instruction = json.loads(scenario_text)["instruction"]
    updates = []
    with pytest.raises(gr.Error, match="Cause: deliberate provider failure"):
        updates.extend(
            run_function(
                scenario_text,
                instruction,
                "openai",
                "",
                "test-only-key",
                False,
                0,
                100,
            )
        )
    assert updates
    assert updates[-1][0] == "Pipeline stopped: Cause: deliberate provider failure"
    assert "FAILURE: Cause: deliberate provider failure" in updates[-1][1]


def test_ui_rejected_bt_explains_why_final_publication_stopped(tmp_path):
    gr = pytest.importorskip("gradio")
    from llm_mr_bt_planner.ui import build_app

    service = PlannerService(tmp_path, client_factory=_factory(_EmptyClient()))
    app = build_app(service=service)
    run_function = next(block.fn for block in app.fns.values() if block.fn.__name__ == "run_pipeline")
    scenario_text = SCENARIO_PATH.read_text(encoding="utf-8")
    instruction = json.loads(scenario_text)["instruction"]
    updates = []
    with pytest.raises(gr.Error, match="LLM-generated BT remained invalid after 0 correction round"):
        updates.extend(
            run_function(
                scenario_text,
                instruction,
                "anthropic",
                "",
                "test-only-key",
                False,
                0,
                100,
            )
        )
    assert "Pipeline did not pass" in updates[-1][0]
    assert updates[-1][2]["valid"] is False
    assert updates[-1][4]
