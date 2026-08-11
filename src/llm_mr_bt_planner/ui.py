"""Local Gradio interface for the complete standalone planning pipeline."""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, NoReturn

from .config import PROJECT_ROOT
from .llm.base import redact_secrets
from .llm.catalog import default_model, is_catalog_model, model_choices
from .projects import ProjectStore
from .secrets import SecretStore
from .service import PlannerService

EXAMPLE_SCENARIO = PROJECT_ROOT / "examples" / "three_robot_courier.json"
SCENARIO_TEMPLATE = PROJECT_ROOT / "templates" / "three_robot_scenario.template.json"
APP_CSS = """
.gradio-container {
    box-sizing: border-box !important;
    width: min(1280px, calc(100vw - 40px)) !important;
    min-width: 0 !important;
    max-width: 1280px !important;
    margin: 0 auto !important;
}
#planner-hero { margin-bottom: 0.25rem; }
#planner-hero p { color: var(--body-text-color-subdued); max-width: 760px; }
#run-actions button { min-height: 48px; }
@media (max-width: 640px) {
    .gradio-container { width: calc(100vw - 24px) !important; }
}
"""


def build_app(
    service: PlannerService | None = None,
    secret_store: SecretStore | None = None,
    project_store: ProjectStore | None = None,
):
    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - depends on optional UI extra
        raise RuntimeError("The UI requires: pip install -e '.[ui]'") from error

    service = service or PlannerService(PROJECT_ROOT / "outputs" / "runs")
    secret_store = secret_store or SecretStore()
    project_store = project_store or ProjectStore(PROJECT_ROOT / "projects")
    initial_json = EXAMPLE_SCENARIO.read_text(encoding="utf-8")
    active_lock = threading.Lock()
    active_cancel: threading.Event | None = None

    def stop_with_error(title: str, error: BaseException | str) -> NoReturn:
        raise gr.Error(_explain_ui_error(error), title=title, duration=None, print_exception=False)

    def upload_scenario(path: str | None) -> tuple[str, str, str]:
        if not path:
            example = json.loads(initial_json)
            return initial_json, example["instruction"], "No file selected; loaded the bundled example."
        try:
            document = service.load_json(path)
            scenario = service.parse_scenario_document(document)
            return json.dumps(document, indent=2), scenario.instruction, f"Loaded and validated `{scenario.task_id}`."
        except Exception as error:
            stop_with_error("Scenario upload failed", error)

    def validate_scenario(text: str | None, instruction_text: str | None) -> str:
        try:
            document = _json_object(text, "scenario")
            document["instruction"] = _clean_text(instruction_text) or document.get("instruction", "")
            scenario = service.parse_scenario_document(document)
            capability_count = sum(len(robot.capabilities) for robot in scenario.robots)
            return (
                f"Scenario is valid: `{scenario.task_id}` | {len(scenario.robots)} robots | "
                f"{capability_count} capabilities | {len(scenario.goal_state)} goals."
            )
        except Exception as error:
            stop_with_error("Scenario validation failed", error)

    def save_project(
        name: str | None,
        text: str | None,
        instruction_text: str | None,
        provider_name: str | None,
        model_name: str | None,
        correction_limit: int,
        tick_limit: int,
    ):
        try:
            scenario_document = _json_object(text, "scenario")
            scenario_document["instruction"] = _clean_text(instruction_text) or scenario_document.get(
                "instruction", ""
            )
            resolved_name = _clean_text(name)
            if not resolved_name:
                raise ValueError("Project name is empty. Enter a name before saving the project.")
            resolved_provider = _clean_text(provider_name)
            if not resolved_provider:
                raise ValueError("AI provider is not selected. Choose OpenAI or Anthropic.")
            resolved_model = _clean_text(model_name) or default_model(resolved_provider)
            if resolved_model and not is_catalog_model(resolved_provider, resolved_model):
                raise ValueError(
                    f"The selected model `{resolved_model}` is not available for "
                    f"{_provider_label(resolved_provider)}. Select a model from the list."
                )
            path = project_store.save(
                resolved_name,
                scenario_document,
                {
                    "provider": resolved_provider,
                    "model": resolved_model,
                    "max_corrections": int(correction_limit),
                    "max_ticks": int(tick_limit),
                },
            )
            choices = project_store.list()
            return gr.Dropdown(choices=choices, value=path.stem), f"Saved project `{path.stem}`."
        except Exception as error:
            stop_with_error("Project was not saved", error)

    def load_project(name: str | None):
        if not name:
            stop_with_error("Project was not loaded", "Select a saved project first.")
        try:
            document = project_store.load(name)
            settings = document["settings"]
            resolved_provider = _clean_text(settings.get("provider")) or "openai"
            stored_model = _clean_text(settings.get("model"))
            resolved_model = (
                stored_model
                if stored_model and is_catalog_model(resolved_provider, stored_model)
                else default_model(resolved_provider)
            )
            model_note = (
                f" Stored model `{stored_model}` is no longer in the current "
                f"{_provider_label(resolved_provider)} list, so the provider default will be used."
                if stored_model and not is_catalog_model(resolved_provider, stored_model)
                else ""
            )
            return (
                json.dumps(document["scenario"], indent=2),
                document["scenario"]["instruction"],
                resolved_provider,
                gr.Dropdown(
                    choices=model_choices(resolved_provider),
                    value=resolved_model,
                    label="Model",
                    info=_model_info(resolved_provider),
                ),
                settings.get("max_corrections", 4),
                settings.get("max_ticks", 100),
                f"Loaded project `{name}`.{model_note}",
            )
        except Exception as error:
            stop_with_error("Project was not loaded", error)

    def load_key(provider: str | None) -> tuple[str, str]:
        try:
            resolved_provider = _clean_text(provider)
            if not resolved_provider:
                raise ValueError("AI provider is not selected. Choose OpenAI or Anthropic.")
            key = secret_store.load(resolved_provider)
            return "", (
                f"A saved {resolved_provider} key is available and will be used when the key field is blank."
                if key
                else f"No saved {resolved_provider} key is available."
            )
        except Exception as error:
            stop_with_error("Credential check failed", error)

    def forget_key(provider: str | None) -> tuple[str, str]:
        try:
            resolved_provider = _clean_text(provider)
            if not resolved_provider:
                raise ValueError("AI provider is not selected. Choose OpenAI or Anthropic.")
            secret_store.delete(resolved_provider)
            return "", f"Deleted the saved {resolved_provider} key."
        except Exception as error:
            stop_with_error("Credential deletion failed", error)

    def update_provider_options(provider_name: str | None):
        resolved_provider = _clean_text(provider_name)
        if not resolved_provider:
            stop_with_error("Provider was not changed", "Choose OpenAI or Anthropic.")
        return (
            gr.Dropdown(
                choices=model_choices(resolved_provider),
                value=default_model(resolved_provider),
                label="Model",
                info=_model_info(resolved_provider),
            ),
            "",
            f"Switched to {_provider_label(resolved_provider)}. The API key field was cleared for safety.",
        )

    def run_pipeline(
        text: str | None,
        instruction_text: str | None,
        provider: str | None,
        model: str | None,
        api_key: str | None,
        remember_key: bool,
        max_corrections: int,
        max_ticks: int,
        progress=gr.Progress(track_tqdm=False),  # noqa: B008 - Gradio injects this dependency
    ):
        nonlocal active_cancel
        started = time.monotonic()
        live_lines: list[str] = []
        validation_state: dict[str, Any] = {}
        simulation_state: dict[str, Any] = {}

        def append_log(message: str, fraction: float) -> str:
            elapsed = time.monotonic() - started
            live_lines.append(f"[{elapsed:8.2f}s] [{fraction:6.1%}] {message}")
            return "\n".join(live_lines)

        with active_lock:
            if active_cancel is not None:
                stop_with_error(
                    "Pipeline could not start",
                    "Another pipeline run is already active. Cancel it or wait for it to finish before starting again.",
                )
            cancel_event = threading.Event()
            active_cancel = cancel_event

        try:
            document = _json_object(text, "scenario")
            document["instruction"] = _clean_text(instruction_text) or document.get("instruction", "")
            resolved_provider = _clean_text(provider)
            if not resolved_provider:
                raise ValueError("AI provider is not selected. Choose OpenAI or Anthropic.")
            resolved_model = _clean_text(model) or default_model(resolved_provider)
            if resolved_model and not is_catalog_model(resolved_provider, resolved_model):
                raise ValueError(
                    f"The selected model `{resolved_model}` is not available for "
                    f"{_provider_label(resolved_provider)}. Select a model from the list."
                )
            resolved_api_key = _clean_text(api_key)
            if remember_key and resolved_api_key:
                secret_store.save(resolved_provider, resolved_api_key)
                append_log(
                    f"Saved the {resolved_provider} key in the operating-system credential store",
                    0.0,
                )
            if not resolved_api_key:
                resolved_api_key = secret_store.load(resolved_provider) or ""
                if resolved_api_key:
                    append_log(
                        f"Using the saved {resolved_provider} key without returning it to the browser",
                        0.0,
                    )
            if not resolved_api_key:
                raise ValueError(
                    f"No {resolved_provider.title()} API key was available. Enter a key in the API key "
                    "field, or save one in the operating-system credential store, then run again."
                )

            event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

            def update(message: str, fraction: float) -> None:
                event_queue.put(("progress", (message, fraction)))

            def worker() -> None:
                try:
                    outcome = service.generate(
                        document,
                        provider=resolved_provider,
                        api_key=resolved_api_key,
                        model=resolved_model,
                        max_corrections=int(max_corrections),
                        max_ticks=int(max_ticks),
                        progress=update,
                        cancelled=cancel_event.is_set,
                    )
                    event_queue.put(("complete", outcome))
                except Exception as error:
                    event_queue.put(("error", error))

            thread = threading.Thread(target=worker, name="lmrbtp-pipeline", daemon=True)
            thread.start()
            current_message = "Starting pipeline"
            last_heartbeat = time.monotonic()
            outcome = None
            while outcome is None:
                try:
                    event_type, payload = event_queue.get(timeout=0.25)
                except queue.Empty:
                    if not thread.is_alive():
                        raise RuntimeError("Pipeline worker stopped without returning a result.") from None
                    if time.monotonic() - last_heartbeat >= 1.0:
                        yield (
                            f"Running: {current_message} ({time.monotonic() - started:.1f}s elapsed)",
                            "\n".join(live_lines),
                            validation_state,
                            simulation_state,
                            [],
                        )
                        last_heartbeat = time.monotonic()
                    continue

                if event_type == "progress":
                    message, fraction = payload
                    current_message = str(message)
                    progress(float(fraction), desc=current_message)
                    log_text = append_log(current_message, float(fraction))
                    yield (
                        f"Running: {current_message}",
                        log_text,
                        validation_state,
                        simulation_state,
                        [],
                    )
                    continue
                if event_type == "error":
                    if isinstance(payload, BaseException):
                        raise payload
                    raise RuntimeError(str(payload))
                outcome = payload

            assert outcome is not None
            passed = outcome.validation.valid and outcome.simulation.success
            artifact_hash = None
            if outcome.artifacts.behavior_tree_json is not None:
                artifact_hash = json.loads(
                    outcome.artifacts.behavior_tree_json.read_text(encoding="utf-8")
                )["artifact_sha256"]
            status = (
                f"Pipeline {'passed' if passed else 'did not pass'} | task `{outcome.scenario.task_id}` | "
                f"provider/model `{outcome.planner_result.provider}/{outcome.planner_result.model}` | "
                f"corrections {outcome.planner_result.correction_rounds}"
                + (f" | SHA-256 `{artifact_hash}`." if artifact_hash else " | no final BT was published.")
            )
            validation = {
                "valid": outcome.validation.valid,
                "errors": outcome.validation.to_dicts(),
            }
            simulation = outcome.simulation.to_dict()
            append_log(
                "SUCCESS: final BT published" if passed else "FAILURE: diagnostics published without a final BT",
                1.0,
            )
            yield status, "\n".join(live_lines), validation, simulation, outcome.artifacts.download_paths()
            if not passed:
                validation_causes = [item["message"] for item in validation["errors"][:4]]
                simulation_causes = [
                    str(item.get("message") or item.get("type", "simulation error"))
                    for item in simulation.get("errors", [])[:4]
                ]
                causes = [*validation_causes, *simulation_causes]
                detail = "; ".join(causes) or "The declared symbolic goals were not reached."
                stop_with_error(
                    "Behavior Tree rejected",
                    f"The LLM-generated BT remained invalid after "
                    f"{outcome.planner_result.correction_rounds} correction round(s). {detail} "
                    "Diagnostic files were saved, but no final BT was published.",
                )
        except gr.Error:
            raise
        except Exception as error:
            safe_message = _explain_ui_error(error)
            log_text = append_log(f"FAILURE: {safe_message}", 1.0)
            yield (
                f"Pipeline stopped: {safe_message}",
                log_text,
                {"valid": False, "errors": [{"type": "pipeline_error", "message": safe_message}]},
                simulation_state,
                [],
            )
            stop_with_error("Pipeline stopped", error)
        finally:
            with active_lock:
                if active_cancel is cancel_event:
                    active_cancel = None

    def cancel_pipeline(current_log: str) -> tuple[str, str]:
        nonlocal active_cancel
        with active_lock:
            cancel_event = active_cancel
        if cancel_event is None:
            gr.Warning("There is no active pipeline run to cancel.", title="Nothing to cancel")
            return "No pipeline run is active.", current_log
        cancel_event.set()
        elapsed_line = "[cancel requested] The current provider response will be discarded before publication."
        return "Cancellation requested. No final BT will be published.", _append_text(current_log, elapsed_line)

    gradio_major = int(gr.__version__.split(".", 1)[0])
    blocks_options = {"css": APP_CSS} if gradio_major < 6 else {}
    with gr.Blocks(
        title="Multi-Robot Behavior Tree Planner",
        fill_width=True,
        **blocks_options,
    ) as app:
        gr.Markdown(
            "# Multi-Robot BT Planner\n"
            "Ask an LLM to construct complete multi-robot Behavior Trees, then validate and simulate them unchanged.",
            elem_id="planner-hero",
        )

        gr.Markdown("## 1. Choose a scenario")
        with gr.Row():
            scenario_file = gr.File(
                label="Scenario JSON file",
                file_types=[".json"],
                type="filepath",
                scale=3,
            )
            with gr.Column(scale=2, min_width=240):
                validate_button = gr.Button("Validate scenario", variant="secondary")
                with gr.Row():
                    gr.DownloadButton("Blank template", value=str(SCENARIO_TEMPLATE), size="sm")
                    gr.DownloadButton("Runnable example", value=str(EXAMPLE_SCENARIO), size="sm")
        scenario_status = gr.Markdown("Bundled three-robot courier example is ready.")
        with gr.Accordion("Advanced: edit scenario JSON", open=False):
            scenario_editor = gr.Code(
                value=initial_json,
                language="json",
                label="Scenario JSON (advanced)",
                lines=12,
            )

        gr.Markdown("## 2. Describe the mission")
        instruction = gr.Textbox(
            value=json.loads(initial_json)["instruction"],
            label="Mission instruction",
            lines=3,
            info="Overrides the instruction in the selected scenario for this run.",
        )
        with gr.Row():
            provider = gr.Dropdown(
                [("OpenAI", "openai"), ("Anthropic", "anthropic")],
                value="openai",
                label="Provider",
                scale=1,
            )
            api_key = gr.Textbox(
                value="",
                label="API key",
                type="password",
                placeholder="Used only for this run",
                scale=2,
            )
        credential_status = gr.Markdown("Keys are never written to project JSON or run artifacts.")
        with gr.Accordion("Advanced provider options", open=False):
            model = gr.Dropdown(
                choices=model_choices("openai"),
                value=default_model("openai"),
                label="Model",
                info=_model_info("openai"),
                allow_custom_value=False,
            )
            remember_key = gr.Checkbox(False, label="Save key in OS credential store")
            with gr.Row():
                load_key_button = gr.Button("Check saved key", size="sm")
                forget_key_button = gr.Button("Forget saved key", size="sm")

        gr.Markdown("## 3. Generate and verify")
        with gr.Accordion("Run settings", open=False):
            with gr.Row():
                max_corrections = gr.Slider(0, 8, value=4, step=1, label="Maximum correction rounds")
                max_ticks = gr.Slider(20, 300, value=100, step=10, label="Simulation tick limit")
        with gr.Row(elem_id="run-actions"):
            run_button = gr.Button("Run complete pipeline", variant="primary", scale=4)
            cancel_button = gr.Button("Cancel", variant="stop", scale=1)
        run_status = gr.Markdown("Ready.")
        downloads = gr.File(label="Generated artifacts", file_count="multiple")

        gr.Markdown("## 4. Inspect the result")
        with gr.Tabs():
            with gr.Tab("Live log"):
                live_log = gr.Textbox(
                    value="Ready. Pipeline events will appear here in real time.",
                    label="Live pipeline log",
                    lines=11,
                    max_lines=30,
                    interactive=False,
                    autoscroll=True,
                    buttons=["copy"],
                )
            with gr.Tab("Validation"):
                validation_output = gr.JSON(label="Static validation")
            with gr.Tab("Simulation"):
                simulation_output = gr.JSON(label="Contract simulation")

        with gr.Accordion("Saved projects", open=False):
            with gr.Row():
                project_name = gr.Textbox(
                    value="",
                    label="Project name",
                    placeholder="courier-demo",
                    scale=2,
                )
                save_project_button = gr.Button("Save current project", scale=1)
            with gr.Row():
                projects = gr.Dropdown(choices=project_store.list(), label="Saved project", scale=2)
                load_project_button = gr.Button("Load selected project", scale=1)

        scenario_file.upload(
            upload_scenario,
            inputs=scenario_file,
            outputs=[scenario_editor, instruction, scenario_status],
        )
        scenario_file.clear(
            upload_scenario,
            inputs=scenario_file,
            outputs=[scenario_editor, instruction, scenario_status],
        )
        validate_button.click(validate_scenario, inputs=[scenario_editor, instruction], outputs=scenario_status)
        save_project_button.click(
            save_project,
            inputs=[project_name, scenario_editor, instruction, provider, model, max_corrections, max_ticks],
            outputs=[projects, scenario_status],
        )
        load_project_button.click(
            load_project,
            inputs=projects,
            outputs=[scenario_editor, instruction, provider, model, max_corrections, max_ticks, scenario_status],
        )
        load_key_button.click(load_key, inputs=provider, outputs=[api_key, credential_status])
        forget_key_button.click(forget_key, inputs=provider, outputs=[api_key, credential_status])
        provider.input(
            update_provider_options,
            inputs=provider,
            outputs=[model, api_key, credential_status],
            queue=False,
            show_progress="hidden",
        )
        run_button.click(
            run_pipeline,
            inputs=[scenario_editor, instruction, provider, model, api_key, remember_key, max_corrections, max_ticks],
            outputs=[run_status, live_log, validation_output, simulation_output, downloads],
            show_progress="hidden",
            concurrency_limit=1,
            concurrency_id="planner-pipeline",
        )
        cancel_button.click(
            cancel_pipeline,
            inputs=live_log,
            outputs=[run_status, live_log],
            queue=False,
        )
    return app.queue(default_concurrency_limit=1)


def launch_ui(*, server_name: str = "127.0.0.1", server_port: int = 7860, inbrowser: bool = True) -> None:
    import gradio as gr

    app = build_app()
    launch_options = {"css": APP_CSS} if int(gr.__version__.split(".", 1)[0]) >= 6 else {}
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=inbrowser,
        show_error=True,
        **launch_options,
    )


def _clean_text(value: str | None) -> str:
    """Normalize an optional Gradio text value at the UI boundary."""
    return value.strip() if isinstance(value, str) else ""


def _provider_label(provider: str) -> str:
    return {"openai": "OpenAI", "anthropic": "Anthropic"}.get(provider, provider)


def _model_info(provider: str) -> str:
    return (
        f"Options follow the selected provider. Leave 'Provider default' selected to use "
        f"{default_model(provider)}."
    )


def _json_object(text: str | None, label: str) -> dict[str, Any]:
    normalized = _clean_text(text)
    if not normalized:
        raise ValueError(
            f"{label.capitalize()} JSON is empty. Upload a JSON file or restore the bundled example."
        )
    document = json.loads(normalized)
    if not isinstance(document, dict):
        raise ValueError(f"{label.capitalize()} JSON root must be an object.")
    return document


def _append_text(current: str | None, line: str) -> str:
    return f"{current.rstrip()}\n{line}" if current and current.strip() else line


def _explain_ui_error(error: BaseException | str) -> str:
    """Return a redacted, user-facing cause for every blocking UI failure."""
    if isinstance(error, json.JSONDecodeError):
        cause = (
            f"Invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}. "
            "Correct the JSON syntax and try again."
        )
    elif isinstance(error, AttributeError) and "NoneType" in str(error) and "strip" in str(error):
        cause = (
            "The interface received an empty optional text value before contacting the AI provider. "
            "This is an input-handling error, not an API-key rejection. Restart the updated app and try again."
        )
    else:
        cause = str(error).strip() or type(error).__name__
    return f"Cause: {redact_secrets(cause)}"
