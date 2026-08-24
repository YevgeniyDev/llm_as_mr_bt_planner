"""Command-line frontend for the same standalone pipeline used by the UI."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .artifacts import load_plan_file
from .config import PROJECT_ROOT, load_dotenv, save_json, save_text
from .domain import load_scenario
from .secrets import SecretStore
from .service import PlannerService
from .simulation import simulate
from .validation import validate_plan
from .viz import plan_to_html
from .xml_export import export_behaviortree_cpp_xml

DEFAULT_SCENARIO = PROJECT_ROOT / "examples" / "three_robot_courier.json"
PACKAGING_SCENARIO = PROJECT_ROOT / "examples" / "three_robot_packaging_delivery.json"
RECOVERY_SCENARIO = PROJECT_ROOT / "examples" / "three_robot_component_installation.json"
RECOVERY_BT = PROJECT_ROOT / "examples" / "three_robot_component_installation.bt.json"
RECOVERY_FAULT = PROJECT_ROOT / "examples" / "three_robot_component_installation.fault.json"
RECOVERY_ORACLE_BT = (
    PROJECT_ROOT / "examples" / "three_robot_component_installation.expected_recovery.bt.json"
)
DEFAULT_TEMPLATE = PROJECT_ROOT / "templates" / "three_robot_scenario.template.json"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = _build_parser()
    command_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(command_argv)
    args.invocation = [parser.prog, *command_argv]
    try:
        return int(args.func(args))
    except Exception as error:
        from .llm.base import redact_secrets

        print(f"error: {redact_secrets(str(error))}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lmrbtp",
        description="Generate and verify synchronized multi-robot Behavior Trees.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", aliases=["run"], help="Run generation, validation, simulation, and export.")
    generate.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    generate.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    generate.add_argument("--model", default=None)
    generate.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "runs"))
    generate.add_argument("--max-corrections", type=int, default=4)
    generate.add_argument("--max-ticks", type=int, default=100)
    generate.add_argument(
        "--use-saved-key",
        action="store_true",
        help="Use the provider key explicitly saved by the UI in the OS credential store.",
    )
    generate.set_defaults(func=_cmd_generate)

    validate = sub.add_parser("validate", help="Statically validate an existing BT JSON artifact.")
    _add_scenario_bt_args(validate)
    validate.add_argument("--output", default=None)
    validate.set_defaults(func=_cmd_validate)

    simulate_parser = sub.add_parser("simulate", help="Validate and run deterministic contract simulation.")
    _add_scenario_bt_args(simulate_parser)
    simulate_parser.add_argument("--max-ticks", type=int, default=100)
    simulate_parser.add_argument("--output", default=None)
    simulate_parser.set_defaults(func=_cmd_simulate)

    render = sub.add_parser("render", help="Render BT JSON as HTML and/or BT.CPP XML.")
    render.add_argument("--bt", required=True)
    render.add_argument("--html", default="outputs/rendered-bt.html")
    render.add_argument("--xml", default=None)
    render.set_defaults(func=_cmd_render)

    template = sub.add_parser("template", help="Copy the strict scenario template to a chosen path.")
    template.add_argument("--output", default="scenario.template.json")
    template.set_defaults(func=_cmd_template)

    doctor = sub.add_parser("doctor", help="Check dependencies and the bundled reference scenario/tree.")
    doctor.set_defaults(func=_cmd_doctor)

    ui = sub.add_parser("ui", help="Launch the local Gradio interface.")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    ui.add_argument("--no-browser", action="store_true", help="Do not open the local UI in a browser.")
    ui.set_defaults(func=_cmd_ui)

    physical = sub.add_parser(
        "mujoco",
        help="Execute a supported BT against physical controllers in a separate MuJoCo process.",
    )
    physical.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    physical.add_argument(
        "--bt",
        default=None,
        help="BT JSON. If omitted, use the .bt.json beside the selected bundled scenario.",
    )
    physical.add_argument("--assets-dir", default=None, help="Existing or target MuJoCo Menagerie cache.")
    physical.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "mujoco"))
    physical.add_argument("--headless", action="store_true", help="Run without opening the MuJoCo viewer.")
    physical.add_argument("--setup-only", action="store_true", help="Download and verify pinned robot assets, then exit.")
    physical.add_argument("--max-seconds", type=float, default=150.0)
    physical.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Viewer playback speed; ignored in headless mode.",
    )
    physical.add_argument(
        "--record-video",
        action="store_true",
        help="Record the complete simulation, including settling, to simulation.mp4.",
    )
    physical.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="Recorded frames per simulated second (default when recording: 30).",
    )
    physical.add_argument(
        "--video-width",
        type=int,
        default=None,
        help="Recorded frame width (default when recording: 1920).",
    )
    physical.add_argument(
        "--video-height",
        type=int,
        default=None,
        help="Recorded frame height (default when recording: 1080).",
    )
    physical.add_argument(
        "--video-camera",
        default=None,
        help=(
            "Force one named camera and disable automatic action-directed cuts "
            "(default: mission-specific multi-angle direction)."
        ),
    )
    physical.set_defaults(func=_cmd_mujoco)

    recovery = sub.add_parser(
        "recovery-experiment",
        help=(
            "Record a fault-only control and same-simulation LLM-adapted MuJoCo recovery trial."
        ),
    )
    recovery.add_argument("--scenario", default=str(RECOVERY_SCENARIO))
    recovery.add_argument("--bt", default=str(RECOVERY_BT), help="Nominal pre-failure BT JSON.")
    recovery.add_argument("--fault", default=str(RECOVERY_FAULT))
    recovery.add_argument("--assets-dir", default=None)
    recovery.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "recovery"))
    recovery.add_argument("--planner", choices=["openai", "oracle"], default="openai")
    recovery.add_argument("--model", default="gpt-5.6-sol")
    recovery.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    recovery.add_argument(
        "--oracle-bt",
        default=str(RECOVERY_ORACLE_BT),
        help="Offline test fixture used only with --planner oracle; never presented as LLM evidence.",
    )
    recovery.add_argument("--max-corrections", type=int, default=2)
    recovery.add_argument("--max-ticks", type=int, default=160)
    recovery.add_argument("--max-seconds", type=float, default=160.0)
    recovery.add_argument(
        "--no-video",
        action="store_true",
        help="Run the integration experiment without encoding the three videos.",
    )
    recovery.add_argument("--video-fps", type=int, default=30)
    recovery.add_argument("--video-width", type=int, default=1920)
    recovery.add_argument("--video-height", type=int, default=1080)
    recovery.set_defaults(func=_cmd_recovery_experiment)

    adaptive = sub.add_parser(
        "adaptive-demo",
        help=(
            "Generate a fault-blind BT with OpenAI, run MuJoCo, adapt after a runtime "
            "failure, resume the same simulation, and record one directed video."
        ),
    )
    adaptive.add_argument("--scenario", default=str(RECOVERY_SCENARIO))
    adaptive.add_argument("--fault", default=str(RECOVERY_FAULT))
    adaptive.add_argument("--assets-dir", default=None)
    adaptive.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "adaptive_demo"))
    adaptive.add_argument("--model", default="gpt-5.6-sol")
    adaptive.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    adaptive.add_argument("--generation-max-corrections", type=int, default=4)
    adaptive.add_argument("--recovery-max-corrections", type=int, default=2)
    adaptive.add_argument("--max-ticks", type=int, default=160)
    adaptive.add_argument("--max-seconds", type=float, default=160.0)
    adaptive.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the live MuJoCo viewer (video recording can still run).",
    )
    adaptive.add_argument(
        "--realtime-factor",
        type=float,
        default=1.0,
        help="Live viewer playback speed; recorded video remains simulation-time based.",
    )
    adaptive.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=5.0,
        help="Print a still-working status at this interval during LLM calls.",
    )
    adaptive.add_argument(
        "--no-video",
        action="store_true",
        help="Run the complete real-LLM integration without encoding adaptive_demo.mp4.",
    )
    adaptive.add_argument("--video-fps", type=int, default=30)
    adaptive.add_argument("--video-width", type=int, default=1920)
    adaptive.add_argument("--video-height", type=int, default=1080)
    adaptive.set_defaults(func=_cmd_adaptive_demo)
    return parser


def _add_scenario_bt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    parser.add_argument("--bt", required=True, help="Plain canonical plan JSON or exported behavior_tree.json.")


def _cmd_generate(args: argparse.Namespace) -> int:
    scenario_document = PlannerService.load_json(args.scenario)
    api_key = _provider_key(args.provider, args.use_saved_key)
    service = PlannerService(args.output_dir)

    def progress(message: str, fraction: float) -> None:
        print(f"[{round(fraction * 100):3d}%] {message}")

    outcome = service.generate(
        scenario_document,
        provider=args.provider,
        api_key=api_key,
        model=args.model,
        max_corrections=args.max_corrections,
        max_ticks=args.max_ticks,
        progress=progress,
    )
    print(f"Artifacts: {outcome.artifacts.directory}")
    if outcome.artifacts.behavior_tree_json is not None:
        print(f"BT JSON:   {outcome.artifacts.behavior_tree_json}")
        print(f"BT XML:    {outcome.artifacts.behavior_tree_xml}")
    else:
        print("BT output:  not published because validation/simulation did not pass")
    print(f"Valid:     {'yes' if outcome.validation.valid else 'no'}")
    print(f"Goal:      {'reached' if outcome.simulation.goal_success else 'not reached'}")
    if outcome.artifacts.behavior_tree_json is not None:
        artifact = json.loads(outcome.artifacts.behavior_tree_json.read_text(encoding="utf-8"))
        print(f"BT_FILE={outcome.artifacts.behavior_tree_json.resolve()}")
        print(f"BT_SHA256={artifact['artifact_sha256']}")
    return 0 if outcome.validation.valid and outcome.simulation.success else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, strict=True)
    plan = load_plan_file(args.bt)
    report = validate_plan(plan, scenario, suggest_producers=True)
    payload = {"valid": report.valid, "errors": report.to_dicts()}
    _emit_json(payload, args.output)
    return 0 if report.valid else 1


def _cmd_simulate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, strict=True)
    plan = load_plan_file(args.bt)
    validation = validate_plan(plan, scenario, suggest_producers=True)
    payload: dict[str, Any]
    if validation.valid:
        report = simulate(plan, scenario, max_ticks=args.max_ticks)
        payload = {"validation": {"valid": True, "errors": []}, "simulation": report.to_dict()}
        success = report.success
    else:
        payload = {"validation": {"valid": False, "errors": validation.to_dicts()}, "simulation": None}
        success = False
    _emit_json(payload, args.output)
    return 0 if success else 1


def _cmd_render(args: argparse.Namespace) -> int:
    plan = load_plan_file(args.bt)
    html_path = Path(args.html)
    save_text(html_path, plan_to_html(plan, title="Multi-robot Behavior Trees"))
    print(f"HTML: {html_path.resolve()}")
    if args.xml:
        xml_path = Path(args.xml)
        save_text(xml_path, export_behaviortree_cpp_xml(plan))
        print(f"XML:  {xml_path.resolve()}")
    return 0


def _cmd_template(args: argparse.Namespace) -> int:
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_TEMPLATE, target)
    print(target.resolve())
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:  # noqa: ARG001
    checks: list[tuple[str, bool, str]] = []
    for dependency in ("jsonschema", "gradio", "keyring"):
        try:
            __import__(dependency)
            checks.append((dependency, True, "installed"))
        except ImportError:
            checks.append((dependency, False, "missing; install pip install -e '.[ui]'"))
    references = (
        ("courier", DEFAULT_SCENARIO, PROJECT_ROOT / "examples" / "three_robot_courier.bt.json"),
        (
            "packaging",
            PACKAGING_SCENARIO,
            PROJECT_ROOT / "examples" / "three_robot_packaging_delivery.bt.json",
        ),
        ("recovery nominal", RECOVERY_SCENARIO, RECOVERY_BT),
    )
    for label, scenario_path, bt_path in references:
        try:
            scenario = load_scenario(scenario_path, strict=True)
            plan = load_plan_file(bt_path)
            validation = validate_plan(plan, scenario)
            simulation = simulate(plan, scenario, max_ticks=140) if validation.valid else None
            checks.append(
                (f"{label} reference validation", validation.valid, f"{len(validation.errors)} error(s)")
            )
            checks.append(
                (
                    f"{label} reference simulation",
                    bool(simulation and simulation.success),
                    "goals reached" if simulation and simulation.success else "failed",
                )
            )
        except Exception as error:
            checks.append((f"{label} reference pipeline", False, str(error)))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL':4}  {name}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def _cmd_ui(args: argparse.Namespace) -> int:
    from .ui import launch_ui

    launch_ui(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser)
    return 0


def _cmd_mujoco(args: argparse.Namespace) -> int:
    """Lazy import keeps MuJoCo entirely outside generation and the web UI."""
    try:
        from .mujoco_sim.runner import run_cli
    except ImportError as error:
        if error.name in {"mujoco", "numpy", "imageio", "imageio_ffmpeg"}:
            raise RuntimeError(
                "MuJoCo simulation dependencies are missing. Install them with "
                "python -m pip install -e \".[mujoco]\"."
            ) from error
        raise
    return run_cli(args)


def _cmd_recovery_experiment(args: argparse.Namespace) -> int:
    """Lazy import keeps the recovery command's MuJoCo stack optional."""
    try:
        from .mujoco_sim.recovery_runner import run_recovery_cli
    except ImportError as error:
        if error.name in {"mujoco", "numpy", "imageio", "imageio_ffmpeg"}:
            raise RuntimeError(
                "MuJoCo recording dependencies are missing. Install them with "
                "python -m pip install -e \".[mujoco]\"."
            ) from error
        raise
    return run_recovery_cli(args)


def _cmd_adaptive_demo(args: argparse.Namespace) -> int:
    """Lazy import keeps the unified adaptive demo's MuJoCo stack optional."""
    try:
        from .mujoco_sim.adaptive_demo_runner import run_adaptive_demo_cli
    except ImportError as error:
        if error.name in {"mujoco", "numpy", "imageio", "imageio_ffmpeg", "PIL"}:
            raise RuntimeError(
                "MuJoCo recording dependencies are missing. Install them with "
                "python -m pip install -e \".[mujoco]\"."
            ) from error
        raise
    return run_adaptive_demo_cli(args)


def _provider_key(provider: str, use_saved: bool) -> str:
    if use_saved:
        key = SecretStore().load(provider)
        if key:
            return key
        raise ValueError(f"No saved {provider} key is available in the OS credential store.")
    env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.environ.get(env_name, "")
    if not key:
        raise ValueError(f"{env_name} is not set. The selected provider is never changed automatically.")
    return key


def _emit_json(payload: dict, output: str | None) -> None:
    if output:
        path = Path(output)
        save_json(path, payload)
        print(path.resolve())
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
