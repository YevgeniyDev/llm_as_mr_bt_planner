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
INSPECTION_SCENARIO = PROJECT_ROOT / "examples" / "five_agent_solar_pipe_inspection.json"
INSPECTION_TOOL_DROP_FAULT = (
    PROJECT_ROOT / "examples" / "five_agent_solar_pipe_inspection_tool_drop.fault.json"
)
PIPE_REPAIR_SCENARIO = PROJECT_ROOT / "examples" / "five_agent_pipe_leak_repair.json"
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

    inspection = sub.add_parser(
        "inspection-demo",
        help="Generate five coordinated inspection BTs with an LLM, then launch MuJoCo automatically.",
    )
    inspection.add_argument("--scenario", default=str(INSPECTION_SCENARIO))
    inspection.add_argument("--model", default="gpt-5.6-sol")
    inspection.add_argument("--use-saved-key", action="store_true")
    inspection.add_argument("--max-corrections", type=int, default=4)
    inspection.add_argument("--max-ticks", type=int, default=300)
    inspection.add_argument("--assets-dir", default=None)
    inspection.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "inspection-demo"))
    inspection.add_argument("--headless", action="store_true")
    inspection.add_argument("--max-seconds", type=float, default=100.0)
    inspection.add_argument("--realtime-factor", type=float, default=1.0)
    inspection.add_argument("--no-video", action="store_true")
    inspection.add_argument("--video-fps", type=int, default=None)
    inspection.add_argument("--video-width", type=int, default=None)
    inspection.add_argument("--video-height", type=int, default=None)
    inspection.add_argument("--video-camera", default=None)
    inspection.set_defaults(func=_cmd_inspection_demo)

    pipe_repair = sub.add_parser(
        "pipe-repair-demo",
        help="Generate a five-agent pipe-leak repair BT with an LLM, then launch MuJoCo.",
    )
    pipe_repair.add_argument("--scenario", default=str(PIPE_REPAIR_SCENARIO))
    pipe_repair.add_argument("--model", default="gpt-5.6-sol")
    pipe_repair.add_argument("--use-saved-key", action="store_true")
    pipe_repair.add_argument("--max-corrections", type=int, default=4)
    pipe_repair.add_argument("--max-ticks", type=int, default=300)
    pipe_repair.add_argument("--assets-dir", default=None)
    pipe_repair.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "pipe-repair-demo"))
    pipe_repair.add_argument("--headless", action="store_true")
    pipe_repair.add_argument("--max-seconds", type=float, default=120.0)
    pipe_repair.add_argument("--realtime-factor", type=float, default=1.0)
    pipe_repair.add_argument("--no-video", action="store_true")
    pipe_repair.add_argument("--video-fps", type=int, default=None)
    pipe_repair.add_argument("--video-width", type=int, default=None)
    pipe_repair.add_argument("--video-height", type=int, default=None)
    pipe_repair.add_argument("--video-camera", default=None)
    pipe_repair.set_defaults(func=_cmd_inspection_demo)

    inspection_adaptive = sub.add_parser(
        "inspection-adaptive-demo",
        help=(
            "Generate fault-blind five-agent inspection BTs, drop the tool in MuJoCo, "
            "adapt with an LLM, and resume the same simulation."
        ),
    )
    inspection_adaptive.add_argument("--scenario", default=str(INSPECTION_SCENARIO))
    inspection_adaptive.add_argument("--fault", default=str(INSPECTION_TOOL_DROP_FAULT))
    inspection_adaptive.add_argument("--model", default="gpt-5.6-sol")
    inspection_adaptive.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default="high",
    )
    inspection_adaptive.add_argument("--generation-max-corrections", type=int, default=4)
    inspection_adaptive.add_argument("--recovery-max-corrections", type=int, default=3)
    inspection_adaptive.add_argument("--max-ticks", type=int, default=400)
    inspection_adaptive.add_argument("--max-seconds", type=float, default=180.0)
    inspection_adaptive.add_argument("--assets-dir", default=None)
    inspection_adaptive.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "inspection-adaptive-demo"),
    )
    inspection_adaptive.add_argument("--headless", action="store_true")
    inspection_adaptive.add_argument("--realtime-factor", type=float, default=1.0)
    inspection_adaptive.add_argument("--heartbeat-seconds", type=float, default=5.0)
    inspection_adaptive.add_argument("--no-video", action="store_true")
    inspection_adaptive.add_argument("--video-fps", type=int, default=30)
    inspection_adaptive.add_argument("--video-width", type=int, default=1920)
    inspection_adaptive.add_argument("--video-height", type=int, default=1080)
    inspection_adaptive.set_defaults(func=_cmd_inspection_adaptive_demo)

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

    compare = sub.add_parser(
        "compare",
        help="Prepare and run paper-baseline reproductions under the common protocol.",
    )
    methods = compare.add_subparsers(dest="comparison_method", required=True)
    llm_as_bt = methods.add_parser(
        "llm-as-bt-planner",
        help="KIOS JSON assembly planner with the paper's four in-context generation schemes.",
    )
    llm_as_bt_actions = llm_as_bt.add_subparsers(dest="comparison_action", required=True)
    llm_as_bt_prepare = llm_as_bt_actions.add_parser(
        "prepare",
        help="Download, hash, and verify the pinned official MIT-licensed KIOS source.",
    )
    llm_as_bt_prepare.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-as-bt-planner" / "source"),
    )
    llm_as_bt_prepare.add_argument("--force", action="store_true", help="Redownload the pinned archive.")
    llm_as_bt_prepare.set_defaults(func=_cmd_llm_as_bt_prepare)

    llm_as_bt_run = llm_as_bt_actions.add_parser(
        "run",
        help="Generate native KIOS trees and evaluate their strict common-protocol observation.",
    )
    llm_as_bt_run.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    llm_as_bt_run.add_argument(
        "--scheme",
        choices=["one-step", "iterative", "human", "recursive"],
        default="one-step",
    )
    llm_as_bt_run.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    llm_as_bt_run.add_argument(
        "--model",
        default="gpt-4",
        help="LLM backbone; GPT-4 matches the model reported for the paper's main comparison.",
    )
    llm_as_bt_run.add_argument(
        "--responses",
        help="JSON containing ordered archived stage responses; replay only, not model evidence.",
    )
    llm_as_bt_run.add_argument(
        "--human-feedback",
        help="For --scheme human, JSON mapping subgoal ids to ordered feedback strings.",
    )
    llm_as_bt_run.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-as-bt-planner" / "runs"),
    )
    llm_as_bt_run.add_argument("--max-iterations", type=int, default=5)
    llm_as_bt_run.add_argument("--max-recursive-depth", type=int, default=12)
    llm_as_bt_run.add_argument("--max-recursive-expansions", type=int, default=80)
    llm_as_bt_run.add_argument("--max-ticks", type=int, default=160)
    llm_as_bt_run.add_argument("--seed", type=int, default=42)
    llm_as_bt_run.add_argument(
        "--use-saved-key",
        action="store_true",
        help="Use the selected provider key explicitly saved by the UI.",
    )
    llm_as_bt_run.set_defaults(func=_cmd_llm_as_bt_run)

    llm_bt = methods.add_parser(
        "llm-bt",
        help="ChatGPT reasoning, released BERT parsing, and deterministic adaptive BT expansion.",
    )
    llm_bt_actions = llm_bt.add_subparsers(dest="comparison_action", required=True)
    llm_bt_prepare = llm_bt_actions.add_parser(
        "prepare",
        help="Download and verify the pinned official method files and released BERT parser.",
    )
    llm_bt_prepare.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-bt" / "source"),
    )
    llm_bt_prepare.add_argument("--force", action="store_true")
    llm_bt_prepare.add_argument(
        "--without-parser-model",
        action="store_true",
        help="Prepare source provenance only, omitting the 265 MB released checkpoint.",
    )
    llm_bt_prepare.set_defaults(func=_cmd_llm_bt_prepare)

    llm_bt_run = llm_bt_actions.add_parser(
        "run",
        help="Reason once, parse goals with BERT, expand the ATL, and evaluate nominal execution.",
    )
    llm_bt_run.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    llm_bt_run.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="ChatGPT model; the paper does not report a model version, so every run records this choice.",
    )
    llm_bt_run.add_argument(
        "--responses",
        help="Archived reasoning response plus NER predictions; replay only, not model evidence.",
    )
    llm_bt_run.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-bt" / "source"),
    )
    llm_bt_run.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-bt" / "runs"),
    )
    llm_bt_run.add_argument("--max-ticks", type=int, default=160)
    llm_bt_run.add_argument("--seed", type=int, default=42)
    llm_bt_run.add_argument(
        "--use-saved-key",
        action="store_true",
        help="Use the OpenAI key explicitly saved by the UI.",
    )
    llm_bt_run.set_defaults(func=_cmd_llm_bt_run)

    llm_bt_recover = llm_bt_actions.add_parser(
        "recover",
        help="Re-expand nominal parsed goals from a standardized post-failure snapshot without an LLM call.",
    )
    llm_bt_recover.add_argument("--scenario", default=str(RECOVERY_SCENARIO))
    llm_bt_recover.add_argument(
        "--nominal-run",
        required=True,
        help="Completed LLM-BT nominal run directory containing native/parsed_goals.json.",
    )
    llm_bt_recover.add_argument(
        "--failure-snapshot",
        required=True,
        help="JSON containing measured_initial_state and failure_observation.",
    )
    llm_bt_recover.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-bt" / "recovery"),
    )
    llm_bt_recover.add_argument("--max-ticks", type=int, default=160)
    llm_bt_recover.set_defaults(func=_cmd_llm_bt_recover)

    betr_xp = methods.add_parser(
        "betr-xp-llm",
        help="Formal LLM goals, reactive backchaining, and failure-time error resolution.",
    )
    betr_xp_actions = betr_xp.add_subparsers(dest="comparison_action", required=True)
    betr_xp_prepare = betr_xp_actions.add_parser(
        "prepare",
        help="Download, hash, and verify the pinned official BSD-licensed source.",
    )
    betr_xp_prepare.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "betr-xp-llm" / "source"),
    )
    betr_xp_prepare.add_argument("--force", action="store_true")
    betr_xp_prepare.set_defaults(func=_cmd_betr_xp_prepare)

    betr_xp_run = betr_xp_actions.add_parser(
        "run",
        help="Formalize the task goal once, generate the reactive policy, and evaluate it.",
    )
    betr_xp_run.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    betr_xp_run.add_argument(
        "--model",
        default="gpt-4-1106-preview",
        help="The paper used GPT-4-1106-Preview; availability depends on the provider account.",
    )
    betr_xp_run.add_argument(
        "--responses",
        help="JSON with an archived goal_response; replay only, not real-model evidence.",
    )
    betr_xp_run.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "betr-xp-llm" / "source"),
        help="Prepared official source root, verified before real-model inference.",
    )
    betr_xp_run.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "betr-xp-llm" / "runs"),
    )
    betr_xp_run.add_argument("--max-ticks", type=int, default=160)
    betr_xp_run.add_argument("--seed", type=int, default=None)
    betr_xp_run.add_argument("--use-saved-key", action="store_true")
    betr_xp_run.set_defaults(func=_cmd_betr_xp_run)

    betr_xp_recover = betr_xp_actions.add_parser(
        "recover",
        help="Resolve the failed pickup parameter with the LLM and regenerate the continuation.",
    )
    betr_xp_recover.add_argument("--scenario", default=str(RECOVERY_SCENARIO))
    betr_xp_recover.add_argument(
        "--nominal-run",
        required=True,
        help="Completed BETR-XP-LLM nominal run directory.",
    )
    betr_xp_recover.add_argument(
        "--failure-snapshot",
        required=True,
        help="JSON containing measured_initial_state and failure_observation.",
    )
    betr_xp_recover.add_argument(
        "--model",
        default="gpt-4-1106-preview",
        help="The paper used GPT-4-1106-Preview; availability depends on the provider account.",
    )
    betr_xp_recover.add_argument(
        "--responses",
        help="JSON with an archived recovery_response; replay only, not real-model evidence.",
    )
    betr_xp_recover.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "betr-xp-llm" / "source"),
        help="Prepared official source root, verified before real-model inference.",
    )
    betr_xp_recover.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "betr-xp-llm" / "recovery"),
    )
    betr_xp_recover.add_argument("--max-ticks", type=int, default=160)
    betr_xp_recover.add_argument("--seed", type=int, default=None)
    betr_xp_recover.add_argument("--use-saved-key", action="store_true")
    betr_xp_recover.set_defaults(func=_cmd_betr_xp_recover)

    llm_hbt = methods.add_parser(
        "llm-hbt",
        help="Dynamic LLM condition initialization, Alex assignment, and online BT updates.",
    )
    llm_hbt_actions = llm_hbt.add_subparsers(dest="comparison_action", required=True)
    llm_hbt_prepare = llm_hbt_actions.add_parser(
        "prepare",
        help="Pin the arXiv v1 source and author project page (official code is unavailable).",
    )
    llm_hbt_prepare.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-hbt" / "source"),
    )
    llm_hbt_prepare.add_argument("--force", action="store_true")
    llm_hbt_prepare.set_defaults(func=_cmd_llm_hbt_prepare)

    llm_hbt_run = llm_hbt_actions.add_parser(
        "run",
        help="Initialize conditions with an LLM and construct the nominal heterogeneous BT forest.",
    )
    llm_hbt_run.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    llm_hbt_run.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    llm_hbt_run.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help=(
            "Reproduction model; the LLM-HBT paper does not identify its model, so the "
            "selected value is recorded as a reproduction choice."
        ),
    )
    llm_hbt_run.add_argument(
        "--responses",
        help="JSON containing ordered archived native decisions; replay only, not model evidence.",
    )
    llm_hbt_run.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-hbt" / "source"),
        help="Prepared provenance root, verified before real-model inference.",
    )
    llm_hbt_run.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-hbt" / "runs"),
    )
    llm_hbt_run.add_argument("--max-extensions", type=int, default=100)
    llm_hbt_run.add_argument("--max-ticks", type=int, default=160)
    llm_hbt_run.add_argument("--seed", type=int, default=42)
    llm_hbt_run.add_argument("--use-saved-key", action="store_true")
    llm_hbt_run.set_defaults(func=_cmd_llm_hbt_run)

    llm_hbt_recover = llm_hbt_actions.add_parser(
        "recover",
        help="Detect a standardized runtime failure and let the LLM construct recovery updates.",
    )
    llm_hbt_recover.add_argument("--scenario", default=str(RECOVERY_SCENARIO))
    llm_hbt_recover.add_argument(
        "--nominal-run",
        required=True,
        help="Completed LLM-HBT nominal run directory whose initial conditions are reused.",
    )
    llm_hbt_recover.add_argument(
        "--failure-snapshot",
        required=True,
        help="JSON containing measured_initial_state and failure_observation.",
    )
    llm_hbt_recover.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="openai",
    )
    llm_hbt_recover.add_argument(
        "--model",
        default="gpt-4o-2024-08-06",
        help="Reproduction model; the paper's model is not reported.",
    )
    llm_hbt_recover.add_argument(
        "--responses",
        help="JSON containing ordered archived post-failure decisions; replay only.",
    )
    llm_hbt_recover.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-hbt" / "source"),
        help="Prepared provenance root, verified before real-model inference.",
    )
    llm_hbt_recover.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "llm-hbt" / "recovery"),
    )
    llm_hbt_recover.add_argument("--max-extensions", type=int, default=100)
    llm_hbt_recover.add_argument("--max-ticks", type=int, default=160)
    llm_hbt_recover.add_argument("--seed", type=int, default=42)
    llm_hbt_recover.add_argument("--use-saved-key", action="store_true")
    llm_hbt_recover.set_defaults(func=_cmd_llm_hbt_recover)

    mrbtp = methods.add_parser(
        "mrbtp",
        help="Non-LLM FIFO multi-robot BT planning with cross-tree expansion.",
    )
    mrbtp_actions = mrbtp.add_subparsers(dest="comparison_action", required=True)
    mrbtp_prepare = mrbtp_actions.add_parser(
        "prepare",
        help="Download, hash, license-check, and extract the pinned official MRBTP source.",
    )
    mrbtp_prepare.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "mrbtp" / "source"),
    )
    mrbtp_prepare.add_argument("--force", action="store_true")
    mrbtp_prepare.set_defaults(func=_cmd_mrbtp_prepare)

    mrbtp_run = mrbtp_actions.add_parser(
        "run",
        help="Run FIFO MRBTP without its optional LLM composite-action plugin.",
    )
    mrbtp_run.add_argument("--scenario", default=str(RECOVERY_SCENARIO))
    mrbtp_run.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "mrbtp" / "source"),
        help="Prepared official source root; every run verifies it before planning.",
    )
    mrbtp_run.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "comparison" / "mrbtp" / "runs"),
    )
    mrbtp_run.add_argument("--max-expansions", type=int, default=10_000)
    mrbtp_run.add_argument("--max-ticks", type=int, default=300)
    mrbtp_run.set_defaults(func=_cmd_mrbtp_run)
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
        (
            "five-agent inspection",
            INSPECTION_SCENARIO,
            PROJECT_ROOT / "examples" / "five_agent_solar_pipe_inspection.bt.json",
        ),
        (
            "five-agent pipe repair",
            PIPE_REPAIR_SCENARIO,
            PROJECT_ROOT / "examples" / "five_agent_pipe_leak_repair.bt.json",
        ),
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


def _cmd_inspection_demo(args: argparse.Namespace) -> int:
    """Generate an audited LLM BT artifact and execute that exact artifact physically."""
    scenario_document = PlannerService.load_json(args.scenario)
    output_root = Path(args.output).resolve()
    service = PlannerService(output_root / "generation")

    def progress(message: str, fraction: float) -> None:
        print(f"[BT generation {round(fraction * 100):3d}%] {message}")

    outcome = service.generate(
        scenario_document,
        provider="openai",
        api_key=_provider_key("openai", args.use_saved_key),
        model=args.model,
        max_corrections=args.max_corrections,
        max_ticks=args.max_ticks,
        progress=progress,
    )
    bt_path = outcome.artifacts.behavior_tree_json
    if bt_path is None or not outcome.validation.valid or not outcome.simulation.success:
        print(f"Generation diagnostics: {outcome.artifacts.directory.resolve()}")
        raise RuntimeError("LLM BT did not pass validation and contract simulation; MuJoCo was not launched.")
    print(f"Accepted LLM BT: {bt_path.resolve()}")
    print("Launching MuJoCo with the exact accepted artifact (no action rewriting).")
    args.bt = str(bt_path)
    args.output = str(output_root / "physical")
    args.record_video = not args.no_video
    args.setup_only = False
    from .mujoco_sim.runner import run_cli

    return run_cli(args)


def _cmd_inspection_adaptive_demo(args: argparse.Namespace) -> int:
    """Run fault-blind five-agent generation and same-state tool recovery."""
    try:
        from .mujoco_sim.inspection_adaptive_runner import run_inspection_adaptive_cli
    except ImportError as error:
        if error.name in {"mujoco", "numpy", "imageio", "imageio_ffmpeg", "PIL"}:
            raise RuntimeError(
                "MuJoCo recording dependencies are missing. Install them with "
                "python -m pip install -e \".[mujoco]\"."
            ) from error
        raise
    return run_inspection_adaptive_cli(args)


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


def _cmd_llm_as_bt_prepare(args: argparse.Namespace) -> int:
    from .comparison.llm_as_bt_source import prepare_official_source

    prepared = prepare_official_source(args.output, force=args.force)
    print(f"Source:   {prepared.source.resolve()}")
    print(f"Files:    {prepared.file_count}")
    print(f"Manifest: {prepared.manifest.resolve()}")
    print("License:  MIT")
    return 0


def _cmd_llm_as_bt_run(args: argparse.Namespace) -> int:
    from .comparison.llm_as_bt import (
        LLMAsBTGenerator,
        ProviderGenerator,
        ReplayGenerator,
        load_human_feedback,
        run_llm_as_bt_planner,
    )

    scenario = load_scenario(args.scenario, strict=True)
    generator: LLMAsBTGenerator
    if args.responses:
        generator = ReplayGenerator.from_file(args.responses)
        print("Mode: archived ordered KIOS replay (not real model evidence)")
    else:
        generator = ProviderGenerator(
            args.provider,
            args.model,
            _provider_key(args.provider, args.use_saved_key),
            seed=args.seed,
        )
    feedback = load_human_feedback(args.human_feedback) if args.human_feedback else None
    if feedback and args.scheme != "human":
        raise ValueError("--human-feedback is valid only with --scheme human.")
    result = run_llm_as_bt_planner(
        scenario,
        generator,
        args.output,
        scheme=args.scheme,
        max_iterations=args.max_iterations,
        human_feedback=feedback,
        max_recursive_depth=args.max_recursive_depth,
        max_recursive_expansions=args.max_recursive_expansions,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    print(f"Artifacts: {result.directory.resolve()}")
    print(f"Native:    {(result.directory / 'native').resolve()}")
    print(f"Canonical: {result.canonical_plan.resolve()}")
    print(f"PS:        {'pass' if result.plan_generation_success else 'fail'}")
    print(f"SV:        {'pass' if result.static_validity else 'fail'}")
    print(f"GS:        {'pass' if result.symbolic_goal_success else 'fail'}")
    if result.accepted_plan is None:
        print("Accepted:  no; no MuJoCo-ready plan was published")
    else:
        print(f"Accepted:  {result.accepted_plan.resolve()}")
    return 0 if (
        result.plan_generation_success and result.static_validity and result.symbolic_goal_success
    ) else 1


def _cmd_llm_bt_prepare(args: argparse.Namespace) -> int:
    from .comparison.llm_bt_source import prepare_official_source

    prepared = prepare_official_source(
        args.output,
        force=args.force,
        include_parser_model=not args.without_parser_model,
    )
    print(f"Source:   {prepared.source.resolve()}")
    print(f"Files:    {prepared.file_count}")
    print(f"Parser:   {prepared.parser.resolve()}")
    print(f"Weights:  {'included (265 MB)' if prepared.parser_model_included else 'omitted'}")
    print(f"Manifest: {prepared.manifest.resolve()}")
    print("License:  no project-wide software/model license declared upstream")
    return 0


def _cmd_llm_bt_run(args: argparse.Namespace) -> int:
    from .comparison.llm_bt import (
        ProviderReasoner,
        Reasoner,
        load_replay_bundle,
        run_llm_bt,
    )
    from .comparison.llm_bt_parser import KeywordParser

    scenario = load_scenario(args.scenario, strict=True)
    reasoner: Reasoner
    keyword_parser: KeywordParser
    if args.responses:
        reasoner, keyword_parser = load_replay_bundle(args.responses)
        print("Mode: archived LLM-BT reasoning/NER replay (not real model evidence)")
    else:
        from .comparison.llm_bt_parser import TransformersKeywordParser
        from .comparison.llm_bt_source import parser_directory, verify_prepared_source

        verify_prepared_source(args.source, require_parser_model=True)
        reasoner = ProviderReasoner(
            args.model,
            _provider_key("openai", args.use_saved_key),
            seed=args.seed,
        )
        keyword_parser = TransformersKeywordParser(parser_directory(args.source))
        print(f"Mode: ChatGPT reasoning ({args.model}) plus released DistilBERT parser")
    result = run_llm_bt(
        scenario,
        reasoner,
        keyword_parser,
        args.output,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    return _print_llm_bt_result(result)


def _cmd_llm_bt_recover(args: argparse.Namespace) -> int:
    from .comparison.llm_bt import run_llm_bt_recovery
    from .recovery import build_runtime_recovery_scenario

    scenario = load_scenario(args.scenario, strict=True)
    document = json.loads(Path(args.failure_snapshot).read_text(encoding="utf-8"))
    measured = document.get("measured_initial_state") if isinstance(document, dict) else None
    observation = document.get("failure_observation") if isinstance(document, dict) else None
    if not isinstance(measured, list) or not all(isinstance(item, str) for item in measured):
        raise ValueError("failure snapshot requires a measured_initial_state string array.")
    if not isinstance(observation, dict):
        raise ValueError("failure snapshot requires a failure_observation object.")
    runtime_scenario = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=tuple(measured),
        failure_observation=observation,
    )
    print("Mode: deterministic LLM-BT runtime ATL expansion (LLM and BERT are not recalled)")
    result = run_llm_bt_recovery(
        runtime_scenario,
        args.nominal_run,
        document,
        args.output,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    return _print_llm_bt_result(result, recovery=True)


def _print_llm_bt_result(result, *, recovery: bool = False) -> int:
    print(f"Artifacts: {result.directory.resolve()}")
    print(f"Native:    {(result.directory / 'native').resolve()}")
    print(f"Canonical: {result.canonical_plan.resolve()}")
    labels = ("RPS", "RV", "RGS") if recovery else ("PS", "SV", "GS")
    print(f"{labels[0]}:       {'pass' if result.plan_generation_success else 'fail'}")
    print(f"{labels[1]}:        {'pass' if result.static_validity else 'fail'}")
    print(f"{labels[2]}:        {'pass' if result.symbolic_goal_success else 'fail'}")
    if result.accepted_plan is None:
        print("Accepted:  no; no MuJoCo-ready plan was published")
    else:
        print(f"Accepted:  {result.accepted_plan.resolve()}")
    return 0 if (
        result.plan_generation_success and result.static_validity and result.symbolic_goal_success
    ) else 1


def _cmd_betr_xp_prepare(args: argparse.Namespace) -> int:
    from .comparison.betr_xp_source import prepare_official_source

    prepared = prepare_official_source(args.output, force=args.force)
    print(f"Source:   {prepared.source.resolve()}")
    print(f"Files:    {prepared.file_count}")
    print(f"Archive:  {prepared.archive.resolve()}")
    print(f"Manifest: {prepared.manifest.resolve()}")
    print("License:  BSD-3-Clause; copyright (c) 2024, ABB")
    return 0


def _cmd_betr_xp_run(args: argparse.Namespace) -> int:
    from .comparison.betr_xp import BetrXPCaller, ProviderCaller, ReplayCaller, run_betr_xp

    scenario = load_scenario(args.scenario, strict=True)
    caller: BetrXPCaller
    if args.responses:
        caller = ReplayCaller.from_file(args.responses)
        print("Mode: archived BETR-XP-LLM goal response replay (not real model evidence)")
    else:
        from .comparison.betr_xp_source import verify_prepared_source

        verify_prepared_source(args.source)
        caller = ProviderCaller(
            args.model,
            _provider_key("openai", args.use_saved_key),
            seed=args.seed,
        )
        print(f"Mode: one formal-goal OpenAI call ({args.model}) plus reactive backchaining")
    result = run_betr_xp(
        scenario,
        caller,
        args.output,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    return _print_betr_xp_result(result)


def _cmd_betr_xp_recover(args: argparse.Namespace) -> int:
    from .comparison.betr_xp import (
        BetrXPCaller,
        ProviderCaller,
        ReplayCaller,
        run_betr_xp_recovery,
    )
    from .recovery import build_runtime_recovery_scenario

    scenario = load_scenario(args.scenario, strict=True)
    document = json.loads(Path(args.failure_snapshot).read_text(encoding="utf-8"))
    measured = document.get("measured_initial_state") if isinstance(document, dict) else None
    observation = document.get("failure_observation") if isinstance(document, dict) else None
    if not isinstance(measured, list) or not all(isinstance(item, str) for item in measured):
        raise ValueError("failure snapshot requires a measured_initial_state string array.")
    if not isinstance(observation, dict):
        raise ValueError("failure snapshot requires a failure_observation object.")
    runtime_scenario = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=tuple(measured),
        failure_observation=observation,
    )
    caller: BetrXPCaller
    if args.responses:
        caller = ReplayCaller.from_file(args.responses)
        print("Mode: archived BETR-XP-LLM failure-resolution replay (not real model evidence)")
    else:
        from .comparison.betr_xp_source import verify_prepared_source

        verify_prepared_source(args.source)
        caller = ProviderCaller(
            args.model,
            _provider_key("openai", args.use_saved_key),
            seed=args.seed,
        )
        print(f"Mode: post-failure OpenAI parameter resolution ({args.model}) and replanning")
    result = run_betr_xp_recovery(
        runtime_scenario,
        args.nominal_run,
        document,
        caller,
        args.output,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    return _print_betr_xp_result(result, recovery=True)


def _print_betr_xp_result(result, *, recovery: bool = False) -> int:
    print(f"Artifacts: {result.directory.resolve()}")
    print(f"Native:    {(result.directory / 'native').resolve()}")
    print(f"Canonical: {result.canonical_plan.resolve()}")
    labels = ("RPS", "RV", "RGS") if recovery else ("PS", "SV", "GS")
    print(f"{labels[0]}:       {'pass' if result.plan_generation_success else 'fail'}")
    print(f"{labels[1]}:        {'pass' if result.static_validity else 'fail'}")
    print(f"{labels[2]}:        {'pass' if result.symbolic_goal_success else 'fail'}")
    if result.accepted_plan is None:
        print("Accepted:  no; no MuJoCo-ready plan was published")
    else:
        print(f"Accepted:  {result.accepted_plan.resolve()}")
    return 0 if (
        result.plan_generation_success and result.static_validity and result.symbolic_goal_success
    ) else 1


def _cmd_llm_hbt_prepare(args: argparse.Namespace) -> int:
    from .comparison.llm_hbt_source import prepare_official_source

    prepared = prepare_official_source(args.output, force=args.force)
    print(f"Source:   {prepared.source.resolve()}")
    print(f"Files:    {prepared.file_count}")
    print(f"Project:  {prepared.project_archive.resolve()}")
    print(f"Paper:    {prepared.paper_archive.resolve()}")
    print(f"Manifest: {prepared.manifest.resolve()}")
    print("Code:     not released; author project page says Coming Soon")
    return 0


def _llm_hbt_generator(args: argparse.Namespace):
    from .comparison.llm_hbt import ProviderGenerator, ReplayGenerator

    if args.responses:
        print("Mode: archived ordered LLM-HBT decision replay (not real model evidence)")
        return ReplayGenerator.from_file(args.responses)
    from .comparison.llm_hbt_source import verify_prepared_source

    verify_prepared_source(args.source)
    print(
        f"Mode: live {args.provider} LLM-HBT reproduction ({args.model}); "
        "the paper model was not reported"
    )
    return ProviderGenerator(
        args.provider,
        args.model,
        _provider_key(args.provider, args.use_saved_key),
        seed=args.seed,
    )


def _cmd_llm_hbt_run(args: argparse.Namespace) -> int:
    from .comparison.llm_hbt import run_llm_hbt

    scenario = load_scenario(args.scenario, strict=True)
    result = run_llm_hbt(
        scenario,
        _llm_hbt_generator(args),
        args.output,
        max_extensions=args.max_extensions,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    return _print_llm_hbt_result(result)


def _cmd_llm_hbt_recover(args: argparse.Namespace) -> int:
    from .comparison.llm_hbt import run_llm_hbt_recovery
    from .recovery import build_runtime_recovery_scenario

    scenario = load_scenario(args.scenario, strict=True)
    document = json.loads(Path(args.failure_snapshot).read_text(encoding="utf-8"))
    measured = document.get("measured_initial_state") if isinstance(document, dict) else None
    observation = document.get("failure_observation") if isinstance(document, dict) else None
    if not isinstance(measured, list) or not all(isinstance(item, str) for item in measured):
        raise ValueError("failure snapshot requires a measured_initial_state string array.")
    if not isinstance(observation, dict):
        raise ValueError("failure snapshot requires a failure_observation object.")
    runtime_scenario = build_runtime_recovery_scenario(
        scenario,
        measured_initial_state=tuple(measured),
        failure_observation=observation,
    )
    result = run_llm_hbt_recovery(
        runtime_scenario,
        args.nominal_run,
        document,
        _llm_hbt_generator(args),
        args.output,
        max_extensions=args.max_extensions,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
    )
    return _print_llm_hbt_result(result, recovery=True)


def _print_llm_hbt_result(result, *, recovery: bool = False) -> int:
    print(f"Artifacts: {result.directory.resolve()}")
    print(f"Native:    {(result.directory / 'native').resolve()}")
    print(f"Canonical: {result.canonical_plan.resolve()}")
    labels = ("RPS", "RV", "RGS") if recovery else ("PS", "SV", "GS")
    print(f"{labels[0]}:       {'pass' if result.plan_generation_success else 'fail'}")
    print(f"{labels[1]}:        {'pass' if result.static_validity else 'fail'}")
    print(f"{labels[2]}:        {'pass' if result.symbolic_goal_success else 'fail'}")
    if result.accepted_plan is None:
        print("Accepted:  no; no MuJoCo-ready plan was published")
    else:
        print(f"Accepted:  {result.accepted_plan.resolve()}")
    return 0 if (
        result.plan_generation_success and result.static_validity and result.symbolic_goal_success
    ) else 1


def _cmd_mrbtp_prepare(args: argparse.Namespace) -> int:
    from .comparison.mrbtp_source import prepare_official_source

    prepared = prepare_official_source(args.output, force=args.force)
    print(f"Source:   {prepared.source.resolve()}")
    print(f"Files:    {prepared.file_count}")
    print(f"Archive:  {prepared.archive.resolve()}")
    print(f"Manifest: {prepared.manifest.resolve()}")
    print("License:  MIT; copyright (c) 2024 MABTPG")
    return 0


def _cmd_mrbtp_run(args: argparse.Namespace) -> int:
    from .comparison.mrbtp import run_mrbtp
    from .comparison.mrbtp_source import verify_prepared_source

    verify_prepared_source(args.source)
    scenario = load_scenario(args.scenario, strict=True)
    print("Mode: deterministic FIFO MRBTP; optional LLM subtree plugin disabled")
    result = run_mrbtp(
        scenario,
        args.output,
        max_expansions=args.max_expansions,
        max_ticks=args.max_ticks,
        invocation=args.invocation,
        verified_source_manifest=Path(args.source) / "source_manifest.json",
    )
    print(f"Artifacts: {result.directory.resolve()}")
    print(f"Native:    {(result.directory / 'native').resolve()}")
    print(f"Canonical: {result.canonical_plan.resolve()}")
    print(f"PS:        {'pass' if result.plan_generation_success else 'fail'}")
    print(f"SV:        {'pass' if result.static_validity else 'fail'}")
    print(f"GS:        {'pass' if result.symbolic_goal_success else 'fail'}")
    print("LLM calls: 0")
    if result.accepted_plan is None:
        print("Accepted:  no; no MuJoCo-ready plan was published")
    else:
        print(f"Accepted:  {result.accepted_plan.resolve()}")
    return 0 if (
        result.plan_generation_success and result.static_validity and result.symbolic_goal_success
    ) else 1


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
