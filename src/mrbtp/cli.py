"""Command-line interface: single runs and multi-trial experiments.

    python -m mrbtp run --scenario data/scenario.json                       # default provider: openai
    python -m mrbtp run --scenario data/scenario.json --provider anthropic --model claude-opus-4-8
    python -m mrbtp experiment --scenario data/scenario.json --scenario data/scenario2.json \\
        --trials 5 --csv outputs/results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import load_dotenv, resolve_project_path, save_json, save_text
from .domain import load_scenario
from .execution import export_behaviortree_cpp_xml
from .experiments import run_experiment, to_csv, to_markdown_table
from .llm import get_client
from .plan import parse_plan
from .planner import run_planner
from .viz import plan_to_html


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    return args.func(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrbtp", description="LLM-guided multi-robot behavior-tree planning.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Generate, validate, and simulate one plan for one scenario.")
    run.add_argument("--scenario", default="data/scenario.json")
    run.add_argument("--output", default=None,
                     help="Result JSON path (default: outputs/run-<scenario>.json, derived from the scenario "
                          "file name so different scenarios don't overwrite each other).")
    run.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                     help="LLM provider (default openai; falls back to anthropic if OPENAI_API_KEY is unset).")
    run.add_argument("--model", default=None, help="Override the provider's default model.")
    run.add_argument("--max-corrections", type=int, default=4,
                     help="LLM self-correction rounds (0 = single-shot).")
    run.add_argument("--max-ticks", type=int, default=80)
    run.add_argument("--hints", choices=["none", "full"], default="none",
                     help="Inject precomputed dependency hints into the prompt (assisted mode).")
    run.add_argument("--feedback", choices=["minimal", "rich"], default="minimal",
                     help="'rich' adds candidate-producer suggestions to validator errors (assisted mode).")
    run.add_argument("--samples", type=int, default=1,
                     help="Best-of-N: sample N plans per generation, keep the first valid+successful "
                          "(needs --temperature > 0 to diversify, OpenAI only).")
    run.add_argument("--temperature", type=float, default=None,
                     help="OpenAI sampling temperature (default 0). Raise for best-of-N diversity.")
    run.add_argument("--two-stage", action="store_true",
                     help="Two-stage generation: LLM emits a validated per-robot action plan first, then "
                          "encodes it into behavior trees with synchronization. Improves pure-mode reliability.")
    run.add_argument("--export-bt", default=None, help="Also write BehaviorTree.CPP XML to this path.")
    run.add_argument("--viz", default=None,
                     help="Write a self-contained HTML report of the per-robot behavior trees to this path.")
    run.set_defaults(func=_cmd_run)

    exp = sub.add_parser("experiment", help="Run multiple trials across scenarios and aggregate metrics.")
    exp.add_argument("--scenario", action="append", dest="scenarios", required=True,
                     help="Scenario file (repeatable).")
    exp.add_argument("--provider", default="openai", choices=["openai", "anthropic"],
                     help="LLM provider (default openai; falls back to anthropic if OPENAI_API_KEY is unset).")
    exp.add_argument("--model", default=None)
    exp.add_argument("--trials", type=int, default=3)
    exp.add_argument("--max-corrections", type=int, default=4,
                     help="LLM self-correction rounds (0 = single-shot).")
    exp.add_argument("--max-ticks", type=int, default=80)
    exp.add_argument("--hints", choices=["none", "full"], default="none")
    exp.add_argument("--feedback", choices=["minimal", "rich"], default="minimal")
    exp.add_argument("--samples", type=int, default=1, help="Best-of-N plans per generation.")
    exp.add_argument("--temperature", type=float, default=None, help="OpenAI sampling temperature.")
    exp.add_argument("--two-stage", action="store_true", help="Two-stage generation (action plan -> BTs).")
    exp.add_argument("--json", dest="json_path", default="outputs/experiment.json")
    exp.add_argument("--csv", dest="csv_path", default=None)
    exp.add_argument("--markdown", dest="markdown_path", default=None)
    exp.set_defaults(func=_cmd_experiment)
    return parser


def _make_client(provider: str, model: str | None, temperature: float | None = None):
    """Build the LLM client, falling back to Anthropic when OpenAI is requested
    (or defaulted) but no OPENAI_API_KEY is available and an ANTHROPIC_API_KEY is.
    ``temperature`` applies to OpenAI only (Opus 4.8 has no temperature knob).
    """
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY"):
        print("[note] OPENAI_API_KEY not set; falling back to provider 'anthropic'.", file=sys.stderr)
        return get_client("anthropic", model=None)
    if provider == "openai":
        return get_client("openai", model=model, temperature=temperature)
    return get_client(provider, model=model)


def _default_output(scenario_path: str) -> str:
    return f"outputs/run-{Path(scenario_path).stem}.json"


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(resolve_project_path(args.scenario))
    client = _make_client(args.provider, args.model, args.temperature)
    result = run_planner(
        scenario, client,
        max_corrections=args.max_corrections, max_ticks=args.max_ticks,
        include_hints=(args.hints == "full"), suggest_producers=(args.feedback == "rich"),
        samples=args.samples, two_stage=args.two_stage,
    )

    output_path = resolve_project_path(args.output or _default_output(args.scenario))
    save_json(output_path, result.to_dict())

    if args.export_bt:
        bt_path = resolve_project_path(args.export_bt)
        save_text(bt_path, export_behaviortree_cpp_xml(parse_plan(result.plan)))

    if args.viz:
        viz_path = resolve_project_path(args.viz)
        meta = {
            "task": result.task_id,
            "provider/model": f"{result.provider}/{result.model}",
            "valid": result.valid,
            "goal reached": result.goal_success,
            "corrections": result.correction_rounds,
        }
        save_text(
            viz_path,
            plan_to_html(
                parse_plan(result.plan),
                title=f"BTs: {result.task_id}",
                meta=meta,
                trace=result.simulation.get("trace", []),
            ),
        )
        print(f"BT visualization: {viz_path}")

    _print_run_summary(result, output_path)
    return 0 if (result.valid and result.success) else 1


def _cmd_experiment(args: argparse.Namespace) -> int:
    scenarios = [load_scenario(resolve_project_path(path)) for path in args.scenarios]
    client = _make_client(args.provider, args.model, args.temperature)

    def progress(record):
        print(f"  [{record.scenario}] trial {record.trial}: "
              f"valid={record.valid} success={record.success} corrections={record.correction_rounds}")

    mode = "assisted" if (args.hints == "full" or args.feedback == "rich") else "pure"
    print(f"Running {len(scenarios)} scenario(s) x {args.trials} trial(s) on "
          f"{client.name}/{client.model} [mode={mode}, max_corrections={args.max_corrections}]")
    report = run_experiment(
        scenarios, client,
        trials=args.trials, max_corrections=args.max_corrections, max_ticks=args.max_ticks,
        include_hints=(args.hints == "full"), suggest_producers=(args.feedback == "rich"),
        samples=args.samples, two_stage=args.two_stage, on_trial=progress,
    )

    save_json(resolve_project_path(args.json_path), report.to_dict())
    if args.csv_path:
        save_text(resolve_project_path(args.csv_path), to_csv(report.trials))
    if args.markdown_path:
        save_text(resolve_project_path(args.markdown_path), to_markdown_table(report.trials))

    print("\nAggregated results:")
    print(to_markdown_table(report.trials))
    print(f"\nFull report: {resolve_project_path(args.json_path)}")
    return 0


def _print_run_summary(result, output_path: Path) -> None:
    print("LLM multi-robot BT planner")
    print("=" * 28)
    print(f"Task: {result.task_id}")
    print(f"Provider/model: {result.provider}/{result.model}")
    print(f"Valid: {_yn(result.valid)}")
    print(f"Goal reached: {_yn(result.goal_success)}")
    print(f"Correction rounds: {result.correction_rounds}")
    print(f"Result file: {output_path}")
    if result.validation_errors:
        print("\nValidation errors:")
        for error in result.validation_errors:
            print(f"- [{error['type']}] {error['message']}")
    elif result.simulation["errors"]:
        print("\nSimulation errors:")
        for error in result.simulation["errors"]:
            print(f"- [{error.get('type')}] {error}")


def _yn(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
