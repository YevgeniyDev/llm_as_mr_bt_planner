"""Command-line interface: single runs and multi-trial experiments.

    python -m llm_mr_bt_planner run --scenario data/scenario.json                       # default provider: openai
    python -m llm_mr_bt_planner run --scenario data/scenario.json --provider anthropic --model claude-opus-4-8
    python -m llm_mr_bt_planner experiment --scenario data/scenario.json --scenario data/scenario2.json \\
        --trials 5 --csv outputs/results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .baselines import METHOD_LABELS, get_runner
from .config import load_dotenv, resolve_project_path, save_json, save_text
from .domain import load_scenario
from .execution import SymbolicExecutionBackend, export_behaviortree_cpp_xml
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
    parser = argparse.ArgumentParser(prog="lmrbtp", description="LLM-guided multi-robot behavior-tree planning.")
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
    run.add_argument("--backend", choices=["symbolic", "mujoco"], default="symbolic",
                     help="Execution backend for the final plan. 'mujoco' replays it on real menagerie "
                          "robots in physics (needs the 'mujoco' extra + third_party/mujoco_menagerie).")
    run.add_argument("--mjcf", default=None,
                     help="Override the MuJoCo scene MJCF (default: scene auto-composed from the scenario).")
    run.add_argument("--menagerie", default=None,
                     help="Path to a mujoco_menagerie checkout (default: third_party/mujoco_menagerie).")
    run.add_argument("--render", action="store_true",
                     help="Open the interactive MuJoCo viewer while replaying (mujoco backend only).")
    run.add_argument("--physics", choices=["settle", "ik"], default="settle",
                     help="MuJoCo fidelity: 'settle' (Stage 1, scripted motion) or 'ik' (Stage 2, the "
                          "arms reach to parts via inverse kinematics and carry them).")
    run.add_argument("--recovery", choices=["off", "on"], default="off",
                     help="Run the execution-time recovery ladder (retry same robot -> reassign to another) "
                          "on the final plan. This is distinct from --max-corrections, which repairs the plan "
                          "with the LLM before execution. Needs a failure source (--inject-failures/--fail-prob).")
    run.add_argument("--max-retries", type=int, default=2,
                     help="Recovery Tier 1: times to retry a failed action on the same robot before reassigning.")
    run.add_argument("--reassign", choices=["on", "off"], default="on",
                     help="Recovery Tier 2: on failure-after-retries, hand the action to another capable robot.")
    run.add_argument("--inject-failures", default=None,
                     help="Deterministic failure model for the recovery ladder, e.g. 'pick_tool:1,mount_gear:2' "
                          "(fail those actions their first N executions). Runs with no LLM and no physics.")
    run.add_argument("--fail-prob", type=float, default=None,
                     help="Alternative to --inject-failures: fail each action with this probability (0..1).")
    run.add_argument("--fail-seed", type=int, default=0, help="Seed for --fail-prob (reproducible).")
    run.add_argument("--skills", choices=["off", "on"], default="off",
                     help="Inject Markdown-authored planning skills (skills/*.md) relevant to the scenario "
                          "into the LLM prompt. Additive and off by default so pure-mode prompts are unchanged.")
    run.add_argument("--skills-dir", default=None,
                     help="Directory of skill .md files (default: the repo's skills/).")
    run.set_defaults(func=_cmd_run)

    exp = sub.add_parser("experiment", help="Run multiple trials across scenarios and aggregate metrics.")
    exp.add_argument("--scenario", action="append", dest="scenarios", required=True,
                     help="Scenario file (repeatable).")
    exp.add_argument("--method", choices=["proposed", "flat", "hier", "mrbtp"], default="proposed",
                     help="Which method to evaluate: 'proposed' (this work) or a baseline "
                          "(flat=LLM-MARS-style, hier=LLM-as-BT-Planner-style, mrbtp=authors' code). "
                          "All methods are scored by the same validator+simulator.")
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
    exp.add_argument("--skills", choices=["off", "on"], default="off",
                     help="Inject Markdown planning skills (skills/*.md) into the proposed method's prompts "
                          "(selected per scenario). Baselines ignore it. Off by default.")
    exp.add_argument("--skills-dir", default=None, help="Directory of skill .md files (default: skills/).")
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
        return get_client("anthropic", model=model)
    if provider == "openai":
        return get_client("openai", model=model, temperature=temperature)
    return get_client(provider, model=model)


def _default_output(scenario_path: str) -> str:
    return f"outputs/run-{Path(scenario_path).stem}.json"


def _skills_dir(args: argparse.Namespace) -> Path | None:
    """The skills directory to use when --skills is on, else None."""
    if getattr(args, "skills", "off") != "on":
        return None
    from .skills import DEFAULT_SKILLS_DIR

    return resolve_project_path(args.skills_dir) if args.skills_dir else DEFAULT_SKILLS_DIR


def _skills_section(args: argparse.Namespace, scenario) -> str:
    """Load + select + render the Markdown skills for one scenario (or '')."""
    skills_dir = _skills_dir(args)
    if skills_dir is None:
        return ""
    from .skills import load_skills, render_skills_section, select_skills

    selected = select_skills(load_skills(skills_dir), scenario)
    if selected:
        print(f"[skills] Injected {len(selected)} skill(s) from {skills_dir}.", file=sys.stderr)
    else:
        print(f"[skills] No matching skills in {skills_dir}.", file=sys.stderr)
    return render_skills_section(selected)


def _cmd_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(resolve_project_path(args.scenario))
    client = _make_client(args.provider, args.model, args.temperature)
    result = run_planner(
        scenario, client,
        max_corrections=args.max_corrections, max_ticks=args.max_ticks,
        include_hints=(args.hints == "full"), suggest_producers=(args.feedback == "rich"),
        samples=args.samples, two_stage=args.two_stage,
        skills_section=_skills_section(args, scenario),
    )

    output_path = resolve_project_path(args.output or _default_output(args.scenario))
    payload = result.to_dict()

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

    # Execution-time recovery ladder (retry -> reassign), a stage separate from the
    # planning-time LLM self-correction that already ran inside run_planner().
    if args.recovery == "on":
        recovery = _run_recovery(args, result, scenario)
        if recovery is not None:
            payload["recovery"] = recovery

    if args.backend == "mujoco":
        mujoco = _run_mujoco(args, result, scenario)
        if mujoco is not None:
            payload["mujoco"] = mujoco

    save_json(output_path, payload)
    return 0 if (result.valid and result.success) else 1


def _build_oracle(args: argparse.Namespace):
    """Build a failure oracle for the recovery ladder from CLI flags, or None."""
    from .recovery import InjectedFailureOracle, StochasticFailureOracle, parse_injection_spec

    if args.inject_failures:
        return InjectedFailureOracle(parse_injection_spec(args.inject_failures))
    if args.fail_prob is not None:
        return StochasticFailureOracle(prob=args.fail_prob, seed=args.fail_seed)
    return None


def _run_recovery(args: argparse.Namespace, result, scenario) -> dict | None:
    """Replay the final plan under a failure oracle with retry/reassign recovery."""
    plan = parse_plan(result.plan)
    if plan.unparsable_trees or not plan.behavior_trees:
        print("\n[recovery] Skipped: the plan has no parsable behavior trees.", file=sys.stderr)
        return None
    oracle = _build_oracle(args)
    if oracle is None:
        print("\n[recovery] Skipped: pass --inject-failures or --fail-prob.", file=sys.stderr)
        return None

    from .recovery import RecoveryController

    controller = RecoveryController(
        oracle, max_retries=args.max_retries,
        allow_reassign=(args.reassign == "on"), max_ticks=args.max_ticks,
    )
    backend = SymbolicExecutionBackend(max_ticks=args.max_ticks, recovery=controller)
    exec_result = backend.execute(plan, scenario)
    recovery = exec_result.details["recovery"]

    print("\nExecution-time recovery ladder")
    print("-" * 30)
    print(f"Episodes: {recovery['episodes']}  |  "
          f"Goal reached after recovery: {_yn(exec_result.goal_success)}")
    for event in recovery["log"]:
        if event["tier"] == "retry":
            print(f"- retry #{event['attempt']}: {event['robot']} {event['action']}")
        else:
            arrow = f"-> {event['to_robot']}" if event["to_robot"] else "(no candidate)"
            print(f"- reassign: {event['robot']} {event['action']} {arrow}")
    if recovery["error"]:
        print(f"Unrecovered: {recovery['error']}")
    return recovery


def _run_mujoco(args: argparse.Namespace, result, scenario) -> dict | None:
    """Replay the final plan in MuJoCo physics; return its result dict (or None)."""
    plan = parse_plan(result.plan)
    if plan.unparsable_trees or not plan.behavior_trees:
        print("\n[mujoco] Skipped: the plan has no parsable behavior trees.", file=sys.stderr)
        return None
    from .execution.mujoco_backend import MujocoExecutionBackend

    backend = MujocoExecutionBackend(
        max_ticks=args.max_ticks,
        menagerie_dir=args.menagerie,
        mjcf=resolve_project_path(args.mjcf) if args.mjcf else None,
        render=args.render,
        fidelity=args.physics,
    )
    exec_result = backend.execute(plan, scenario)

    print("\nMuJoCo physics replay")
    print("-" * 28)
    d = exec_result.details
    print(f"Scene: {d['model']['nbody']} bodies, robots={', '.join(d['robots'])}")
    print(f"Physics actions executed: {d['physics_actions']} (sim time {d['sim_time']}s)")
    print(f"Goal reached in physics run: {_yn(exec_result.goal_success)}")
    if exec_result.errors:
        print(f"Execution errors: {exec_result.errors}")
    return exec_result.to_dict()


def _cmd_experiment(args: argparse.Namespace) -> int:
    scenarios = [load_scenario(resolve_project_path(path)) for path in args.scenarios]
    runner = get_runner(args.method)
    # MRBTP is not LLM-driven (it ingests the authors' code's results), so it needs no API key.
    client = None if args.method == "mrbtp" else _make_client(args.provider, args.model, args.temperature)

    def progress(record):
        print(f"  [{record.scenario}] trial {record.trial}: "
              f"valid={record.valid} success={record.success} corrections={record.correction_rounds}")

    mode = "assisted" if (args.hints == "full" or args.feedback == "rich") else "pure"
    engine = f"{client.name}/{client.model}" if client is not None else "authors' code (offline)"
    print(f"Running method '{METHOD_LABELS.get(args.method, args.method)}' on "
          f"{len(scenarios)} scenario(s) x {args.trials} trial(s) via {engine} "
          f"[mode={mode}, max_corrections={args.max_corrections}]")
    report = run_experiment(
        scenarios, client,
        trials=args.trials, max_corrections=args.max_corrections, max_ticks=args.max_ticks,
        include_hints=(args.hints == "full"), suggest_producers=(args.feedback == "rich"),
        samples=args.samples, two_stage=args.two_stage, skills_dir=_skills_dir(args), on_trial=progress,
        runner=runner, method=args.method,
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
