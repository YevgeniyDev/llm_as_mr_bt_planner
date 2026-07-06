"""Replay a previously generated plan in the MuJoCo viewer - no LLM/API calls.

Loads a result JSON (the `plan` saved by `lmrbtp run`) and the scenario it was
generated for, then opens the interactive viewer and plays the behavior trees on
the real menagerie robots. Use this to *watch* a plan as many times as you like
without spending API tokens.

    python scripts/view_mujoco.py --result outputs/run-gear-ik.json --scenario data/scenario.json
    python scripts/view_mujoco.py --result outputs/run-gear-ik.json --physics settle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when run straight from the repo (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_mr_bt_planner.domain import load_scenario  # noqa: E402
from llm_mr_bt_planner.execution.mujoco_backend import MujocoExecutionBackend  # noqa: E402
from llm_mr_bt_planner.plan import parse_plan  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch a saved plan in MuJoCo (no API calls).")
    ap.add_argument("--result", default="outputs/run-gear-ik.json",
                    help="Result JSON written by `lmrbtp run` (must contain a 'plan').")
    ap.add_argument("--scenario", default="data/scenario.json",
                    help="The scenario the plan was generated for.")
    ap.add_argument("--physics", choices=["settle", "ik"], default="ik",
                    help="'ik' = arms reach to parts (Stage 2); 'settle' = scripted motion (Stage 1).")
    ap.add_argument("--menagerie", default=None, help="Path to a mujoco_menagerie checkout.")
    ap.add_argument("--speed", type=float, default=0.4,
                    help="Seconds to pause per action (higher = slower, easier to watch).")
    args = ap.parse_args()

    result = json.loads(Path(args.result).read_text())
    plan = parse_plan(result["plan"])
    scenario = load_scenario(args.scenario)

    backend = MujocoExecutionBackend(
        render=True, fidelity=args.physics, realtime=args.speed,
        menagerie_dir=args.menagerie, hold_open=True,
    )
    out = backend.execute(plan, scenario)
    print(f"Goal reached in physics: {out.goal_success} | actions: {out.details['physics_actions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
