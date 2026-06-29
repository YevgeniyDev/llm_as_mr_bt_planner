"""Run the MRBTP baseline (Cai et al. 2025) on our scenarios and emit results.

This drives the authors' released code in ``third_party/MRBTP`` (clone it there and
``pip install -e`` it in a Python 3.10 env with its requirements). It ports each of our
scenarios into MRBTP's symbolic input (ground ``PlanningAction``s), runs the planner,
converts the per-robot trees back into our Plan JSON, and writes
``outputs/mrbtp_results.json`` keyed by ``task_id``. The main package's
``--method mrbtp`` then ingests that file and re-scores each plan with our
validator+simulator, so MRBTP lands in the same comparison table as every other method.

Usage (from the repo root, inside the MRBTP env):

    python scripts/run_mrbtp.py --scenario data/scenario.json --scenario data/scenario2.json

Notes:
  * MRBTP is symbolic (no LLM) - no API key needed.
  * If ``import mabtpg`` fails, the MRBTP deps are not installed; see third_party/MRBTP.
  * The exact AnyTreeNode arg layout is environment-dependent; if conversion looks off,
    inspect a serialized tree (``--dump-trees``) and adjust mrbtp_adapter._convert_node.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "MRBTP"))

from llm_mr_bt_planner.baselines.mrbtp_port import port_scenario  # noqa: E402
from llm_mr_bt_planner.config import resolve_project_path  # noqa: E402
from llm_mr_bt_planner.domain import load_scenario  # noqa: E402


def _import_mrbtp():
    try:
        from mabtpg.btp.multi_robot_optimal import MAOBTP  # noqa: F401
        from mabtpg.envs.gridenv.minigrid.planning_action import PlanningAction  # noqa: F401
    except Exception as error:  # pragma: no cover - depends on external install
        raise SystemExit(
            f"Could not import MRBTP ({error!r}). Install it first:\n"
            f"  cd third_party/MRBTP && pip install -e .  (Python 3.10, see requirements.txt)"
        ) from error
    return MAOBTP, PlanningAction


def _serialize_anytree(node) -> dict:
    """AnyTreeNode -> plain dict (used only for optional --dump-trees inspection)."""
    return {
        "node_type": getattr(node, "node_type", None),
        "cls_name": getattr(node, "cls_name", None),
        "args": list(getattr(node, "args", ()) or ()),
        "children": [_serialize_anytree(c) for c in getattr(node, "children", []) or []],
    }


def _build_action_lists(ported: dict, PlanningAction):
    action_lists = []
    robot_ids = []
    for agent in ported["agents"]:
        robot_ids.append(agent["robot"])
        action_lists.append([
            PlanningAction(
                name=a["name"], pre=set(a["pre"]), add=set(a["add"]),
                del_set=set(a["del_set"]), cost=a["cost"],
            )
            for a in agent["actions"]
        ])
    return robot_ids, action_lists


def run_one(scenario, MAOBTP, PlanningAction, dump_trees: bool, time_limit: float) -> dict:
    """Run MRBTP and report native metrics.

    Success is taken from the planner's own outcome, not from our blocking-guard
    simulator (which uses incompatible Condition semantics). MRBTP is sound and
    complete: it sets ``expanded_time`` only when it *times out*, so finishing within
    the budget on a solvable scenario means it found a goal-reaching plan, which is
    correct by construction. We therefore report ``valid``/``goal_success`` = "found",
    with planning time and expanded-condition count as MRBTP's native cost metrics.
    """
    ported = port_scenario(scenario)
    robot_ids, action_lists = _build_action_lists(ported, PlanningAction)

    planner = MAOBTP(verbose=False, start=frozenset(ported["start"]), env=None,
                     max_time_limit=time_limit)
    t0 = time.time()
    planner.bfs_planning(frozenset(ported["goal"]), action_lists=action_lists)
    planning_time = time.time() - t0

    timed_out = getattr(planner, "expanded_time", 0.0) > 0.0  # set only on timeout
    found = not timed_out

    if dump_trees:
        trees = {robot_ids[i]: _serialize_anytree(planner.get_btml_list()[i].anytree_root)
                 for i in range(len(robot_ids))}
        print(json.dumps(trees, indent=2))

    return {
        "variant": "MAOBTP",
        "valid": found,
        "success": found,
        "goal_success": found,
        "timed_out": timed_out,
        "planning_time": planning_time,
        "expanded_count": getattr(planner, "record_expanded_num", None),
        "feedback_rounds": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run MRBTP on our scenarios -> outputs/mrbtp_results.json")
    parser.add_argument("--scenario", action="append", dest="scenarios", required=True,
                        help="Scenario file (repeatable).")
    parser.add_argument("--output", default="outputs/mrbtp_results.json")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="Per-scenario planning budget in seconds (MRBTP times out past this).")
    parser.add_argument("--dump-trees", action="store_true", help="Print serialized trees for inspection.")
    args = parser.parse_args(argv)

    MAOBTP, PlanningAction = _import_mrbtp()

    results = {}
    for path in args.scenarios:
        scenario = load_scenario(resolve_project_path(path))
        print(f"[mrbtp] planning {scenario.task_id} ...")
        results[scenario.task_id] = run_one(
            scenario, MAOBTP, PlanningAction, args.dump_trees, args.time_limit
        )
        r = results[scenario.task_id]
        print(f"  done in {r['planning_time']:.3f}s, expanded={r['expanded_count']}, "
              f"found={r['goal_success']} (timed_out={r['timed_out']})")

    out = resolve_project_path(args.output)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"scenarios": results}, indent=2), encoding="utf-8")
    print(f"[mrbtp] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
