"""Run the MRBTP baseline (Cai et al. 2025) on our scenarios and emit results.

This drives the authors' released code in ``third_party/MRBTP`` (clone it there and
``pip install -e`` it in a Python 3.10 env with its requirements). It ports each of our
scenarios into MRBTP's symbolic input (ground ``PlanningAction``s), runs the planner,
writes MRBTP's native outcome and serialized trees to ``outputs/mrbtp_results.json``.
MRBTP uses FAILURE-returning conditions while this project uses blocking guards,
so its native result is deliberately *not* passed through our simulator. The
comparison table labels it as ``mrbtp_native_v1`` rather than implying identical
execution semantics.

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
import subprocess
import sys
import time
from datetime import datetime, timezone
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


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _grounded_conditions(planner, start: frozenset[str]) -> list[frozenset[str]]:
    """Return expanded MRBTP conditions satisfied by the initial state.

    ``MAOBTP.bfs_planning`` does not return a success flag. Its actual success
    termination is ``start >= popped_condition``. The popped condition is not
    retained, but every popped condition was first inserted in at least one
    planning agent's ``expanded_condition_dict``. Checking those dictionaries is
    therefore a faithful post-run reconstruction; merely observing no timeout is
    not sufficient because the frontier may have been exhausted.
    """
    grounded: set[frozenset[str]] = set()
    for agent in getattr(planner, "planned_agent_list", None) or []:
        for condition in getattr(agent, "expanded_condition_dict", {}):
            condition = frozenset(condition)
            if start >= condition:
                grounded.add(condition)
    return sorted(grounded, key=lambda item: (len(item), sorted(item)))


def run_one(scenario, MAOBTP, PlanningAction, dump_trees: bool, time_limit: float) -> dict:
    """Run MRBTP and report native metrics.

    Native success requires all of: no timeout, an expanded condition grounded in
    the initial state, and extractable per-robot trees. This avoids the old and
    incorrect ``found = not timed_out`` shortcut. The result remains a native MRBTP
    metric because Condition semantics are incompatible with our blocking simulator.
    """
    ported = port_scenario(scenario)
    robot_ids, action_lists = _build_action_lists(ported, PlanningAction)

    planner = MAOBTP(verbose=False, start=frozenset(ported["start"]), env=None,
                     max_time_limit=time_limit)
    t0 = time.time()
    error = None
    try:
        planner.bfs_planning(frozenset(ported["goal"]), action_lists=action_lists)
    except Exception as exc:  # external baseline must produce an auditable record
        error = f"{type(exc).__name__}: {exc}"
    planning_time = time.time() - t0

    timed_out = getattr(planner, "expanded_time", 0.0) > 0.0
    grounded = [] if error else _grounded_conditions(planner, frozenset(ported["start"]))
    trees: dict[str, dict] = {}
    tree_error = None
    if grounded and not timed_out:
        try:
            btml_list = planner.get_btml_list()
            if len(btml_list) != len(robot_ids):
                raise RuntimeError(f"expected {len(robot_ids)} trees, got {len(btml_list)}")
            trees = {
                robot_ids[i]: _serialize_anytree(btml_list[i].anytree_root)
                for i in range(len(robot_ids))
            }
        except Exception as exc:
            tree_error = f"{type(exc).__name__}: {exc}"

    native_success = bool(grounded and trees and not timed_out and error is None and tree_error is None)
    if error:
        outcome = "error"
    elif timed_out:
        outcome = "timeout"
    elif not grounded:
        outcome = "frontier_exhausted"
    elif tree_error:
        outcome = "tree_extraction_error"
    else:
        outcome = "grounded_plan"

    if dump_trees and trees:
        print(json.dumps(trees, indent=2))

    return {
        "variant": "MAOBTP",
        "comparison_protocol": "mrbtp_native_v1",
        "metric_scope": "native_mrbtp",
        "outcome": outcome,
        "valid": native_success,
        "success": native_success,
        "goal_success": native_success,
        "native_success": native_success,
        "grounded_condition_found": bool(grounded),
        "grounded_conditions": [sorted(condition) for condition in grounded[:10]],
        "tree_count": len(trees),
        "expected_tree_count": len(robot_ids),
        "native_trees": trees,
        "timed_out": timed_out,
        "error": error or tree_error,
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
              f"outcome={r['outcome']} success={r['goal_success']}")

    out = resolve_project_path(args.output)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "mrbtp_native_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mrbtp_commit": _git_revision(REPO_ROOT / "third_party" / "MRBTP"),
        "runner_commit": _git_revision(REPO_ROOT),
        "time_limit_seconds": args.time_limit,
        "scenarios": results,
    }
    Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[mrbtp] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
