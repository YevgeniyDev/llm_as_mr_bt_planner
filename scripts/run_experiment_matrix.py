"""Run the fixed evaluation matrix and create an immutable result snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_mr_bt_planner.baselines import get_runner  # noqa: E402
from llm_mr_bt_planner.config import load_dotenv, resolve_project_path  # noqa: E402
from llm_mr_bt_planner.domain import load_scenario  # noqa: E402
from llm_mr_bt_planner.experiments import aggregate, run_experiment, to_latex_tables  # noqa: E402
from llm_mr_bt_planner.llm import get_client  # noqa: E402


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GITHUB_SHA", "unknown")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _checksums(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "checksums.sha256":
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def run_matrix(matrix_path: Path, snapshot_root: Path, paper_output: Path) -> Path:
    matrix_bytes = matrix_path.read_bytes()
    matrix = json.loads(matrix_bytes)
    scenarios = [load_scenario(resolve_project_path(path)) for path in matrix["scenarios"]]
    all_trials = []
    raw_conditions = []
    for condition in matrix["conditions"]:
        method = condition["method"]
        trials = int(condition.get("trials", matrix["trials"]))
        seeds = [int(matrix["seed_start"]) + index for index in range(trials)]
        if method == "mrbtp":
            client = None
        else:
            client = get_client(
                matrix["provider"], model=matrix["model"], temperature=float(matrix["temperature"])
            )
        print(f"[matrix] {condition['id']}: {len(scenarios)} scenarios x {trials} trials")
        report = run_experiment(
            scenarios,
            client,
            trials=trials,
            max_corrections=int(condition["max_corrections"]),
            max_ticks=int(matrix["max_ticks"]),
            include_hints=condition["hints"] == "full",
            suggest_producers=condition["feedback"] == "rich",
            samples=int(condition["samples"]),
            two_stage=bool(condition["two_stage"]),
            runner=get_runner(method),
            method=method,
            condition=condition["id"],
            seeds=seeds,
        )
        all_trials.extend(report.trials)
        raw_conditions.append(report.to_dict())

    commit = _git_sha()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = snapshot_root / f"{stamp}-{commit[:12]}"
    if snapshot.exists():
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {snapshot}")
    snapshot.mkdir(parents=True)
    rows = aggregate(all_trials)
    manifest = {
        "status": "complete",
        "protocol_version": matrix["protocol_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit,
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "provider_seed_semantics": "best-effort; OpenAI does not guarantee bit-identical outputs",
        "metric_scope_note": "LLM methods share the validator/simulator; MRBTP uses mrbtp_native_v1.",
    }
    _write_json(snapshot / "manifest.json", manifest)
    _write_json(snapshot / "matrix.json", matrix)
    _write_json(snapshot / "raw_results.json", {"conditions": raw_conditions})
    _write_json(snapshot / "aggregates.json", rows)
    (snapshot / "aggregates.csv").write_text(_csv(rows), encoding="utf-8")
    latex = to_latex_tables(all_trials) + "\n"
    (snapshot / "paper_tables.tex").write_text(latex, encoding="utf-8")
    paper_output.parent.mkdir(parents=True, exist_ok=True)
    paper_output.write_text(latex, encoding="utf-8")
    checksums = _checksums(snapshot)
    (snapshot / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()), encoding="utf-8"
    )
    for path in snapshot.iterdir():
        path.chmod(0o444)
    snapshot.chmod(0o555)
    return snapshot


def validate_matrix(matrix_path: Path) -> dict:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    required = {"protocol_version", "provider", "model", "temperature", "trials", "seed_start",
                "max_ticks", "scenarios", "conditions"}
    missing = sorted(required - matrix.keys())
    if missing:
        raise ValueError(f"Matrix is missing required keys: {missing}")
    condition_ids = [item["id"] for item in matrix["conditions"]]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("Condition ids must be unique")
    return {
        "protocol_version": matrix["protocol_version"],
        "conditions": len(condition_ids),
        "scenarios": len(matrix["scenarios"]),
        "planned_llm_trials": sum(
            int(item.get("trials", matrix["trials"])) * len(matrix["scenarios"])
            for item in matrix["conditions"] if item["method"] != "mrbtp"
        ),
        "planned_native_trials": sum(
            int(item.get("trials", matrix["trials"])) * len(matrix["scenarios"])
            for item in matrix["conditions"] if item["method"] == "mrbtp"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="experiments/protocol_v1.json")
    parser.add_argument("--snapshot-root", default="results/snapshots")
    parser.add_argument("--paper-output", default="generated/experiment_tables.tex")
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize without API calls.")
    args = parser.parse_args(argv)
    load_dotenv()
    summary = validate_matrix(resolve_project_path(args.matrix))
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    snapshot = run_matrix(
        resolve_project_path(args.matrix),
        resolve_project_path(args.snapshot_root),
        resolve_project_path(args.paper_output),
    )
    print(f"Immutable snapshot: {snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
